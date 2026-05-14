# Hardware-Coherent Cross-Device Memory Access

**Status**: Proposed

## Motivation

Warp currently enforces a strict rule: every array argument passed to `wp.launch()` must reside on the same device as the kernel launch target. If a user creates an array on the CPU and attempts to launch a GPU kernel that reads it, Warp raises a `RuntimeError`. This enforcement exists in `warp/_src/context.py::pack_arg`:

```python
# check device
if value.device != device:
    raise RuntimeError(
        f"Error launching kernel '{kernel.key}', trying to launch on "
        f"device='{device}', but input array for argument '{arg_name}' "
        f"is on device={value.device}."
    )
```

This restriction is correct on discrete-GPU systems (e.g., a workstation with a PCIe-connected NVIDIA GPU) where the GPU genuinely cannot dereference a pointer to unpinned CPU memory. However, a growing class of NVIDIA hardware uses **unified memory architectures** where the GPU _can_ directly access CPU memory. Some systems also let the CPU directly access GPU-resident CUDA managed memory, but that is a separate CUDA-reported capability and must not be inferred from ATS alone:

- **Grace C2C systems (GH200, GB200, DGX Spark)** -- Grace ARM CPU + Hopper or Blackwell GPU connected via NVLink Chip-to-Chip (C2C). These systems can report host-page-table ATS, allowing the GPU to access ordinary system memory. CPU direct access to GPU-resident CUDA managed memory depends on `cudaDevAttrDirectManagedMemAccessFromHost`; do not assume it from the product family name.
- **Jetson Orin and other limited Tegra systems** -- Integrated GPUs sharing the same DRAM as the CPU, but with a limited unified memory model where ordinary system allocations are not necessarily GPU-accessible.
- **Jetson Thor** -- Tegra Blackwell SoC with CUDA-reported ATS. On a Thor development kit tested with CUDA 13.0, the GPU can directly access ordinary system allocations (`malloc`, anonymous `mmap`, and file-backed `mmap`) and host-native atomics work, but CPU direct access to `cudaMalloc` memory is still not supported.
- **HMM-capable discrete systems** -- Linux kernel 6.1.24+ with Heterogeneous Memory Management (HMM) enabled allows software-coherent access to all system memory from PCIe GPUs, without requiring explicit CUDA allocation APIs.

On all systems where the CUDA device reports `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`, the strict `value.device != device` check is overly conservative and forces users into unnecessary `wp.copy()` or `.to(device)` calls that are both a performance penalty and an ergonomic burden. On HMM and ATS systems in particular, a plain `malloc`'d pointer is directly accessible from the GPU -- there is no need to copy data at all.

### User impact

A user on DGX Spark writing:

```python
data = wp.array([1.0, 2.0, 3.0], device="cpu")
wp.launch(my_kernel, dim=3, inputs=[data], device="cuda:0")
```

gets a `RuntimeError` even though the hardware can handle this directly. The user must write:

```python
data = wp.array([1.0, 2.0, 3.0], device="cpu")
data_gpu = data.to("cuda:0")  # unnecessary copy on ATS systems
wp.launch(my_kernel, dim=3, inputs=[data_gpu], device="cuda:0")
```

This is not just an inconvenience -- it defeats one of the primary benefits of unified-memory hardware, which is eliminating explicit data movement.

## Background: CUDA Unified Memory Paradigms

CUDA exposes unified memory capabilities through device attributes. The sections below group the relevant attribute combinations into four capability buckets so the implementation can reason about access rules mechanically, not by assuming behavior from a product family name. Three buckets have concrete platform examples in this document; the managed-only bucket is retained as a conservative fallback for the attribute combination where managed memory is fully shared but ordinary pageable system memory is not.

### Paradigm 1: Limited Unified Memory (Limited Tegra, Windows, WSL)

**Detection:** `cudaDevAttrConcurrentManagedAccess == 0`

Applies to Windows systems including WSL and to Tegra/Jetson devices whose CUDA attributes report limited managed access. Do not infer this from the Jetson family name alone: Jetson Thor tested with CUDA 13.0 reports `concurrentManagedAccess == 1`, `pageableMemoryAccess == 1`, and `pageableMemoryAccessUsesHostPageTables == 1`, so it does not fall into this paradigm.

Characteristics:
- Only memory explicitly allocated via `cudaMallocManaged` (or `cudaMallocFromPoolAsync` with `cudaMemAllocationTypeManaged`, or `__managed__` globals) behaves as unified memory.
- Managed memory starts in CPU physical memory, is bulk-migrated to the GPU when a kernel begins executing, and is bulk-migrated back on synchronization.
- The CPU must not access managed memory while the GPU is active.
- Oversubscription of GPU memory is not allowed.
- System allocations (`malloc`, `mmap`) are NOT GPU-accessible.

On limited/non-I/O-coherent Tegra specifically:
- `cudaHostRegister()` is not supported on non-I/O-coherent Tegra devices.
- `cudaMallocHost` produces uncached memory from the GPU's perspective on non-I/O-coherent Tegra.
- All memory physically resides in the same shared DRAM, but visibility is controlled by the CUDA driver.

### Paradigm 2: Full Unified Memory for CUDA-Managed Allocations Only

**Detection:** `cudaDevAttrConcurrentManagedAccess == 1` AND `cudaDevAttrPageableMemoryAccess == 0`

Characteristics:
- Memory allocated via `cudaMallocManaged` has full unified memory support (page-granularity migration, concurrent CPU/GPU access, oversubscription).
- System allocations (`malloc`, `mmap`) are still NOT GPU-accessible.
- This is an attribute-defined bucket. This document does not currently identify a specific tested platform for it, but the implementation should handle it separately because its access rules differ from both limited unified memory and HMM/ATS.

### Paradigm 3: Full Unified Memory with Software Coherency (HMM)

**Detection:** `cudaDevAttrPageableMemoryAccess == 1` AND `cudaDevAttrPageableMemoryAccessUsesHostPageTables == 0` AND `cudaDevAttrConcurrentManagedAccess == 1`

Available on Linux with kernel 6.1.24+ / 6.2.11+ / 6.3+ with HMM enabled. Can be verified via `nvidia-smi -q | grep Addressing` showing `HMM`.

Characteristics:
- ALL system-allocated memory (`malloc`, `mmap`, file-backed mappings) automatically behaves as unified memory. No CUDA allocation APIs are required.
- Migration happens via page faults at page granularity (software coherence).
- Oversubscription is allowed.
- `cudaMallocManaged` still works but is unnecessary for basic access -- `malloc` suffices.
- GPU `cudaMalloc` allocations are NOT CPU-accessible.

### Paradigm 4: Full System-Memory Access with Host Page Tables (ATS)

**Detection:** `cudaDevAttrPageableMemoryAccessUsesHostPageTables == 1` AND `cudaDevAttrPageableMemoryAccess == 1` AND `cudaDevAttrConcurrentManagedAccess == 1`

Available on Grace Hopper, Grace Blackwell (including DGX Spark), Jetson Thor, and future systems where CUDA reports pageable memory access through host page tables. `nvidia-smi -q` reports these systems as `Addressing Mode: ATS`.

Characteristics:
- ALL system-allocated memory is GPU-accessible (same as HMM).
- GPU-resident CUDA managed memory is CPU-accessible without migration only when `cudaDevAttrDirectManagedMemAccessFromHost == 1`. This attribute is independent of ATS and must be queried directly. It is false on Jetson Thor as tested with CUDA 13.0, and false on a DGX Spark / GB10 system tested with CUDA Toolkit 13.0 and driver 580.95.05.
- Native CPU-GPU atomics work when `cudaDevAttrHostNativeAtomicSupported == 1`. This is a separate capability bit and does not imply CPU access to `cudaMalloc` allocations.
- Host page tables are used for system-memory access. On systems with distinct CPU and GPU memory pools (Grace Hopper / Grace Blackwell), physical placement still matters for performance. On integrated SoCs such as Jetson Thor, the CPU and GPU share a single DRAM pool.
- ATS subsumes the system-memory access capabilities of HMM. When ATS is available, HMM is automatically disabled.

#### Observed Jetson Thor Behavior

The previous version of this document speculated that Jetson Thor would follow the limited Tegra model. Testing on a Jetson Thor development kit on 2026-05-11 showed otherwise:

- Platform: Linux `6.8.12-tegra`, CUDA Toolkit 13.0, Driver 13.0, GPU `NVIDIA Thor`, `sm_110`.
- `nvidia-smi -q` reports `Addressing Mode: ATS`.
- CUDA attributes: `integrated == 1`, `unifiedAddressing == 1`, `managedMemory == 1`, `concurrentManagedAccess == 1`, `pageableMemoryAccess == 1`, `pageableMemoryAccessUsesHostPageTables == 1`, `directManagedMemAccessFromHost == 0`, `hostNativeAtomicSupported == 1`, `canUseHostPointerForRegisteredMem == 1`.
- GPU kernels successfully read and wrote ordinary `malloc`, anonymous `mmap`, file-backed `mmap`, `cudaMallocHost`, `cudaHostRegister`, and `cudaMallocManaged` allocations.
- `cudaMemPrefetchAsync` succeeded for both managed memory and ordinary `malloc` memory.
- Direct CPU load/store of a `cudaMalloc` pointer faulted, matching `directManagedMemAccessFromHost == 0`.
- A stress test with overlapping CPU atomic increments and GPU `atomicAdd()` produced the exact expected result for ordinary `malloc`, pinned host memory, and managed memory.

The implementation must therefore treat "GPU can access system memory", "CPU can access GPU-resident CUDA managed memory", and "native CPU-GPU atomics work" as three independent capabilities.

#### Observed DGX Spark / GB10 Behavior

Testing on a DGX Spark-class GB10 system on 2026-05-14 showed that ATS and C2C
do not imply CPU direct access to GPU-resident CUDA managed memory:

- Platform: CUDA Toolkit 13.0, Driver 580.95.05, GPU `NVIDIA GB10`, `sm_121`.
- `nvidia-smi -q` reports `Addressing Mode: ATS` and `GPU C2C Mode: Enabled`.
- CUDA attributes queried directly through the CUDA driver:
  `managedMemory == 1`, `concurrentManagedAccess == 1`,
  `pageableMemoryAccess == 1`, `pageableMemoryAccessUsesHostPageTables == 1`,
  `directManagedMemAccessFromHost == 0`, and
  `hostNativeAtomicSupported == 1`.
- Warp reports the corresponding Python properties as
  `is_cpu_memory_access_from_gpu_supported == True`,
  `is_gpu_memory_access_from_cpu_supported == False`, and
  `is_cpu_gpu_atomic_supported == True`.

This corrects the earlier assumption that DGX Spark / Grace Blackwell systems
should be classified as "bidirectional ATS" for managed-memory host access.
For Phase 1, the relevant launch feature remains GPU access to CPU arrays via
`pageableMemoryAccess`; CPU direct access to GPU-resident managed memory remains
attribute-gated and is not used to validate Warp default CUDA arrays.

### Summary of Access Rules by Paradigm

| Allocation type | Limited (Tegra/Win) | Full Managed Only | HMM (Software) | ATS system-memory only (Thor/GB10 observed) | ATS with direct managed host access |
|---|---|---|---|---|---|
| `malloc` / system | CPU only | CPU only | CPU + GPU | CPU + GPU | CPU + GPU |
| `mmap` / file-backed | CPU only | CPU only | CPU + GPU | CPU + GPU | CPU + GPU |
| `cudaMallocManaged` | Limited shared | Full shared | Full shared | Full shared; no direct host access to GPU-resident pages unless attribute reports it | Full shared with direct host access to GPU-resident pages |
| `cudaMallocHost` (pinned) | CPU + GPU (zero-copy) | CPU + GPU | CPU + GPU | CPU + GPU | CPU + GPU |
| `cudaHostRegister` | Device-dependent | CPU + GPU | CPU + GPU | CPU + GPU | CPU + GPU |
| `cudaMalloc` | GPU only | GPU only | GPU only | GPU only | GPU only for Warp default arrays |

### Performance Considerations on ATS Systems

Even when all system memory is GPU-accessible on ATS systems, physical placement can still matter for performance. On systems with distinct CPU and GPU memory pools, a GPU kernel repeatedly reading data physically resident in CPU LPDDR5X over NVLink C2C pays the C2C latency on every cache miss. CUDA provides mechanisms to control placement:

1. **Explicit prefetch** (`cudaMemPrefetchAsync`): Stream-ordered migration of a memory region to a specified device. Works on any allocation including system `malloc`. This is the primary tool for optimizing data placement.

2. **Access counter migration**: On ATS systems, the GPU hardware tracks access frequency to remote pages. When enabled via `cudaMemAdviseSetAccessedBy`, pages that the GPU accesses frequently are automatically migrated to GPU-local memory. Available for system-allocated memory starting with CUDA 12.4. Does not apply to file-backed `mmap` allocations.

3. **Placement hints** (`cudaMemAdvise`):
   - `cudaMemAdviseSetPreferredLocation(device)` -- encourages data to stay on the specified device.
   - `cudaMemAdviseSetReadMostly` -- allows read replication across devices.
   - `cudaMemAdviseSetAccessedBy(device)` -- enables access counter migration on ATS systems; establishes direct mappings on other systems.

On integrated ATS systems such as Jetson Thor, CPU and GPU memory share one DRAM pool, so prefetch may still succeed but may not provide a useful "closer" placement. Automatic prefetch should therefore remain disabled on integrated GPUs.

**Important performance caveat**: On host-page-table ATS systems with distinct CPU and GPU memory pools, CPU writes to GPU-resident memory may be expensive even if a future platform reports direct managed-memory host access. ARM (Grace) caches require all memory operations to pass through the cache hierarchy, so writing to GPU-resident memory can cause cache misses that pull data across C2C before writing. The recommended pattern is: write to CPU-resident memory, let the GPU read it remotely or prefetch it.

### Comparison: DGX Spark / GB10 vs. Jetson Thor

Both DGX Spark / GB10 and Jetson Thor use Blackwell-generation GPUs, but their memory architectures differ and the CUDA attributes must still be queried independently:

| Aspect | DGX Spark / GB10 (observed) | Jetson Thor (Tegra Blackwell) |
|---|---|---|
| CPU-GPU interconnect | NVLink C2C (high bandwidth, coherent) | On-chip SoC fabric |
| ATS available | Yes | Yes (`nvidia-smi` reports ATS) |
| Coherency model | Host-page-table ATS with distinct CPU/GPU memory pools | Host-page-table ATS for system memory on an integrated SoC |
| `malloc` GPU-accessible | Yes | Yes |
| CPU direct access to GPU-resident CUDA managed memory | No (`directManagedMemAccessFromHost == 0` on CUDA 13.0 / driver 580.95.05) | No (`directManagedMemAccessFromHost == 0` on CUDA 13.0) |
| Native CPU-GPU atomics | Yes | Yes for host-visible memory |
| Memory topology | Grace LPDDR5X + Blackwell HBM (NUMA) | Single shared DRAM pool |
| Unified memory paradigm | ATS system-memory access (Paradigm 4) | ATS system-memory access (Paradigm 4) |
| Best default allocator | System allocator (`malloc`) for shared CPU/GPU data | System allocator (`malloc`) for CPU-produced GPU-readable data; `cudaMalloc` for GPU-private data |

This means the implementation must query capabilities independently instead of assuming a single "ATS" behavior. DGX Spark / GB10 and Jetson Thor can launch GPU kernels directly over CPU arrays, but CPU kernels still cannot dereference Warp default CUDA arrays.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | `wp.launch()` must remove the per-argument device check so cross-device array arguments are always passed through to the hardware | Must | Core behavior change, zero launch overhead |
| R2 | Provide an opt-in verification mode (`warp.config.verify_launch_array_access`) that restores device-access checking with clear diagnostics | Must | Debuggability for users who hit CUDA illegal memory access errors; compatible with CUDA graph capture |
| R3 | Provide `wp.prefetch()` API for explicit data migration hints | Should | Performance optimization for HMM / host-page-table ATS |
| R4 | Optional automatic prefetch in `wp.launch()` for cross-device arrays on coherent systems | Could | Convenience, but needs careful defaults |
| R5 | `wp.copy()` should skip staging buffers when direct access is available between devices | Could | Performance optimization, marked as TODO in current code |

**Non-goals:**
- Changing the default allocator strategy (e.g., using `cudaMallocManaged` by default on limited Tegra systems). Allocator selection is a separate concern.
- Changing CUDA graph capture semantics. Phase 1 supports using `verify_launch_array_access` during graph capture, but does not add new cross-device synchronization, placement, or capture-time migration behavior beyond the same access checks used for ordinary launches.
- Automatically determining the optimal physical placement for every array. This is a performance tuning concern best left to the user via hints.
- Proactively detecting and warning about cross-device launches at `wp.launch()` time. The hardware enforces access rules; the verification mode is available for diagnosis when needed.

## Design

### CUDA Version Compatibility

Warp currently supports building with CUDA 12.0 through 13.2. The default toolkit distributed on PyPI is CUDA 12.9. Full support for this feature set on CUDA 12.0 through 12.7 is not a goal, but the feature must degrade cleanly (compile without errors, disable gracefully at runtime) on older toolkit versions.

**Device attribute queries (Phases 1, 2, 3, 5):** All device attributes used in this plan are `CUdevice_attribute` enum values present in `cuda.h` since CUDA 8.0 or 9.2 at the latest:

| Attribute | Enum value | Present since | Phase |
|---|---|---|---|
| `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` | 88 | CUDA 8.0 | 1 |
| `CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST` | 101 | CUDA 9.2 | 1 |
| `CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED` | 86 | CUDA 8.0 | 1 |
| `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES` | 100 | CUDA 9.2 | 2 |
| `CU_DEVICE_ATTRIBUTE_INTEGRATED` | 18 | CUDA 2.0 | 3 |
| `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS` | 89 | CUDA 8.0 | 5 |

All predate Warp's minimum of CUDA 12.0, so no `#if CUDA_VERSION` compile-time guards are needed for attribute queries. The attributes are queried via `cuDeviceGetAttribute`, which Warp already loads dynamically via `cuGetProcAddress` at version 2000. The driver returns 0 for any attribute the hardware does not support, which is the correct "feature not available" default.

**`cuMemPrefetchAsync` (Phase 2):** This driver API has two versions:

| API version | Signature | Toolkit requirement | Driver requirement |
|---|---|---|---|
| v1 (version 8000) | `(CUdeviceptr, size_t, CUdevice, CUstream)` | CUDA 8.0+ | CUDA 8.0+ driver |
| v2 (version 12080) | `(CUdeviceptr, size_t, CUmemLocation, unsigned int, CUstream)` | CUDA 12.8+ | CUDA 12.8+ driver |

In CUDA 13.0 headers, `cuMemPrefetchAsync` is `#define`'d to `cuMemPrefetchAsync_v2`. Warp must handle both via `cuGetProcAddress` dynamic dispatch, following the existing pattern used for `cuMemcpyBatchAsync`. The v1 API is sufficient for all planned use cases. The v2 API adds NUMA node targeting but is not required. When compiled with CUDA 12.0--12.7, only v1 is available; this is fine. See Phase 2 for the full dispatch implementation.

**Summary by toolkit version:**

| Feature | CUDA 12.0 -- 12.7 | CUDA 12.8 -- 12.9 (PyPI default) | CUDA 13.0+ |
|---|---|---|---|
| Phase 1 (cross-device launch) | Full support | Full support | Full support |
| Phase 2 (prefetch) | v1 API only | v2 API available | v2 API available |
| Phase 3 (auto-prefetch) | Full support (uses Phase 2 API) | Full support | Full support |
| Phase 4 (`wp.copy()` optimization) | Full support | Full support | Full support |
| Phase 5 (public allocation-aware access API) | Full support | Full support | Full support |

No phase requires a minimum toolkit version beyond CUDA 12.0. Degradation on older toolkits only affects which `cuMemPrefetchAsync` signature is available, which is handled transparently by the dynamic dispatch.

### Overview: What Each Phase Introduces

Each phase introduces only the device attributes, native functions, and Python APIs that it directly consumes. No phase adds speculative API surface for a future phase to use.

| Phase | What it delivers | Attributes introduced | Native functions introduced |
|---|---|---|---|
| 1 | Remove device check from `wp.launch()`, add verification mode, redesign `can_access()`, add allocation-aware launch verification for Warp-owned arrays | Native: `pageable_memory_access`, `direct_managed_mem_access_from_host`, `host_native_atomic_supported`; Python: `is_cpu_memory_access_from_gpu_supported`, `is_gpu_memory_access_from_cpu_supported`, `is_cpu_gpu_atomic_supported` | Three `wp_cuda_device_get_*` accessors |
| 2 | `wp.prefetch()` for explicit data placement | `pageable_memory_access_uses_host_page_tables` (to distinguish HMM from host-page-table ATS for warning/no-op behavior) | `wp_cuda_mem_prefetch_async` |
| 3 | Auto-prefetch in `wp.launch()` | `is_integrated` (to avoid pointless prefetches on shared-DRAM SoCs) | None |
| 4 | `wp.copy()` staging-buffer optimization | None (reuses Phase 1 access predicates) | None |
| 5 | Public allocation-aware access API and managed/pinned/custom allocation refinements | `concurrent_managed_access` (to distinguish limited vs. full managed memory) | None |

### Phase 1: Cross-Device Launch Support

**Goal:** Remove the per-argument device check from `wp.launch()` so that cross-device array arguments are passed straight through to the hardware. On systems with unified system-memory access (HMM or host-page-table ATS), this means GPU kernels can directly consume CPU arrays with zero launch overhead and zero friction. On systems where the access is illegal, the CUDA runtime produces an error. A verification mode (`warp.config.verify_launch_array_access`) is available to diagnose such errors with clear, argument-level diagnostics before the kernel runs, including during CUDA graph capture.

This phase delivers five things: (a) query three new device attributes, (b) redesign `Device.can_access()` as a conservative device-level/default-allocation query, (c) remove the default `pack_arg()` same-device check, (d) add `warp.config.verify_launch_array_access` with allocation-aware verification for Warp-owned arrays where Warp can identify the allocator, including pinned CPU arrays on CUDA devices with UVA, and (e) add tests and advanced user documentation for the CPU/GPU memory access model.

#### 1a. Query Device Attributes

Three CUDA device attributes are needed:

- **`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`** -- answers "can this GPU access ordinary `malloc`'d CPU memory?" This is the attribute that determines whether a Warp `wp.array(device="cpu")` (backed by `malloc` via `CpuDefaultAllocator`) can be dereferenced by a GPU kernel. Without it, we cannot distinguish a system where the GPU can read CPU pointers (HMM, host-page-table ATS, Jetson Thor) from one where it cannot (discrete GPU without HMM, limited Tegra, Windows).

- **`CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST`** -- answers "can the CPU directly access CUDA managed memory resident on the GPU without migration?" This does not imply that Warp `wp.array(device="cuda:0")` allocations backed by `cuMemAlloc` via `CudaDefaultAllocator` can be passed to CPU kernels. Phase 1 exposes the capability as a device property, but `Device.can_access()` and `verify_launch_array_access` remain conservative for CPU-to-CUDA Warp arrays because Warp's built-in CUDA arrays are not CUDA managed-memory allocations.

- **`CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED`** -- answers "do CPU-GPU atomics work natively across the interconnect?" On systems where this is true (DGX Spark / GB10 and Jetson Thor as tested), a GPU `atomicAdd` targeting a CPU-resident address produces correct results via hardware coherency. On HMM systems, the same operation can silently produce wrong results -- the GPU atomic hits a page backed by CPU physical memory without hardware coherency for atomic operations. Exposing this as a device property lets users and downstream tools (e.g., documentation, `wp.prefetch()` heuristics) reason about atomic safety. This attribute must be treated independently from `direct_managed_mem_access_from_host`.

The first attribute is needed to gate the GPU-accessing-CPU branch in `can_access()` and launch verification. The second and third are exposed as queryable device properties for users who need to reason about managed-memory host access and cross-device atomic safety. `can_access()` and launch verification do not use `direct_managed_mem_access_from_host` for CPU-to-CUDA default arrays because those are not CUDA managed-memory allocations.

**Native layer changes (`warp/native/warp.cu`, `warp/native/warp.h`)**

Add three fields to `DeviceInfo`:

```cpp
struct DeviceInfo {
    // ... existing fields ...
    int pageable_memory_access = 0;
    int direct_managed_mem_access_from_host = 0;
    int host_native_atomic_supported = 0;
};
```

Query them during device enumeration, alongside the existing `cuDeviceGetAttribute` calls:

```cpp
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].pageable_memory_access,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device));
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].direct_managed_mem_access_from_host,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, device));
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].host_native_atomic_supported,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device));
```

**CUDA version requirements:** All three attributes are enum values in `CUdevice_attribute` that have been present since well before CUDA 12.0 (Warp's minimum supported CUDA toolkit version): `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` (= 88, CUDA 8.0+), `CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST` (= 101, CUDA 9.2+), `CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED` (= 86, CUDA 8.0+). No `#if CUDA_VERSION` guard is needed. They are queried via `cuDeviceGetAttribute`, which Warp already loads dynamically via `cuGetProcAddress` at version 2000. The driver will return 0 for any attribute the hardware does not support, which is the correct default (feature not available).

Add accessor functions, following the existing pattern of `wp_cuda_device_is_uva()`:

```cpp
// warp.h
WP_API int wp_cuda_device_get_pageable_memory_access(int ordinal);
WP_API int wp_cuda_device_get_direct_managed_mem_access_from_host(int ordinal);
WP_API int wp_cuda_device_get_host_native_atomic_supported(int ordinal);

// warp.cu
int wp_cuda_device_get_pageable_memory_access(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pageable_memory_access;
    return 0;
}

int wp_cuda_device_get_direct_managed_mem_access_from_host(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].direct_managed_mem_access_from_host;
    return 0;
}

int wp_cuda_device_get_host_native_atomic_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].host_native_atomic_supported;
    return 0;
}
```

**Python layer changes (`warp/_src/context.py`)**

Register the new ctypes bindings in `Runtime.__init__`, following the existing native binding pattern:

```python
self.core.wp_cuda_device_get_pageable_memory_access.argtypes = [ctypes.c_int]
self.core.wp_cuda_device_get_pageable_memory_access.restype = ctypes.c_int
self.core.wp_cuda_device_get_direct_managed_mem_access_from_host.argtypes = [ctypes.c_int]
self.core.wp_cuda_device_get_direct_managed_mem_access_from_host.restype = ctypes.c_int
self.core.wp_cuda_device_get_host_native_atomic_supported.argtypes = [ctypes.c_int]
self.core.wp_cuda_device_get_host_native_atomic_supported.restype = ctypes.c_int
```

Add documented Python `Device` properties using Warp's CPU/GPU terminology rather than CUDA attribute names. These properties are `False` for CPU devices and meaningful for GPU devices:

```python
        is_cpu_memory_access_from_gpu_supported (bool): Indicates whether GPU kernels on this device can
            directly access ordinary CPU memory. ``False`` for CPU devices.
        is_gpu_memory_access_from_cpu_supported (bool): Indicates whether CPU code can directly access
            CUDA managed memory resident on this device without migration. Does not imply Warp default CUDA
            arrays are CPU-accessible. ``False`` for CPU devices.
        is_cpu_gpu_atomic_supported (bool): Indicates whether native atomic operations between CPU and GPU
            memory are supported for this device. ``False`` for CPU devices.
```

Add the properties to `Device.__init__` for CUDA devices:

```python
# Unified CPU/GPU memory access capability properties
self.is_cpu_memory_access_from_gpu_supported = (
    runtime.core.wp_cuda_device_get_pageable_memory_access(ordinal) > 0
)
self.is_gpu_memory_access_from_cpu_supported = (
    runtime.core.wp_cuda_device_get_direct_managed_mem_access_from_host(ordinal) > 0
)
self.is_cpu_gpu_atomic_supported = (
    runtime.core.wp_cuda_device_get_host_native_atomic_supported(ordinal) > 0
)
```

For the CPU device branch, set all three Python properties to `False`.

Do not add aliases or duplicate convenience properties. Phase 1 exposes exactly one Python property per capability. The native C++ fields and accessors intentionally keep CUDA attribute-derived names so maintainers can map them directly to CUDA documentation.

#### 1b. Redesign `Device.can_access()`

The pre-Phase 1 implementation had only a same-context check, with a TODO acknowledging that access needs to be redesigned in terms of both devices and resources:

```python
def can_access(self, other):
    # TODO: this function should be redesigned in terms of (device, resource).
    # - a device can access any resource on the same device
    # - a CUDA device can access pinned memory on the host
    # - a CUDA device can access regular allocations on a peer device if peer access is enabled
    # - a CUDA device can access mempool allocations on a peer device if mempool access is enabled
    other = self.runtime.get_device(other)
    if self.context == other.context:
        return True
    else:
        return False
```

Replace with:

```python
def can_access(self, other):
    # TODO: this function should be redesigned in terms of (device, resource).
    # - a device can access any resource on the same device
    # - a CUDA device can access CPU memory when the device supports it
    # - a CUDA device can access regular CUDA allocations on a peer device if peer access is enabled
    # - a CUDA device can access mempool allocations on a peer device if mempool access is enabled
    other = self.runtime.get_device(other)

    if self.context == other.context:
        return True

    if self.is_cuda and other.is_cpu:
        return self.is_cpu_memory_access_from_gpu_supported

    if self.is_cpu and other.is_cuda:
        # Warp's default CUDA arrays are device allocations, not CUDA managed-memory allocations.
        return False

    if self.is_cuda and other.is_cuda:
        return is_peer_access_enabled(other, self)

    return False
```

**Notes on `can_access()` implementation details:**

- The `is_peer_access_enabled()` call in the GPU-to-GPU branch is the module-level function in `warp/_src/context.py`. It lives in the same module as `Device`, so no additional import is needed.
- The `self.context == other.context` check catches same-device access and same-context aliases. It also covers CPU-to-CPU access because the CPU device has no CUDA context.
- The TODO in the existing code mentions that access depends on the _resource_ (allocation type), not just the device pair. `Device.can_access()` remains a conservative device-level check: it returns `True` only when standard/default allocations on the other device are accessible. This avoids turning a device-only API into an allocation-specific oracle.
- `can_access()` is NOT called in the default launch path. It is invoked by APIs and user code that need the device-level/default-allocation answer. The launch verification path uses a private array-aware helper so CUDA default allocations and CUDA memory-pool allocations can be checked against the correct access state.

#### 1c. Remove the Launch Device Check

Change `pack_arg()`.

**Scope:** This change only removes the device check for `array` arguments. Other device-bound types (textures, volumes, hash grids) have their own device checks later in `pack_arg()` and remain strict (`value.device != device`). Relaxing those is out of scope for Phase 1: textures and volumes have GPU-side handles (CUDA texture objects, device pointers to internal structures) that may not be accessible cross-device even on systems with ATS system-memory access.

The `pack_arg()` function is called for both forward and adjoint arguments through `pack_args()`. The removed check applies to both paths, so cross-device arrays work in backward passes on capable hardware.

Replace:

```python
# check device
if value.device != device:
    raise RuntimeError(
        f"Error launching kernel '{kernel.key}', trying to launch on "
        f"device='{device}', but input array for argument '{arg_name}' "
        f"is on device={value.device}."
    )
```

With:

```python
# Verify array allocation accessibility (opt-in diagnostic mode).
# By default, no check is performed and the pointer is passed
# straight through to the hardware.  On systems with unified
# system-memory access (HMM or host-page-table ATS) this is correct;
# on discrete GPUs without
# HMM the kernel will fault with CUDA_ERROR_ILLEGAL_ADDRESS if the
# access is invalid.  Enable warp.config.verify_launch_array_access to get a
# clear Python error *before* the kernel runs.
if (
    warp.config.verify_launch_array_access
    and value.device != device
    and not _is_array_accessible_from_device(value, device)
):
    raise RuntimeError(
        f"Error launching kernel '{kernel.key}', trying to launch on device='{device}', "
        f"but input array for argument '{arg_name}' is on device={value.device}, "
        f"whose array allocation is not accessible from '{device}'. Move the array to "
        f"'{device}', enable the required peer/coherent access, or disable warp.config.verify_launch_array_access "
        f"only if this launch is valid for the hardware and allocation type."
    )
```

The private helper checks the actual Warp array allocation where Warp can determine it:

```python
def _is_array_accessible_from_device(value, device):
    device = runtime.get_device(device)
    value_device = value.device

    if device.context == value_device.context:
        return True
    if device.is_cuda and value_device.is_cpu:
        if value.pinned and device.is_uva:
            return True
        return device.is_cpu_memory_access_from_gpu_supported
    if device.is_cpu and value_device.is_cuda:
        return False
    if device.is_cuda and value_device.is_cuda:
        allocator = getattr(value, "_allocator", None)
        if isinstance(allocator, CudaMempoolAllocator):
            return is_mempool_access_enabled(value_device, device)
        if isinstance(allocator, CudaDefaultAllocator):
            return is_peer_access_enabled(value_device, device)
        return False
    return False
```

**Design rationale:** The previous design called `can_access()` on every array argument of every launch. Even though `can_access()` is cheap (property lookups), it adds up in hot launch paths with many array arguments. Removing the check entirely by default means `pack_arg()` does strictly less work for the normal path: when `verify_launch_array_access` is `False`, the branch short-circuits before any device comparison, allocation inspection, or native peer/mempool query. The verification mode gates the check behind a single boolean test, which the branch predictor will eliminate after the first few calls.

#### 1d. Verification Mode Config Flag

Add to `warp/config.py`:

```python
verify_launch_array_access: bool = False
"""Enable kernel launch argument accessibility checking.

When enabled, Warp checks whether array arguments are accessible from the launch
device before passing their pointers to the kernel. For Warp-owned arrays, this
uses the array's allocation type where Warp can determine it. This restores
pre-launch diagnostics for mixed-device launches that depend on hardware CPU/GPU
memory access support.

Unlike ``verify_cuda``, this setting can be used during CUDA graph capture
because checks run before each launch is recorded. For cross-GPU graph capture,
enable peer or memory-pool access with Warp APIs before capture begins.

Note: Enabling this flag impacts performance.
"""
```

**When to use:** If a user on a discrete GPU (without HMM) accidentally passes a CPU array to a GPU kernel, the kernel will fault with `CUDA_ERROR_ILLEGAL_ADDRESS`. This error is asynchronous and can corrupt the CUDA context, requiring a process restart. The recommended workflow is:

1. Observe the CUDA error.
2. Set `warp.config.verify_launch_array_access = True`.
3. Re-run. The clear Python `RuntimeError` identifies which kernel and which argument caused the mismatch, before the kernel ever launches.
4. Fix the code, disable verification.

`verify_launch_array_access` is compatible with CUDA graph capture because the checks happen before each launch is recorded and do not depend on post-launch CUDA error polling. Cross-GPU graph captures still depend on the correct access mode being enabled before capture: peer access for default CUDA allocations and memory-pool access for CUDA memory-pool allocations. Warp records peer and memory-pool access state when `wp.set_peer_access_enabled()` and `wp.set_mempool_access_enabled()` are called so graph-capture verification does not need to issue CUDA access-query calls while capture is active.

#### 1e. User-facing Documentation

Add `docs/deep_dive/memory_access.rst` and link it from the docs index and device guide. The page should explain the CPU/GPU memory model for advanced users, including:

- The three public `Device` properties and that they are `False` for CPU devices.
- How to guard mixed CPU/GPU launches using the capability properties.
- How `Device.can_access()` relates to CPU/GPU capability properties and GPU/GPU peer access, and how launch verification differs by checking the actual Warp array allocation where possible.
- How and when to use `wp.config.verify_launch_array_access`, including its CUDA graph capture compatibility.
- Why direct loads/stores do not imply CPU/GPU atomic safety.

#### Behavior matrix after Phase 1

Default mode (`verify_launch_array_access = False`): no Python-level checking. The hardware decides.

| Launch device | Array device | Discrete GPU (no HMM) | HMM system | Jetson Thor | Host-page-table ATS (DGX Spark / GB10 observed) |
|---|---|---|---|---|---|
| `cuda:0` | `cuda:0` | OK (same device) | OK | OK | OK |
| `cuda:0` | `cpu` (pageable) | **CUDA fault** | **OK** (HMM) | **OK** (ATS system memory) | **OK** (ATS) |
| `cuda:0` | `cpu` (pinned) | **OK** (UVA pinned zero-copy) | **OK** | **OK** | **OK** |
| `cpu` | `cuda:0` | **Segfault** | **Segfault** | **Segfault** | **Segfault** for Warp default arrays |
| `cuda:0` | `cuda:1` | CUDA fault / OK (peer or mempool access, depending on allocation) | CUDA fault / OK (peer or mempool access, depending on allocation) | N/A on single-GPU Thor | CUDA fault / OK (peer or mempool access, depending on allocation) |

Verification mode (`verify_launch_array_access = True`): each Warp-owned array argument is checked with allocation-aware launch verification where Warp can determine the allocator.

| Launch device | Array device | Discrete GPU (no HMM) | HMM system | Jetson Thor | Host-page-table ATS (DGX Spark / GB10 observed) |
|---|---|---|---|---|---|
| `cuda:0` | `cuda:0` | OK (same device) | OK | OK | OK |
| `cuda:0` | `cpu` (pageable) | **RuntimeError** | **OK** (HMM) | **OK** (ATS system memory) | **OK** (ATS) |
| `cuda:0` | `cpu` (pinned) | **OK** (UVA pinned zero-copy) | **OK** | **OK** | **OK** |
| `cpu` | `cuda:0` | **RuntimeError** | **RuntimeError** | **RuntimeError** | **RuntimeError** for Warp default arrays |
| `cuda:0` | `cuda:1` | RuntimeError / OK (peer or mempool access, depending on allocation) | RuntimeError / OK (peer or mempool access, depending on allocation) | N/A on single-GPU Thor | RuntimeError / OK (peer or mempool access, depending on allocation) |

On a standard discrete-GPU workstation without HMM, users who pass a CPU array to a GPU kernel will get a CUDA fault instead of the current Python `RuntimeError`. This is a deliberate tradeoff: zero overhead in the launch path for all users, at the cost of a less friendly error for an incorrect program. The verification mode restores the friendly error for diagnosis.

#### Stream selection for cross-device launches

When an array on device A is passed to a kernel on device B, Warp must ensure proper synchronization. The current `wp.launch()` already selects a stream based on the launch device. The kernel launch happens on that stream, and since the pointer is passed through without checking, the hardware coherency or HMM page fault mechanism handles visibility on capable systems.

However, if the array was _produced_ by a kernel on a different stream, the caller is responsible for synchronizing (e.g., via `wp.synchronize()` or stream events). This is the same requirement as for same-device multi-stream usage and does not need special handling here.

### Phase 2: Explicit Prefetch API (`wp.prefetch()`)

**Goal:** Provide a public API for users to request migration of array data to a specific device, without copying. This is a performance optimization for HMM and ATS systems where data is accessible across processors but performance can depend on physical placement.

This phase introduces one additional device attribute and one new native function.

#### New device attribute: `pageable_memory_access_uses_host_page_tables`

**CUDA attribute:** `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`

This attribute distinguishes HMM (software coherency) from host-page-table ATS. Both have `pageable_memory_access == 1`, but prefetch behavior differs:
- On host-page-table ATS with distinct CPU/GPU memory pools, prefetch can migrate physical pages via hardware DMA with cache-line coherency. On integrated systems such as Jetson Thor, prefetch may succeed but may not improve placement because CPU and GPU share the same DRAM.
- On HMM, prefetch triggers software page migration with TLB shootdowns. It works but has higher overhead and different failure modes.

The `wp.prefetch()` implementation needs this to provide accurate diagnostics (e.g., warning when prefetching on a system where it may cause page-fault storms) and to choose the right native API call path.

**Native layer:** Same pattern as Phase 1 -- add to `DeviceInfo`, query during enumeration, add accessor function. `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES` (= 100) has been present since CUDA 9.2, well before Warp's minimum of CUDA 12.0. No compile-time guard needed.

```cpp
// DeviceInfo addition
int pageable_memory_access_uses_host_page_tables = 0;

// Query
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].pageable_memory_access_uses_host_page_tables,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device));

// Accessor
WP_API int wp_cuda_device_get_pageable_memory_access_uses_host_page_tables(int ordinal);
```

**Python layer:**

```python
self.pageable_memory_access_uses_host_page_tables = (
    runtime.core.wp_cuda_device_get_pageable_memory_access_uses_host_page_tables(ordinal) > 0
)
```

#### New native function: `wp_cuda_mem_prefetch_async`

```cpp
// warp.h
WP_API int wp_cuda_mem_prefetch_async(void* ptr, size_t size_in_bytes,
                                       int device_ordinal, void* stream);
```

This wraps `cuMemPrefetchAsync` (driver API). The `device_ordinal` can be `-1` to indicate the CPU as the target (maps to `CU_DEVICE_CPU` in the driver API).

**CUDA API versioning:** The `cuMemPrefetchAsync` driver API has two versions:

- **v1** (CUDA 8.0+, version 8000): `cuMemPrefetchAsync(CUdeviceptr, size_t, CUdevice dstDevice, CUstream)` -- takes a simple `CUdevice` ordinal for the destination.
- **v2** (CUDA 12.8+, version 12080): `cuMemPrefetchAsync(CUdeviceptr, size_t, CUmemLocation location, unsigned int flags, CUstream)` -- takes a `CUmemLocation` struct (supports NUMA node targeting) and flags.

In CUDA 13.0 headers, `cuMemPrefetchAsync` is `#define`'d to `cuMemPrefetchAsync_v2`. Warp dynamically loads driver entry points via `cuGetProcAddress`, so the implementation must handle both versions:

```cpp
// In init_cuda_driver(), load the prefetch entry point:
#if CUDA_VERSION >= 12080
if (driver_version >= 12080)
    get_driver_entry_point("cuMemPrefetchAsync", 12080, &(void*&)pfn_cuMemPrefetchAsync_v2);
else
#endif
    get_driver_entry_point("cuMemPrefetchAsync", 8000, &(void*&)pfn_cuMemPrefetchAsync_v1);
```

The `wp_cuda_mem_prefetch_async` wrapper dispatches to whichever version was loaded:

```cpp
int wp_cuda_mem_prefetch_async(void* ptr, size_t size_in_bytes,
                                int device_ordinal, void* stream)
{
    CUdeviceptr devPtr = (CUdeviceptr)ptr;
    CUstream hStream = (CUstream)stream;

#if CUDA_VERSION >= 12080
    if (pfn_cuMemPrefetchAsync_v2) {
        CUmemLocation location;
        if (device_ordinal >= 0) {
            location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            location.id = device_ordinal;
        } else {
            location.type = CU_MEM_LOCATION_TYPE_HOST;
            location.id = 0;
        }
        return check_cu(pfn_cuMemPrefetchAsync_v2(devPtr, size_in_bytes,
                                                    location, 0, hStream)) ? 0 : -1;
    }
#endif
    if (pfn_cuMemPrefetchAsync_v1) {
        CUdevice dstDevice = (device_ordinal >= 0)
            ? g_devices[device_ordinal].device
            : CU_DEVICE_CPU;
        return check_cu(pfn_cuMemPrefetchAsync_v1(devPtr, size_in_bytes,
                                                    dstDevice, hStream)) ? 0 : -1;
    }
    return -1;  // prefetch not available
}
```

This pattern follows the existing convention in `cuda_util.cpp` (e.g., `cuMemcpyBatchAsync` at line 234 uses the same `#if CUDA_VERSION >= 12080` / `driver_version >= 12080` pattern).

**Compile-time / runtime compatibility matrix for Phase 2:**

| Toolkit used to build Warp | Runtime driver | Prefetch available? | API version used |
|---|---|---|---|
| CUDA 12.0 -- 12.7 | Any 12.0+ | Yes | v1 (CUdevice) |
| CUDA 12.8+ | Driver < 12.8 | Yes | v1 (CUdevice) |
| CUDA 12.8+ | Driver >= 12.8 | Yes | v2 (CUmemLocation) |

The v1 API is fully sufficient for the `wp.prefetch()` use case (migrate to a device or to the CPU). The v2 API adds NUMA node targeting which is not needed initially but is available when both toolkit and driver support it.

**Disabling prefetch on older CUDA:** If Warp is compiled with CUDA 12.0 -- 12.7, only the v1 entry point is loaded. The v1 API works for `cudaMallocManaged` allocations on all systems, and also for system-allocated (`malloc`) memory on HMM / host-page-table ATS systems. The Python `wp.prefetch()` wrapper should catch errors from the driver (e.g., if the pointer is not in a prefetchable region) and emit a warning rather than raising, since prefetch is a performance hint.

Implementation notes:
- `cuMemPrefetchAsync` works on any pointer that falls within a unified memory region -- including plain `malloc` on HMM / host-page-table ATS systems, `cuMemAllocManaged` allocations, and `cuMemAlloc` allocations on systems where device allocations are host-accessible.
- On systems where the pointer is not in a prefetchable region, the call returns an error. The Python wrapper should catch this and either warn or silently ignore, since prefetch is a hint.
- The prefetch is stream-ordered: it begins after all prior operations on the stream complete and finishes before any subsequent operations on the stream begin.

#### Python API

```python
def prefetch(
    array: warp.array,
    device: DeviceLike = None,
    stream: Stream | None = None,
):
    """Request asynchronous migration of ``array`` data toward ``device``.

    On systems with host-page-table ATS or software coherency (HMM),
    this issues a ``cuMemPrefetchAsync`` to migrate the
    array's physical pages closer to the specified device. The array
    remains valid and accessible from any device during and after the
    prefetch.

    On systems without unified memory support for the array's allocation
    type, this function is a no-op and emits a warning.

    This is a performance hint, not a correctness requirement. Kernels
    will produce correct results regardless of whether prefetch is
    called.

    Args:
        array: The array whose data should be migrated.
        device: The target device. If ``None``, uses the default device.
        stream: The stream on which to order the prefetch. If ``None``,
            uses the current stream on the target device.
    """
```

#### Usage example

```python
# On DGX Spark / GB10 (host-page-table ATS system):
data = wp.array(np.random.randn(1000000), dtype=wp.float32, device="cpu")

# Prefetch to GPU before a compute-heavy kernel
wp.prefetch(data, device="cuda:0")
wp.launch(heavy_compute_kernel, dim=data.size, inputs=[data], device="cuda:0")

# Prefetch back to CPU before CPU-side post-processing
wp.prefetch(data, device="cpu")
result = data.numpy()
```

### Phase 3: Optional Automatic Prefetch in `wp.launch()` (Future)

**Goal:** When a cross-device array argument is detected in `pack_arg()` on a coherent system, optionally issue a prefetch automatically before the kernel launch. This is a convenience optimization that should be off by default.

This phase introduces one additional device attribute.

#### New device attribute: `is_integrated`

**CUDA attribute:** `CU_DEVICE_ATTRIBUTE_INTEGRATED`

This attribute indicates whether the GPU is physically integrated into the same chip/package as the CPU (Tegra/Jetson SoCs). It is already queried in the native layer but stored in a local variable `device_attribute_integrated` used only for the IPC check. This phase promotes it to a stored `DeviceInfo` field exposed to Python.

The auto-prefetch heuristic needs this because prefetch on an integrated GPU is usually pointless -- the CPU and GPU share the same physical DRAM, so there is no "closer" location to migrate data to. Jetson Thor testing showed that `cuMemPrefetchAsync` can succeed for ordinary `malloc` memory, but that does not make automatic prefetch useful. Without this attribute, the auto-prefetch code would waste time issuing low-value prefetch calls on every integrated-GPU kernel launch with cross-device arrays.

#### Config flag

Add to `warp/config.py`:

```python
auto_prefetch = False
"""When True and launching a kernel on a device that can access memory on
another device (e.g., GPU accessing CPU memory on an HMM or ATS system),
automatically prefetch cross-device array arguments to the launch device
before the kernel begins. Default is False because automatic prefetch is
not always beneficial -- for example, streaming read-once access patterns
are better served by remote access over NVLink C2C than by migrating
the data."""
```

#### Implementation in `pack_arg()`

After the device accessibility check passes (Phase 1), and before returning the packed argument:

```python
if value.device != device and warp.config.auto_prefetch:
    # Skip prefetch on integrated GPUs -- CPU and GPU share the same
    # DRAM, so migration is meaningless.
    if not device.is_integrated:
        try:
            stream_handle = device.stream.cuda_stream if device.is_cuda else 0
            device_ordinal = device.ordinal if device.is_cuda else -1
            runtime.core.wp_cuda_mem_prefetch_async(
                value.ptr, value.capacity, device_ordinal, stream_handle
            )
        except Exception:
            pass  # Prefetch is best-effort
```

#### Why off by default

Automatic prefetch has several cases where it hurts more than it helps:

1. **Read-once data**: If a kernel reads an array once and never again, prefetching (which may involve a DMA transfer or page-table work) can be slower than direct access.
2. **CPU-produced, GPU-consumed streaming data**: If the CPU is continuously writing to a buffer that the GPU reads, prefetching would fight with the CPU's writes. The CUDA documentation explicitly recommends keeping such data CPU-resident and letting the GPU read remotely.
3. **Small arrays**: The overhead of issuing a prefetch (driver call, DMA setup) exceeds the benefit for small transfers.
4. **Multiple kernels**: If multiple kernels on different devices access the same array, prefetching to one device may pessimize access from another.

Users who want automatic prefetch can enable it globally via `warp.config.auto_prefetch = True` or per-launch by calling `wp.prefetch()` explicitly before the launch.

### Phase 4: Improve `wp.copy()` for Coherent Systems (Future)

**Goal:** When source and destination arrays are on different devices and the destination-side copy kernel can directly access the source allocation, skip the staging buffer logic in `wp.copy()`.

No new attributes or native functions. This should reuse the Phase 1 access predicates, but the final implementation must be allocation-aware for arrays where Warp can determine the allocator. `Device.can_access()` is sufficient for the device-level/default-allocation cases; CUDA memory-pool allocations need the same allocation-aware distinction used by `verify_launch_array_access`.

The current `wp.copy()` implementation has a TODO for this:

```python
# Copying between different devices requires contiguous arrays.  If the arrays
# are not contiguous, we must use temporary staging buffers for the transfer.
# TODO: We can skip the staging if device access is enabled.
```

On systems where `is_cpu_memory_access_from_gpu_supported` is true, the GPU can directly read non-contiguous CPU memory, so staging is unnecessary for destination-device copy kernels that read CPU source arrays. The reverse direction for Warp default CUDA arrays remains invalid, and multi-GPU CUDA arrays need allocation-specific peer or memory-pool access checks. The GPU-reading-CPU fix is straightforward:

```python
if src.device != dest.device:
    # If direct access is available, we can copy non-contiguous arrays
    # without staging, using a kernel on the destination device.
    if _is_array_accessible_from_device(src, dest.device):
        # Launch a copy kernel on the destination device that reads
        # directly from the source array's memory.
        pass  # Implementation details TBD
    else:
        # Existing staging buffer logic for non-contiguous arrays...
        ...
```

This is a performance optimization and not required for correctness -- the existing staging approach works correctly on all systems.

### Phase 5: Public Allocation-Aware Access Checks (Future)

**Goal:** Extend the internal allocation-aware checks introduced in Phase 1 into a public, fine-grained access model that can answer questions about specific arrays, including pinned CPU memory, future managed-memory allocators, custom allocators, and externally wrapped allocations.

This phase introduces one additional device attribute.

#### New device attribute: `concurrent_managed_access`

**CUDA attribute:** `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`

This attribute distinguishes the "limited" unified memory paradigm (limited Tegra, Windows -- `concurrent_managed_access == 0`) from the "full" paradigms (`concurrent_managed_access == 1`). On limited systems, `cudaMallocManaged` allocations bulk-migrate and cannot be concurrently accessed by CPU and GPU. On full systems, managed allocations support page-granularity migration with concurrent access.

The public allocation-aware API needs this because it must answer: "if this specific array was allocated with `cudaMallocManaged` (a future Warp managed allocator), can the GPU access it concurrently with the CPU?" The answer depends on this attribute.

#### Allocator tracking

Phase 1 launch verification already distinguishes the two built-in CUDA
allocator classes when checking GPU-to-GPU array arguments:

- `CudaDefaultAllocator` uses CUDA peer access (`set_peer_access_enabled()` /
  `is_peer_access_enabled()`).
- `CudaMempoolAllocator` uses CUDA memory-pool access
  (`set_mempool_access_enabled()` / `is_mempool_access_enabled()`).

Future allocation-aware work remains useful for exposing CPU pinned-memory
behavior through a public API, and for future managed-memory allocators, custom
allocators, and externally wrapped allocations. Currently, Warp has four allocator
classes:

- `CpuDefaultAllocator` -- uses `wp_alloc_host` (wraps `malloc`/`calloc`)
- `CpuPinnedAllocator` -- uses `wp_alloc_pinned` (wraps `cudaMallocHost`)
- `CudaDefaultAllocator` -- uses `wp_alloc_device_default` (wraps `cuMemAlloc`)
- `CudaMempoolAllocator` -- uses `wp_alloc_device_async` (wraps `cuMemAllocAsync`)

On a discrete GPU without HMM:
- Pinned CPU allocations (`CpuPinnedAllocator`) ARE GPU-accessible through UVA, and Phase 1 launch verification accepts Warp-owned pinned CPU arrays when `device.is_uva` is true. `Device.can_access()` remains a device-level/default-allocation query and still does not distinguish pinned CPU arrays from ordinary CPU arrays.
- Default CPU allocations (`CpuDefaultAllocator`) are NOT GPU-accessible.
- Both CUDA allocators produce GPU-only memory.

For CPU pinned memory and future managed-memory allocators, prefer a public
array-level API such as `array.is_accessible_from(device)` over adding optional
allocation parameters to `Device.can_access()`:

```python
def is_accessible_from(self, device):
    """..."""
    device = runtime.get_device(device)

    # GPU accessing CPU memory without HMM/host-page-table ATS.
    if device.is_cuda and self.device.is_cpu:
        if device.is_cpu_memory_access_from_gpu_supported:
            return True
        # Pinned allocations are always GPU-accessible via UVA.
        if self.pinned and device.is_uva:
            return True
        return False
```

Phase 1 uses this logic internally for `wp.launch()` verification. Phase 5 would
make the same allocation-aware answer available to user code without overloading
`Device.can_access()` with allocation-specific parameters.

## Testing Strategy

### Phase 1 tests

Add a test module `warp/tests/cuda/test_unified_memory.py` (registered in `warp/tests/unittest_suites.py`) and extend `warp/tests/test_graph.py` for CUDA graph capture coverage.

**Attribute query tests (run on all hardware):**
- Verify `is_cpu_memory_access_from_gpu_supported`, `is_gpu_memory_access_from_cpu_supported`, and `is_cpu_gpu_atomic_supported` are `bool` for CUDA devices and `False` for CPU devices.
- Do not assert that `is_cpu_gpu_atomic_supported` implies `is_gpu_memory_access_from_cpu_supported`; Jetson Thor reports native CPU-GPU atomics while still rejecting direct CPU access to `cudaMalloc` memory.

**`can_access()` tests (run on all hardware):**
- `device.can_access(device)` is always `True` for every device.
- CPU-to-CPU: always `True`.
- GPU-to-CPU and CPU-to-GPU: assert the result is consistent with Warp default allocation rules:
  - If `is_cpu_memory_access_from_gpu_supported` is `True`, GPU-to-CPU should be `True`.
  - CPU-to-GPU should be `False` for Warp default CUDA arrays, even if managed-memory host access is supported.
- GPU-to-GPU peer: enable peer access, verify `can_access()` returns `True`. Disable, verify `False`.

**Cross-device launch tests (hardware-dependent, skip on incapable systems):**
- On systems where `cuda_device.is_cpu_memory_access_from_gpu_supported` is `True`: allocate a CPU array, launch a GPU kernel that reads and writes it, verify results match expected values.
- On CUDA devices with `device.is_uva`: allocate pinned CPU arrays and verify GPU kernels can read from and write to them with `warp.config.verify_launch_array_access = True`.
- Test with output arrays (not just inputs).
- Test with multi-dimensional arrays with non-trivial strides.

**Verification mode tests (run on all hardware):**
- With `warp.config.verify_launch_array_access = True` on a discrete GPU without HMM: verify that launching with a CPU array raises `RuntimeError` (not a CUDA fault).
- With `warp.config.verify_launch_array_access = True` on an HMM / host-page-table ATS system: verify that GPU launches with CPU arrays still succeed (no false positive).
- With `warp.config.verify_launch_array_access = False` (default): verify that no Python-level device check occurs (cross-device arrays are passed through without error from `pack_arg()`).
- With `warp.config.verify_launch_array_access = True` during CUDA graph capture: capture and replay a same-device CUDA launch successfully.
- On multi-GPU systems with a peer-access-supported pair: allocate with CUDA memory pools disabled, enable peer access before capture, pass an array from the source GPU to a kernel launched on the peer GPU with `verify_launch_array_access = True`, capture and replay the graph, and verify the results. Skip cleanly when no peer-access pair exists.
- On multi-GPU systems with a memory-pool-access-supported pair: allocate with CUDA memory pools enabled, enable memory-pool access before capture, pass an array from the source GPU to a kernel launched on the peer GPU with `verify_launch_array_access = True`, capture and replay the graph, and verify the results.
- On multi-GPU systems, test default CUDA allocations and CUDA memory-pool allocations separately:
  - Default CUDA allocations should be accepted by `verify_launch_array_access` when peer access is enabled.
  - CUDA memory-pool allocations should be accepted by `verify_launch_array_access` when memory-pool access is enabled, even if peer access is disabled.
  - CUDA memory-pool allocations should be rejected by `verify_launch_array_access` when memory-pool access is disabled, even if peer access is enabled.

### Phase 2 tests (prefetch)

- On HMM / host-page-table ATS systems: prefetch a CPU array to GPU, launch a kernel, verify correctness.
- On systems without HMM / host-page-table ATS: calling `wp.prefetch()` should not raise (no-op or warning).
- Test stream ordering: prefetch then kernel on same stream, verify results.
- Test prefetch back to CPU: prefetch to GPU, then prefetch to CPU, verify CPU access.

### Phase 3 tests (auto-prefetch)

- Enable `warp.config.auto_prefetch`, launch cross-device kernel, verify correctness.
- Verify auto-prefetch is not issued on integrated GPUs (may require mocking or checking driver call counts).

### CI considerations

- The existing CI may not have HMM, ATS, Jetson Thor, or DGX Spark / GB10 hardware. Tests that require specific paradigms should use `unittest.skipUnless` based on the device attributes queried in Phase 1.
- Tests that only query attributes (Phase 1 attribute and `can_access()` invariant tests) should run on all hardware.
- Consider adding a CI label or tag for "unified memory" tests so they can be selectively run on appropriate hardware.

### Device compatibility matrix for test expectations

| Test scenario | Discrete (no HMM) | Discrete (HMM) | Host-page-table ATS with direct managed host access | Jetson Orin / limited Tegra | Jetson Thor | DGX Spark / GB10 observed |
|---|---|---|---|---|---|---|
| GPU can access CPU arrays | No | Yes | Yes | No | Yes | Yes |
| CPU can access Warp default GPU arrays | No | No | No | No | No | No |
| CPU can access GPU-resident CUDA managed memory | No | No | Yes | No | No | No |
| Native CPU-GPU atomics on host-visible memory | No | No | Yes | Device-dependent | Yes | Yes |
| Cross-device launch GPU->CPU array (default) | CUDA fault | OK | OK | CUDA fault | OK | OK |
| Cross-device launch CPU->GPU array (default) | Segfault | Segfault | Segfault for Warp default arrays | Segfault | Segfault | Segfault for Warp default arrays |
| Cross-device launch GPU->CPU array (verify mode) | RuntimeError | OK | OK | RuntimeError | OK | OK |
| Cross-device launch CPU->GPU array (verify mode) | RuntimeError | RuntimeError | RuntimeError for Warp default arrays | RuntimeError | RuntimeError | RuntimeError for Warp default arrays |
| `wp.prefetch()` for CPU arrays | No-op / warning | Yes (SW) | Yes (HW) | No-op / warning | Accepted; low expected benefit on integrated DRAM | Yes (HW) |
