CPU/GPU Memory Access
=====================

.. currentmodule:: warp

Warp arrays are associated with a device such as ``"cpu"`` or ``"cuda:0"``.
On many systems, an array can only be accessed by kernels launched on the same
device.  Newer CPU/GPU systems can be more flexible: a GPU may be able to read
ordinary CPU memory directly, and some systems can directly access CUDA managed
memory resident on the GPU without an explicit copy.

This page describes how Warp exposes those hardware capabilities and how to use
them when writing advanced code.


The Basic Rule
--------------

The launch device determines where a kernel runs.  The array device describes
where the array allocation lives:

.. code:: python

    cpu_array = wp.zeros(1024, dtype=float, device="cpu")
    gpu_array = wp.zeros(1024, dtype=float, device="cuda:0")

    wp.launch(kernel, dim=cpu_array.size, inputs=[cpu_array], device="cpu")
    wp.launch(kernel, dim=gpu_array.size, inputs=[gpu_array], device="cuda:0")

The same-device pattern works on all supported systems.  Passing an array from
one device to a kernel running on another device depends on the capabilities of
the device that performs the access.


Capability Properties
---------------------

Each :class:`Device` exposes three CPU/GPU memory access properties.  They are
``False`` on CPU devices and meaningful on GPU devices:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Property
     - Meaning
   * - ``device.is_cpu_memory_access_from_gpu_supported``
     - GPU kernels launched on this device can directly access ordinary CPU
       memory, including arrays allocated with ``device="cpu"``.
   * - ``device.is_gpu_memory_access_from_cpu_supported``
     - CPU code can directly access CUDA managed memory resident on this
       device without migration. This does not imply that Warp's default CUDA
       arrays are CPU-accessible.
   * - ``device.is_cpu_gpu_atomic_supported``
     - Native atomic operations between CPU and GPU memory are supported by the
       hardware.

The properties are directional.  A system can allow GPU access to CPU memory
without allowing CPU access to GPU-resident managed memory.

.. code:: python

    device = wp.get_device("cuda:0")

    if device.is_cpu_memory_access_from_gpu_supported:
        print("GPU kernels can access CPU arrays directly")

    if device.is_gpu_memory_access_from_cpu_supported:
        print("CPU code can access GPU-resident managed memory directly")

    if device.is_cpu_gpu_atomic_supported:
        print("CPU/GPU atomics are supported")


Common Hardware Models
----------------------

The exact values are reported by the CUDA driver and may vary by platform,
driver, kernel, and GPU generation.  The following table summarizes the models
advanced users commonly need to reason about:

.. list-table::
   :header-rows: 1
   :widths: 24 25 25 26

   * - System model
     - GPU access to CPU arrays
     - CPU access to GPU-resident managed memory
     - CPU/GPU atomics
   * - Discrete GPU without HMM
     - Usually no
     - Usually no
     - Usually no
   * - Discrete GPU with Linux HMM
     - Yes
     - Usually no
     - Usually no
   * - Jetson Thor-style ATS
     - Yes
     - Platform-dependent for managed memory
     - Yes, when reported by the driver
   * - Grace Hopper / Grace Blackwell-style coherent memory
     - Yes
     - Yes for managed memory
     - Yes, when reported by the driver

HMM stands for Heterogeneous Memory Management.  ATS stands for Address
Translation Services.  Warp does not require users to classify the platform
manually.  Query the :class:`Device` properties and branch on the behavior your
program needs.


Launching GPU Kernels With CPU Arrays
-------------------------------------

When ``device.is_cpu_memory_access_from_gpu_supported`` is true, a GPU kernel can
directly read or write a CPU array:

.. code:: python

    device = wp.get_device("cuda:0")
    a = wp.zeros(1024, dtype=float, device="cpu")

    if device.is_cpu_memory_access_from_gpu_supported:
        wp.launch(kernel, dim=a.size, inputs=[a], device=device)
    else:
        a_gpu = a.to(device)
        wp.launch(kernel, dim=a_gpu.size, inputs=[a_gpu], device=device)

This can avoid explicit copies on HMM and coherent CPU/GPU systems.  If the
capability is false and the kernel actually dereferences the CPU pointer, CUDA
will report a runtime error such as an illegal memory access.


Accessing GPU Data From CPU Code
--------------------------------

CPU access to GPU-resident managed memory is a separate capability:

.. code:: python

    device = wp.get_device("cuda:0")
    if device.is_gpu_memory_access_from_cpu_supported:
        ...

Warp's default CUDA arrays are device allocations, not CUDA managed-memory
allocations.  For those arrays, use an explicit copy before CPU code reads or
writes the data:

.. code:: python

    a = wp.zeros(1024, dtype=float, device=device)
    a_cpu = a.to("cpu")
    wp.launch(cpu_kernel, dim=a_cpu.size, inputs=[a_cpu], device="cpu")

Do not infer CPU access to GPU-resident memory from GPU access to CPU memory.
Some systems support the first direction but not the second.


Using ``Device.can_access()``
------------------------------

The method :meth:`Device.can_access` answers whether code running on one device
can access allocations associated with another device:

.. code:: python

    launch_device = wp.get_device("cuda:0")
    array_device = wp.get_device("cpu")

    if launch_device.can_access(array_device):
        ...

For GPU kernels accessing CPU arrays, this method uses
``is_cpu_memory_access_from_gpu_supported``.  For CPU code accessing CUDA arrays,
it returns ``False`` for Warp's default CUDA allocations.  For GPU/GPU pairs, it
reflects CUDA peer access state for default CUDA allocations.  Memory pool
allocations have separate access controls described in :ref:`mempool_access`.

``Device.can_access()`` is a device-level query.  It does not inspect a specific
array allocation, so it does not report pinned CPU arrays separately from
ordinary CPU arrays.  Launch verification, described below, uses an internal
array-aware check for ``warp.array`` arguments.  When a cross-device
``warp.array`` uses an allocation whose accessibility Warp cannot verify, launch
verification fails closed instead of assuming the pointer is safe to use.


.. _launch_verification:

Launch Verification
-------------------

By default, Warp passes array pointers through to :func:`launch` without a
pre-launch same-device check.  This keeps the launch path lightweight and allows
hardware-supported mixed CPU/GPU launches to work.

If you want a clear Python error before the kernel runs, enable launch
verification:

.. code:: python

    wp.config.verify_launch_array_access = True

When enabled, Warp checks each ``warp.array`` argument against the launch device
before the pointer is passed to the kernel.  For CPU arrays passed to CUDA
kernels, pinned CPU arrays are accepted on CUDA devices with unified virtual
addressing, and ordinary CPU arrays require
``is_cpu_memory_access_from_gpu_supported``.  For CUDA arrays, this check uses
the allocation type where Warp can determine it: default CUDA allocations use
CUDA peer-access state, while memory pool allocations use memory-pool access
state.  If the launch device cannot access the array allocation, or if Warp
cannot verify a cross-device ``warp.array`` allocation, Warp raises a
``RuntimeError`` identifying the offending argument.  This is useful when
debugging mixed-device launches on systems that do not support direct CPU/GPU
memory access or on multi-GPU systems where peer and memory-pool access are
configured separately.

Arrays backed by custom or externally wrapped allocators are a limitation of this
diagnostic.  Warp does not know the allocation kind for those arrays, so
cross-device launches fail closed when ``verify_launch_array_access`` is enabled
unless a future allocator protocol exposes enough allocation metadata to select
the correct access predicate.

Directly passing an object that exposes ``__array_interface__`` or
``__cuda_array_interface__`` is different from passing a ``warp.array``.  Those
protocols let Warp construct the kernel argument at launch time, but they do not
give Warp enough allocation information to perform the same allocation-aware
accessibility check.  In this phase, ``verify_launch_array_access`` does not
fully verify those directly passed interface objects.  Advanced users who know
such an allocation is valid are responsible for ensuring that the launch device
can legally access the pointer.

.. code:: python

    with wp.ScopedDevice("cuda:0"):
        wp.config.verify_launch_array_access = True
        wp.launch(kernel, dim=a.size, inputs=[a])

``verify_launch_array_access`` is a diagnostic option.  It adds launch overhead and should
usually be left disabled in performance-sensitive code.

Unlike ``wp.config.verify_cuda``, ``verify_launch_array_access`` can be used during CUDA
graph capture because the checks run before each launch is recorded.  For
cross-GPU graph capture, enable peer access or memory-pool access with Warp APIs
before capture begins so verification can use the recorded access state during
capture.


Atomic Operations
-----------------

Direct loads and stores do not imply atomic safety.  Code that uses atomics
between CPU and GPU memory should also check
``device.is_cpu_gpu_atomic_supported``:

.. code:: python

    device = wp.get_device("cuda:0")

    if not device.is_cpu_gpu_atomic_supported:
        raise RuntimeError("This algorithm requires CPU/GPU atomic support")

This property is independent from CPU access to GPU-resident managed memory.
For example, a system may support native CPU/GPU atomics for CPU memory while
still requiring explicit copies before CPU code can read or write default GPU
allocations.


Practical Guidance
------------------

Use the same-device pattern unless you need zero-copy CPU/GPU sharing.  When you
do need zero-copy sharing, query the specific direction your algorithm requires:

- GPU kernel reads or writes ordinary CPU arrays: check
  ``device.is_cpu_memory_access_from_gpu_supported``.
- GPU kernel reads or writes pinned CPU arrays: use ``pinned=True`` and check
  ``device.is_uva``.
- CPU code reads or writes default GPU arrays: copy the data to ``"cpu"`` first.
- CPU code accesses GPU-resident CUDA managed memory: check
  ``device.is_gpu_memory_access_from_cpu_supported``.
- CPU and GPU both use atomics on shared memory: check
  ``device.is_cpu_gpu_atomic_supported``.
- GPU kernels use arrays from another GPU: enable peer access for default CUDA
  allocations, or memory-pool access for CUDA memory-pool allocations.
- Debugging mixed-device launch failures: temporarily set
  ``wp.config.verify_launch_array_access = True``.

Prefer capability checks over platform-name checks.  They make code portable
across discrete GPUs, HMM-enabled systems, Jetson, Grace, and future coherent
CPU/GPU platforms.
