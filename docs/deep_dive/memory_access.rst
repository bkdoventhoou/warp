CPU/GPU Cross-Device Memory Access
==================================

.. currentmodule:: warp

Warp arrays are associated with an allocation device such as ``"cpu"`` or
``"cuda:0"``, and kernels run on a launch device. The portable default is to
launch kernels on the same device as their array arguments. On systems with
hardware-supported CPU/GPU memory access, some cross-device patterns can also be
valid: a GPU may be able to read or write ordinary CPU memory directly, and some
systems can let CPU code directly access GPU-resident CUDA managed memory.

This page describes how Warp exposes those hardware capabilities and how to use
them when writing mixed CPU/GPU code.


Same-Device Default
-------------------

The launch device determines where a kernel runs, and the array device describes
where the array allocation lives:

.. code:: python

    cpu_array = wp.zeros(1024, dtype=float, device="cpu")
    gpu_array = wp.zeros(1024, dtype=float, device="cuda:0")

    wp.launch(kernel, dim=cpu_array.size, inputs=[cpu_array], device="cpu")
    wp.launch(kernel, dim=gpu_array.size, inputs=[gpu_array], device="cuda:0")

The same-device pattern works on all supported systems. Passing an array from
one device to a kernel running on another device depends on the capabilities of
the device that performs the access.


Capability Properties
---------------------

Each :class:`Device` exposes three CPU/GPU memory access properties:
:attr:`Device.is_cpu_memory_access_from_gpu_supported <warp.Device.is_cpu_memory_access_from_gpu_supported>`,
:attr:`Device.is_gpu_memory_access_from_cpu_supported <warp.Device.is_gpu_memory_access_from_cpu_supported>`,
and :attr:`Device.is_cpu_gpu_atomic_supported <warp.Device.is_cpu_gpu_atomic_supported>`.
See the :class:`Device` API reference for the exact attribute definitions. This
deep dive focuses on how those capabilities affect cross-device launches,
managed memory, atomics, and diagnostics.

On CPU devices, these properties are always ``False``. On GPU devices, each
property describes a specific access path or operation; support for one does not
imply support for another. For example, a system can allow GPU access to CPU
memory without allowing CPU access to GPU-resident managed memory.

.. code:: python

    device = wp.get_device("cuda:0")

    gpu_can_access_cpu = device.is_cpu_memory_access_from_gpu_supported
    cpu_can_access_gpu_managed_memory = device.is_gpu_memory_access_from_cpu_supported
    cpu_gpu_atomics_are_supported = device.is_cpu_gpu_atomic_supported


Common Hardware Models
----------------------

The exact values are reported by the CUDA driver and may vary by platform,
driver, kernel, and GPU generation. The following table summarizes the models
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
   * - Host-page-table ATS with distinct CPU/GPU memory pools
     - Yes
     - Only when reported by the driver
     - Yes, when reported by the driver

HMM stands for Heterogeneous Memory Management; for background, see NVIDIA's
`HMM overview <https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/>`__.
ATS stands for Address Translation Services. Warp does not require users to
classify the platform manually. Query the :class:`Device` properties and branch
on the behavior your program needs.

Do not infer CPU access to GPU-resident CUDA managed memory from ATS, C2C, or a
product family name. For example, a DGX Spark-class GB10 system can report ATS
and GPU access to CPU memory while
``device.is_gpu_memory_access_from_cpu_supported`` is ``False``. Query the
property directly before CPU code reads or writes GPU-resident managed memory.


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

This can avoid explicit copies on HMM and coherent CPU/GPU systems. If the
capability is false and the kernel actually dereferences the CPU pointer, CUDA
will report a runtime error such as an illegal memory access.


Accessing GPU Data From CPU Code
--------------------------------

CPU access to GPU-resident managed memory is a separate capability:

.. code:: python

    device = wp.get_device("cuda:0")
    if device.is_gpu_memory_access_from_cpu_supported:
        ...

.. important::

   ``device.is_gpu_memory_access_from_cpu_supported`` reports a hardware
   capability for CUDA managed memory. Warp exposes the property today, but
   standard Warp CUDA arrays are not managed-memory allocations. Until Warp
   provides managed-memory allocation APIs, copy CUDA arrays to ``"cpu"`` before
   CPU code reads or writes them.

CUDA arrays created by standard Warp array constructors, such as
:func:`zeros`, :func:`empty`, and :func:`ones`, are not CUDA managed-memory
allocations. This is true whether the array comes from Warp's :ref:`mempool
allocator <mempool_allocators>` or the built-in default CUDA allocator. For
those arrays, use an explicit copy before CPU code reads or writes the data:

.. code:: python

    a = wp.zeros(1024, dtype=float, device=device)
    a_cpu = a.to("cpu")
    wp.launch(cpu_kernel, dim=a_cpu.size, inputs=[a_cpu], device="cpu")

Do not assume that GPU access to CPU memory implies CPU access to GPU-resident
memory. Some systems support the former but not the latter.


Using ``Device.can_access()``
------------------------------

The method :meth:`Device.can_access` answers whether code running on one device
can access standard Warp allocations associated with another device:

.. code:: python

    launch_device = wp.get_device("cuda:0")
    array_device = wp.get_device("cpu")

    if launch_device.can_access(array_device):
        ...

For GPU kernels accessing CPU arrays, this method uses
``is_cpu_memory_access_from_gpu_supported`` because standard Warp CPU arrays use
ordinary CPU memory. For CPU code accessing CUDA arrays, it returns ``False`` for
Warp CUDA arrays because the built-in CUDA allocators do not create CUDA
managed-memory allocations. For GPU/GPU pairs, it reflects CUDA peer access state
for default CUDA allocations. See :ref:`mempool_access` for the distinction
between peer access for default CUDA allocations and memory-pool access for
mempool allocations.

``Device.can_access()`` is a conservative device-level query, not a guarantee
that every possible allocation for the other device is accessible. It does not
inspect a specific array allocation, so it does not report pinned CPU arrays
separately from ordinary CPU arrays, and it does not use
``is_gpu_memory_access_from_cpu_supported`` to accept standard Warp CUDA arrays
as CPU-accessible. Launch verification, described below, uses an internal
array-aware check for Warp array arguments. When a cross-device Warp array uses
an allocation whose accessibility Warp cannot verify, launch verification fails
closed instead of assuming the pointer is safe to use.


.. _launch_verification:

Launch Verification
-------------------

By default, Warp passes array pointers through to :func:`launch` after type,
dtype, and dimension validation. This keeps the launch path lightweight and
allows hardware-supported mixed CPU/GPU launches to work.

If you want a clear Python error before the kernel runs, set
:attr:`warp.config.launch_verification_mode`:

.. code:: python

    wp.config.launch_verification_mode = wp.config.LaunchVerificationMode.CHECKED

``LaunchVerificationMode.RELAXED`` is the default and performs no pre-launch
array access checks. Warp passes pointers through after type, dtype, and
dimension validation.

``LaunchVerificationMode.STRICT`` restores Warp's original same-device rule and
rejects every cross-device Warp array argument before launch.

``LaunchVerificationMode.CHECKED`` checks each cross-device Warp array argument
against the launch device. For CPU arrays passed to CUDA kernels, pinned CPU
arrays are accepted on CUDA devices with unified virtual addressing, and
ordinary CPU arrays require ``is_cpu_memory_access_from_gpu_supported``. For CUDA
arrays, default CUDA allocations use CUDA peer-access state, while memory pool
allocations use memory-pool access state.

If the launch device cannot access the array allocation, or if Warp cannot
verify a cross-device Warp array allocation, ``LaunchVerificationMode.CHECKED``
raises a ``RuntimeError`` identifying the offending argument. This is useful
when debugging mixed-device launches on systems that do not support direct
CPU/GPU memory access or on multi-GPU systems where peer and memory-pool access
are configured separately.

Arrays backed by custom or externally wrapped allocators are a limitation of this
diagnostic. Warp does not know the allocation kind for those arrays, so
cross-device launches fail closed in ``LaunchVerificationMode.CHECKED`` unless a
future allocator protocol exposes enough allocation metadata to select the
correct access predicate.

Directly passing an object that exposes ``__array_interface__`` or
``__cuda_array_interface__`` is different from passing a Warp array. Those
protocols let Warp construct the kernel argument at launch time, but they do not
give Warp enough allocation information to perform the same allocation-aware
accessibility check. In this phase, ``LaunchVerificationMode.CHECKED`` does not
fully verify directly passed objects exposing these protocols. Advanced users
who know such an allocation is valid are responsible for ensuring that the
launch device can legally access the pointer.

.. code:: python

    with wp.ScopedDevice("cuda:0"):
        wp.config.launch_verification_mode = wp.config.LaunchVerificationMode.CHECKED
        wp.launch(kernel, dim=a.size, inputs=[a])

:attr:`warp.config.launch_verification_mode` can add launch overhead in
``LaunchVerificationMode.STRICT`` and ``LaunchVerificationMode.CHECKED`` modes.
Use ``LaunchVerificationMode.RELAXED`` in performance-sensitive code that has
already validated its launch accessibility assumptions.

Unlike :attr:`warp.config.verify_cuda`,
:attr:`warp.config.launch_verification_mode` can be used during CUDA graph
capture because ``LaunchVerificationMode.CHECKED`` checks run before each launch
is recorded. For cross-GPU graph capture, enable peer access or memory-pool
access with Warp APIs before capture begins so verification can use the recorded
access state during capture. When a CUDA graph captures a launch with CPU array
arguments, replay uses the same captured CPU pointers. If the arrays remain
alive, CPU updates made between replays are visible to kernels on devices that
can access CPU memory.


Atomic Operations
-----------------

Direct loads and stores do not imply atomic safety. Code that uses atomics
between CPU and GPU memory should also check
:attr:`Device.is_cpu_gpu_atomic_supported <warp.Device.is_cpu_gpu_atomic_supported>`:

.. code:: python

    device = wp.get_device("cuda:0")

    if not device.is_cpu_gpu_atomic_supported:
        raise RuntimeError("This algorithm requires CPU/GPU atomic support")

:attr:`Device.is_cpu_gpu_atomic_supported <warp.Device.is_cpu_gpu_atomic_supported>`
answers only the atomic-operation part of a shared-memory workflow. The memory
must still be accessible from both processors, and the program must provide any
required synchronization.

For example, GPU atomics into a CPU allocation require both GPU access to CPU
memory and CPU/GPU atomic support:

.. code:: python

    device = wp.get_device("cuda:0")
    counters = wp.zeros(1, dtype=wp.int32, device="cpu")

    if (
        device.is_cpu_memory_access_from_gpu_supported
        and device.is_cpu_gpu_atomic_supported
    ):
        wp.launch(update_counters, dim=n, inputs=[counters], device=device)
        wp.synchronize_device(device)
        print(counters.numpy()[0])

The same requirements apply when CPU and GPU work overlap. If a CPU kernel and
a GPU kernel both write the same shared allocation concurrently, all conflicting
accesses must use atomic operations, and the device must report CPU/GPU atomic
support. Atomicity prevents lost updates, but it does not provide a deterministic
ordering for non-commutative operations or floating-point accumulation:

.. code:: python

    # Assume both kernels call wp.atomic_add(counters, 0, 1) once per thread.
    counters = wp.zeros(1, dtype=wp.int32, device="cpu")

    if (
        device.is_cpu_memory_access_from_gpu_supported
        and device.is_cpu_gpu_atomic_supported
    ):
        wp.launch(gpu_increment, dim=num_gpu_threads, inputs=[counters], device=device)
        wp.launch(cpu_increment, dim=num_cpu_threads, inputs=[counters], device="cpu")

        wp.synchronize_device(device)
        assert counters.numpy()[0] == num_gpu_threads + num_cpu_threads

If :attr:`Device.is_cpu_gpu_atomic_supported <warp.Device.is_cpu_gpu_atomic_supported>`
is ``False``, do not rely on concurrent CPU/GPU atomics, even on systems where
the GPU can directly load and store CPU memory.

That does not make ordinary CUDA device allocations CPU-accessible. CPU code
should still copy CUDA arrays before reading or writing them:

.. code:: python

    values = wp.zeros(1024, dtype=float, device=device)
    values_cpu = values.to("cpu")


Practical Guidance
------------------

Use the same-device pattern unless you need zero-copy CPU/GPU sharing. When you
do need zero-copy sharing, query the specific direction your algorithm requires:

- GPU kernel reads or writes ordinary CPU arrays: check
  ``device.is_cpu_memory_access_from_gpu_supported``.
- GPU kernel reads or writes pinned CPU arrays: use ``pinned=True`` and check
  ``device.is_uva``.
- CPU code reads or writes default GPU arrays: copy the data to ``"cpu"`` first.
- CPU code accesses externally provided GPU-resident CUDA managed memory: check
  ``device.is_gpu_memory_access_from_cpu_supported``.
- CPU and GPU both use atomics on shared memory: make sure the allocation is
  accessible from both processors, and check
  ``device.is_cpu_gpu_atomic_supported``.
- GPU kernels use arrays from another GPU: enable peer access for default CUDA
  allocations, or memory-pool access for CUDA memory-pool allocations.
- Debugging mixed-device launch failures: temporarily set
  :attr:`warp.config.launch_verification_mode` to
  ``wp.config.LaunchVerificationMode.CHECKED``.

Prefer capability checks over platform-name checks. They make code portable
across discrete GPUs, HMM-enabled systems, Jetson, Grace, and future coherent
CPU/GPU platforms.
