# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-device array access and launch verification.

These tests cover Warp's conservative memory-access capability reporting,
default launch behavior for mixed-device array arguments, and opt-in launch
verification through ``wp.config.verify_launch_array_access``. They also check
that verification uses allocation-specific CUDA access rules where possible:
ordinary CPU memory, pinned CPU memory, default CUDA allocations, CUDA memory
pool allocations, and array views backed by a parent allocation.
"""

import contextlib
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@contextlib.contextmanager
def launch_verification(enabled: bool):
    """Temporarily set launch array-access verification and restore the previous value."""

    old_value = wp.config.verify_launch_array_access
    wp.config.verify_launch_array_access = enabled
    try:
        yield
    finally:
        wp.config.verify_launch_array_access = old_value


@wp.kernel
def read_cpu_write_gpu(src: wp.array[wp.float32], dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = src[i] * 2.0


@wp.kernel
def write_output_array(dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = float(i) + 10.0


@wp.kernel
def read_gpu_write_cpu(src: wp.array[wp.float32], dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = src[i] + 3.0


def test_unified_memory_device_capabilities(test, device):
    """Memory-access capability flags are exposed as booleans on every device."""

    for attr in (
        "is_cpu_memory_access_from_gpu_supported",
        "is_gpu_memory_access_from_cpu_supported",
        "is_cpu_gpu_atomic_supported",
    ):
        test.assertIsInstance(getattr(device, attr), bool)

    if device.is_cpu:
        test.assertFalse(device.is_cpu_memory_access_from_gpu_supported)
        test.assertFalse(device.is_gpu_memory_access_from_cpu_supported)
        test.assertFalse(device.is_cpu_gpu_atomic_supported)


def test_unified_memory_launch_verification_mode_config(test, device):
    """Launch verification mode is an enum-backed public config setting."""

    test.assertEqual(int(wp.config.LaunchVerificationMode.STRICT), 0)
    test.assertEqual(int(wp.config.LaunchVerificationMode.RELAXED), 1)
    test.assertEqual(int(wp.config.LaunchVerificationMode.CHECKED), 2)
    test.assertIs(wp.config.launch_verification_mode, wp.config.LaunchVerificationMode.RELAXED)
    test.assertFalse(hasattr(wp.config, "verify_launch_array_access"))


def test_unified_memory_can_access(test, device):
    """Device.can_access() reports conservative default-allocation reachability."""

    cpu = wp.get_device("cpu")

    test.assertTrue(device.can_access(device))
    test.assertTrue(cpu.can_access(cpu))

    if device.is_cuda:
        test.assertEqual(device.can_access(cpu), device.is_cpu_memory_access_from_gpu_supported)
        test.assertFalse(cpu.can_access(device))

        for other in wp.get_cuda_devices():
            if other == device:
                test.assertTrue(device.can_access(other))
            else:
                test.assertEqual(device.can_access(other), wp.is_peer_access_enabled(other, device))


def test_unified_memory_record_cmd_skips_default_access_check(test, device):
    """Command recording should not restore the old unconditional same-device check."""

    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device="cpu")
    dst = wp.empty(4, dtype=wp.float32, device=device)

    with launch_verification(False):
        cmd = wp.launch(read_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device, record_cmd=True)

    test.assertIsInstance(cmd, wp.Launch)


def test_unified_memory_verify_rejects_gpu_reading_cpu_when_unsupported(test, device):
    """Opt-in launch verification catches unsupported GPU access to CPU memory."""

    if device.is_cpu_memory_access_from_gpu_supported:
        test.skipTest(f"{device} can access CPU memory")

    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device="cpu")
    dst = wp.empty(4, dtype=wp.float32, device=device)

    with launch_verification(True):
        with test.assertRaisesRegex(RuntimeError, "array allocation is not accessible or cannot be verified"):
            wp.launch(read_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device, record_cmd=True)


def test_unified_memory_verify_rejects_cpu_reading_gpu_when_unsupported(test, device):
    """Warp default CUDA allocations are not treated as CPU-accessible managed memory."""

    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device=device)
    dst = wp.empty(4, dtype=wp.float32, device="cpu")

    with launch_verification(True):
        with test.assertRaisesRegex(RuntimeError, "array allocation is not accessible or cannot be verified"):
            wp.launch(read_gpu_write_cpu, dim=src.size, inputs=[src], outputs=[dst], device="cpu", record_cmd=True)


def test_unified_memory_cpu_launch_always_rejects_gpu_array(test, device):
    """CPU kernels must never accept CUDA-backed arrays."""

    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device=device)
    dst = wp.empty(4, dtype=wp.float32, device="cpu")

    with launch_verification(False):
        with test.assertRaisesRegex(RuntimeError, "array allocation is not accessible or cannot be verified"):
            wp.launch(read_gpu_write_cpu, dim=src.size, inputs=[src], outputs=[dst], device="cpu", record_cmd=True)


def test_unified_memory_cuda_launch_reads_cpu_array_when_supported(test, device):
    """On coherent systems, GPU kernels can read ordinary CPU arrays directly."""

    if not device.is_cpu_memory_access_from_gpu_supported:
        test.skipTest(f"{device} cannot access CPU memory")

    src_np = np.arange(8, dtype=np.float32)
    src = wp.array(src_np, dtype=wp.float32, device="cpu")
    dst = wp.empty(src.size, dtype=wp.float32, device=device)

    with launch_verification(True):
        wp.launch(read_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device)

    np.testing.assert_allclose(dst.numpy(), src_np * 2.0)


def test_unified_memory_cuda_launch_writes_cpu_array_when_supported(test, device):
    """On coherent systems, GPU kernels can write ordinary CPU arrays directly."""

    if not device.is_cpu_memory_access_from_gpu_supported:
        test.skipTest(f"{device} cannot access CPU memory")

    dst = wp.empty(8, dtype=wp.float32, device="cpu")

    with launch_verification(True):
        wp.launch(write_output_array, dim=dst.size, outputs=[dst], device=device)

    # dst is CPU memory written by the GPU; CPU-backed .numpy() does not synchronize the launch.
    wp.synchronize_device(device)
    np.testing.assert_allclose(dst.numpy(), np.arange(8, dtype=np.float32) + 10.0)


def test_unified_memory_cuda_launch_reads_pinned_cpu_array_when_uva_supported(test, device):
    """Pinned CPU arrays are GPU-accessible on CUDA devices with unified virtual addressing."""

    if not device.is_uva:
        test.skipTest(f"{device} does not support unified virtual addressing")

    src_np = np.arange(8, dtype=np.float32)
    src = wp.array(src_np, dtype=wp.float32, device="cpu", pinned=True)
    dst = wp.empty(src.size, dtype=wp.float32, device=device)

    test.assertTrue(src.pinned)

    with launch_verification(True):
        wp.launch(read_cpu_write_gpu, dim=src.size, inputs=[src], outputs=[dst], device=device)

    np.testing.assert_allclose(dst.numpy(), src_np * 2.0)


def test_unified_memory_cuda_launch_writes_pinned_cpu_array_when_uva_supported(test, device):
    """Pinned CPU output arrays are valid GPU launch targets on UVA CUDA devices."""

    if not device.is_uva:
        test.skipTest(f"{device} does not support unified virtual addressing")

    dst = wp.empty(8, dtype=wp.float32, device="cpu", pinned=True)

    test.assertTrue(dst.pinned)

    with launch_verification(True):
        wp.launch(write_output_array, dim=dst.size, outputs=[dst], device=device)

    # dst is CPU memory written by the GPU; CPU-backed .numpy() does not synchronize the launch.
    wp.synchronize_device(device)
    np.testing.assert_allclose(dst.numpy(), np.arange(8, dtype=np.float32) + 10.0)


def test_unified_memory_array_view_allocator_lookup_uses_parent_array(test, device):
    """Array views must use the base allocation when launch verification checks access."""

    src = wp.array(np.arange(8, dtype=np.float32), dtype=wp.float32, device=device)
    src_slice = src[1:]

    test.assertIs(wp._src.context._get_array_allocator(src_slice), src._allocator)


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestUnifiedMemory(unittest.TestCase):
    @unittest.skipUnless(get_cuda_device_pair_with_peer_access_support(), "Requires devices with peer access support")
    def test_unified_memory_verify_uses_peer_access_for_default_cuda_allocations(self):
        """Default CUDA allocations use peer-access state for cross-GPU verification."""

        target_device, peer_device = get_cuda_device_pair_with_peer_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_mempool_access_enabled(target_device, peer_device, False)
            wp.set_peer_access_enabled(target_device, peer_device, True)

            with wp.ScopedMempool(target_device, False), wp.ScopedMempool(peer_device, False):
                src = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=target_device)
                dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src._allocator).__name__, "CudaDefaultAllocator")

            wp.load_module(device=peer_device)
            wp.synchronize_device(target_device)
            with launch_verification(True):
                wp.launch(read_cpu_write_gpu, dim=n, inputs=[src], outputs=[dst], device=peer_device)

            np.testing.assert_allclose(dst.numpy(), np.arange(n, dtype=np.float32) * 2.0)
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)

    @unittest.skipUnless(get_cuda_device_pair_with_peer_access_support(), "Requires devices with peer access support")
    def test_unified_memory_verify_uses_parent_allocator_for_default_cuda_slices(self):
        """Slices of default CUDA allocations should follow the base array's allocator."""

        target_device, peer_device = get_cuda_device_pair_with_peer_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_mempool_access_enabled(target_device, peer_device, False)
            wp.set_peer_access_enabled(target_device, peer_device, True)

            with wp.ScopedMempool(target_device, False), wp.ScopedMempool(peer_device, False):
                src_base = wp.array(np.arange(n + 1, dtype=np.float32), dtype=wp.float32, device=target_device)
                src = src_base[1:]
                dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src_base._allocator).__name__, "CudaDefaultAllocator")
            self.assertIs(src._ref, src_base)

            wp.load_module(device=peer_device)
            wp.synchronize_device(target_device)
            with launch_verification(True):
                wp.launch(read_cpu_write_gpu, dim=n, inputs=[src], outputs=[dst], device=peer_device)

            np.testing.assert_allclose(dst.numpy(), np.arange(1, n + 1, dtype=np.float32) * 2.0)
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)

    @unittest.skipUnless(
        get_cuda_device_pair_with_mempool_access_support(), "Requires devices with mempool access support"
    )
    def test_unified_memory_verify_uses_mempool_access_for_cuda_mempool_allocations(self):
        """CUDA mempool allocations use mempool-access state for cross-GPU verification."""

        target_device, peer_device = get_cuda_device_pair_with_mempool_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_peer_access_enabled(target_device, peer_device, False)
            wp.set_mempool_access_enabled(target_device, peer_device, True)

            with wp.ScopedMempool(target_device, True):
                src = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=target_device)
            dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src._allocator).__name__, "CudaMempoolAllocator")

            wp.load_module(device=peer_device)
            wp.synchronize_device(target_device)
            with launch_verification(True):
                wp.launch(read_cpu_write_gpu, dim=n, inputs=[src], outputs=[dst], device=peer_device)

            np.testing.assert_allclose(dst.numpy(), np.arange(n, dtype=np.float32) * 2.0)
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)

    @unittest.skipUnless(
        get_cuda_device_pair_with_mempool_access_support(), "Requires devices with mempool access support"
    )
    def test_unified_memory_verify_uses_parent_allocator_for_cuda_mempool_slices(self):
        """Slices of CUDA mempool allocations should follow the base array's allocator."""

        target_device, peer_device = get_cuda_device_pair_with_mempool_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_peer_access_enabled(target_device, peer_device, False)
            wp.set_mempool_access_enabled(target_device, peer_device, True)

            with wp.ScopedMempool(target_device, True):
                src_base = wp.array(np.arange(n + 1, dtype=np.float32), dtype=wp.float32, device=target_device)
                src = src_base[1:]
            dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src_base._allocator).__name__, "CudaMempoolAllocator")
            self.assertIs(src._ref, src_base)

            wp.load_module(device=peer_device)
            wp.synchronize_device(target_device)
            with launch_verification(True):
                wp.launch(read_cpu_write_gpu, dim=n, inputs=[src], outputs=[dst], device=peer_device)

            np.testing.assert_allclose(dst.numpy(), np.arange(1, n + 1, dtype=np.float32) * 2.0)
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)

    @unittest.skipUnless(
        get_cuda_device_pair_with_mempool_access_support(), "Requires devices with mempool access support"
    )
    def test_unified_memory_verify_rejects_mempool_allocation_without_mempool_access(self):
        """Peer access alone should not validate cross-GPU CUDA mempool allocations."""

        target_device, peer_device = get_cuda_device_pair_with_mempool_access_support()
        n = 8

        peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
        mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
        try:
            wp.set_peer_access_enabled(target_device, peer_device, True)
            wp.set_mempool_access_enabled(target_device, peer_device, False)

            with wp.ScopedMempool(target_device, True):
                src = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=target_device)
            dst = wp.empty(n, dtype=wp.float32, device=peer_device)

            self.assertEqual(type(src._allocator).__name__, "CudaMempoolAllocator")

            with launch_verification(True):
                with self.assertRaisesRegex(RuntimeError, "array allocation is not accessible or cannot be verified"):
                    wp.launch(
                        read_cpu_write_gpu, dim=n, inputs=[src], outputs=[dst], device=peer_device, record_cmd=True
                    )
        finally:
            wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)
            wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)


add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_device_capabilities",
    test_unified_memory_device_capabilities,
    devices=devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_launch_verification_mode_config",
    test_unified_memory_launch_verification_mode_config,
    devices=[wp.get_device("cpu")],
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_can_access",
    test_unified_memory_can_access,
    devices=devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_record_cmd_skips_default_access_check",
    test_unified_memory_record_cmd_skips_default_access_check,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_verify_rejects_gpu_reading_cpu_when_unsupported",
    test_unified_memory_verify_rejects_gpu_reading_cpu_when_unsupported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_verify_rejects_cpu_reading_gpu_when_unsupported",
    test_unified_memory_verify_rejects_cpu_reading_gpu_when_unsupported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_cpu_launch_always_rejects_gpu_array",
    test_unified_memory_cpu_launch_always_rejects_gpu_array,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_cuda_launch_reads_cpu_array_when_supported",
    test_unified_memory_cuda_launch_reads_cpu_array_when_supported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_cuda_launch_writes_cpu_array_when_supported",
    test_unified_memory_cuda_launch_writes_cpu_array_when_supported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_cuda_launch_reads_pinned_cpu_array_when_uva_supported",
    test_unified_memory_cuda_launch_reads_pinned_cpu_array_when_uva_supported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_cuda_launch_writes_pinned_cpu_array_when_uva_supported",
    test_unified_memory_cuda_launch_writes_pinned_cpu_array_when_uva_supported,
    devices=cuda_devices,
)
add_function_test(
    TestUnifiedMemory,
    "test_unified_memory_array_view_allocator_lookup_uses_parent_array",
    test_unified_memory_array_view_allocator_lookup_uses_parent_array,
    devices=devices,
)
if __name__ == "__main__":
    unittest.main(verbosity=2)
