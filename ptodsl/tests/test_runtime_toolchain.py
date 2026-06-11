import unittest
import importlib.util
from pathlib import Path


_TOOLCHAIN_PATH = (
    Path(__file__).resolve().parents[1] / "ptodsl" / "_runtime" / "toolchain.py"
)
_SPEC = importlib.util.spec_from_file_location("ptodsl_runtime_toolchain", _TOOLCHAIN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOLCHAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOLCHAIN)
aicore_arch_for_kernel_kind = _TOOLCHAIN.aicore_arch_for_kernel_kind


class RuntimeToolchainTest(unittest.TestCase):
    def test_a2_a3_use_c220_aicore_arch(self):
        self.assertEqual(
            aicore_arch_for_kernel_kind("vector", "a3"), "dav-c220-vec"
        )
        self.assertEqual(
            aicore_arch_for_kernel_kind("cube", "a3"), "dav-c220-cube"
        )
        self.assertEqual(
            aicore_arch_for_kernel_kind("vector", "a2"), "dav-c220-vec"
        )

    def test_a5_uses_c310_aicore_arch(self):
        self.assertEqual(
            aicore_arch_for_kernel_kind("vector", "a5"), "dav-c310-vec"
        )
        self.assertEqual(
            aicore_arch_for_kernel_kind("cube", "a5"), "dav-c310-cube"
        )


if __name__ == "__main__":
    unittest.main()
