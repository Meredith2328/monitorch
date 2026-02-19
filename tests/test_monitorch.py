import inspect
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

import monitorch
from monitorch.core import _line_target_dependencies
from monitorch.render import build_tensor_view


def block(x, scale: float, bias):
    y = x * scale
    return y + bias


class MonitorchSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        monitorch.reset()

    def tearDown(self) -> None:
        monitorch.reset()

    def _line_target(self) -> torch.Tensor:
        base = torch.randn(4, 4)
        scaled = base * 2
        return torch.relu(scaled)

    def test_capture_same_frame_after_on(self) -> None:
        out_dir = Path("monitorch_outputs/tests")
        monitorch.on(show=False, save=True, save_dir=out_dir)

        x = torch.randn(8, 8)
        y = x + 1
        _z = torch.relu(y)

        result = monitorch.off(filename="same_frame.png")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(result.node_count, 0)
        self.assertIsNotNone(result.saved_path)
        assert result.saved_path is not None
        self.assertTrue(Path(result.saved_path).exists())

    def test_lines_filter_can_include_or_exclude_capture(self) -> None:
        lines, start = inspect.getsourcelines(self._line_target)
        idx = next(i for i, line in enumerate(lines) if "scaled = base * 2" in line)
        scaled_line = start + idx

        monitorch.lines(scaled_line, scaled_line, file=Path(__file__))
        monitorch.on(show=False, save=False)
        _ = self._line_target()
        result = monitorch.off(save=False)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(result.node_count, 0)

        monitorch.lines(1, 1, file=Path(__file__))
        monitorch.on(show=False, save=False)
        _ = self._line_target()
        excluded = monitorch.off(save=False)
        self.assertIsNone(excluded)

    def test_run_uses_args_hints_scope_and_overrides(self) -> None:
        x = torch.ones(2, 2)
        bias = torch.full((2, 2), 3.0)

        monitorch.args(scale=0.5)
        out = monitorch.run(block, x=x)
        self.assertTrue(torch.allclose(out, x * 0.5 + bias))

        override = monitorch.run(block, x=x, scale=2.0, bias=torch.zeros_like(x))
        self.assertTrue(torch.allclose(override, x * 2.0))

    def test_high_dim_tensor_is_sampled_for_view(self) -> None:
        t = torch.randn(2, 3, 64, 64)
        view = build_tensor_view(t, max_matrix=12, max_slices=4)

        self.assertEqual(view.kind, "tensor4")
        self.assertTrue(view.sampled)
        self.assertLessEqual(view.displayed_shape[0], 4)
        self.assertLessEqual(view.displayed_shape[1], 12)
        self.assertLessEqual(view.displayed_shape[2], 12)
        self.assertGreaterEqual(len(view.slices), 1)
        self.assertTrue(all(slice_view.label for slice_view in view.slices))

    def test_vector_keeps_strict_1d_shape_in_table(self) -> None:
        vector = torch.arange(8, dtype=torch.float32)
        view = build_tensor_view(vector, max_vector=8)

        self.assertEqual(view.kind, "vector")
        self.assertEqual(view.displayed_shape, (1, 8))
        self.assertEqual(len(view.slices), 1)
        self.assertEqual(tuple(view.slices[0].data.shape), (1, 8))

    def test_dependency_parser_extracts_binary_inputs(self) -> None:
        deps = _line_target_dependencies("y = x @ w", ("x", "w", "z"))
        self.assertIn("y", deps)
        self.assertEqual(deps["y"], ("x", "w"))


if __name__ == "__main__":
    unittest.main()
