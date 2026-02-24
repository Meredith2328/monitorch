import inspect
from pathlib import Path

import torch
import torch.nn.functional as F

import monitorch


def affine_block(x: torch.Tensor, w: torch.Tensor, *, scale: float, bias: torch.Tensor) -> torch.Tensor:
    mixed = x @ w
    activated = torch.tanh(mixed * scale)
    shifted = activated + bias
    return shifted


def sequence_block(x: torch.Tensor) -> torch.Tensor:
    s1 = x + 0.2
    s2 = s1 * s1
    s3 = F.gelu(s2)
    return s3


def fused_block(x: torch.Tensor, w: torch.Tensor, scale: float, bias: torch.Tensor) -> torch.Tensor:
    proj = x @ w
    normalized = F.layer_norm(proj, proj.shape[-1:])
    gated = torch.sigmoid(normalized) * normalized
    return gated * scale + bias


def _line_range(fn) -> tuple[int, int]:
    lines, start = inspect.getsourcelines(fn)
    return start, start + len(lines) - 1


def demo_lines_flush() -> None:
    torch.manual_seed(7)
    this_file = Path(__file__).resolve()
    start, end = _line_range(affine_block)

    monitorch.reset()
    monitorch.lines(start, end, file=this_file)
    monitorch.on(show=False, save=True, save_dir="monitorch_outputs", max_nodes=10, max_matrix=6)

    x = torch.randn(8, 8)
    w = torch.randn(8, 8)
    bias = torch.randn(8, 8)

    y1 = affine_block(x, w, scale=0.35, bias=bias)
    mid = monitorch.flush(filename="combo_lines_flush_mid.png", clear=True)
    y2 = affine_block(y1, w.t(), scale=0.2, bias=bias * 0.1)
    final = monitorch.off(filename="combo_lines_flush_final.png")

    print("demo_lines_flush -> y2:", tuple(y2.shape))
    if mid and mid.saved_path:
        print("Saved mid:", mid.saved_path)
    if final and final.saved_path:
        print("Saved final:", final.saved_path)
    monitorch.lines()


def demo_watch() -> None:
    torch.manual_seed(11)
    monitorch.reset()

    with monitorch.watch(show=False, save=True, save_dir="monitorch_outputs", max_nodes=10, max_matrix=6):
        x = torch.randn(8, 8)
        h = sequence_block(x)
        mask = (torch.rand_like(h) > 0.25).float()
        kept = h * mask / 0.75
        out = kept.mean(dim=1)
        result = monitorch.flush(filename="combo_watch_manual.png")
        print("demo_watch -> out:", tuple(out.shape))
        if result and result.saved_path:
            print("Saved watch flush:", result.saved_path)


def demo_args_run() -> None:
    torch.manual_seed(23)
    monitorch.reset()
    monitorch.args(scale=0.7, bias=torch.zeros(10, 10))
    monitorch.on(show=False, save=True, save_dir="monitorch_outputs", max_nodes=12, max_matrix=6)

    x = torch.randn(10, 10)
    w = torch.randn(10, 10)
    value = monitorch.run(fused_block, x=x, w=w)
    result = monitorch.off(filename="combo_args_run.png")

    print("demo_args_run -> mean:", float(value.mean()))
    if result and result.saved_path:
        print("Saved args+run:", result.saved_path)


def main() -> None:
    demo_lines_flush()
    demo_watch()
    demo_args_run()


if __name__ == "__main__":
    main()
