from pathlib import Path

import torch

import monitorch


def train_step(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    y = x @ w
    y = torch.tanh(y)
    return y[:, :8]


def main() -> None:
    this_file = Path(__file__).resolve()
    start = train_step.__code__.co_firstlineno
    end = start + 8

    monitorch.lines(start, end, file=this_file)
    monitorch.on(show=False, save=True, save_dir="monitorch_outputs")

    x = torch.randn(16, 16)
    w = torch.randn(16, 16)
    out = train_step(x, w)
    print("Output shape:", tuple(out.shape))

    result = monitorch.off(filename="vscode_flow.png")
    if result and result.saved_path:
        print("Saved:", result.saved_path)


if __name__ == "__main__":
    main()
