import torch

import monitorch


def demo_chain() -> None:
    monitorch.on(show=False, save=True, save_dir="monitorch_outputs")
    x = torch.randn(8, 8)
    w = torch.randn(8, 8)
    y = x @ w
    z = torch.relu(y)
    out = z.mean(dim=1)
    print("Output shape:", tuple(out.shape))
    result = monitorch.off(filename="basic_chain.png")
    if result and result.saved_path:
        print("Saved:", result.saved_path)


def block(x, scale: float, bias):
    y = x * scale
    return y + bias


def demo_infer() -> None:
    monitorch.args(scale=0.3, bias=torch.ones(8, 8))
    monitorch.on(show=False, save=True, save_dir="monitorch_outputs")
    value = monitorch.run(block, x=torch.randn(8, 8))
    print("Result mean:", float(value.mean()))
    result = monitorch.off(filename="inferred_args.png")
    if result and result.saved_path:
        print("Saved:", result.saved_path)


if __name__ == "__main__":
    demo_chain()
    demo_infer()
