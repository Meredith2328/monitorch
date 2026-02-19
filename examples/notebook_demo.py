import torch

import monitorch


def main() -> None:
    monitorch.on(show=True, save=False, max_nodes=10)

    x = torch.randn(8, 8)
    w = torch.randn(8, 8)
    y = x @ w
    z = torch.relu(y)
    out = z.mean(dim=1)

    print("Output shape:", tuple(out.shape))
    monitorch.off()


if __name__ == "__main__":
    main()
