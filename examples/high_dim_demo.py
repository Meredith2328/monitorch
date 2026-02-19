import torch

import monitorch


def main() -> None:
    monitorch.on(show=False, save=True, save_dir="monitorch_outputs", max_matrix=6, max_slices=6)

    t3 = torch.randn(4, 8, 8)
    t3_out = torch.relu(t3)

    t4 = torch.randn(2, 3, 8, 8)
    t4_out = t4 * 0.5 + t3_out[:2].unsqueeze(1)

    print("t3_out:", tuple(t3_out.shape), "t4_out:", tuple(t4_out.shape))
    result = monitorch.off(filename="high_dim_demo.png")
    if result and result.saved_path:
        print("Saved:", result.saved_path)


if __name__ == "__main__":
    main()
