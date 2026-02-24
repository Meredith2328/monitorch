import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import monitorch


class TinyClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.fc1(x)
        a1 = F.gelu(z1)
        z2 = self.fc2(a1)
        a2 = F.relu(z2)
        logits = self.fc3(a2)
        return logits


def forward_probe(model: TinyClassifier, x: torch.Tensor) -> torch.Tensor:
    z1 = model.fc1(x)
    a1 = torch.relu(z1)
    z2 = model.fc2(a1)
    a2 = torch.tanh(z2)
    logits = model.fc3(a2)
    probs = torch.softmax(logits, dim=-1)
    return probs


def train_step(
    model: TinyClassifier,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)

    ce_loss = F.cross_entropy(logits, target)
    onehot = F.one_hot(target, num_classes=logits.shape[-1]).float()
    calib = (probs - onehot).pow(2).mean()
    loss = ce_loss + 0.2 * calib

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = probs.argmax(dim=-1)
        acc = (pred == target).float().mean()
        fc1_norm = model.fc1.weight.norm()

    return {
        "loss": float(loss.detach()),
        "acc": float(acc),
        "fc1_norm": float(fc1_norm),
    }


def _line_range(fn) -> tuple[int, int]:
    lines, start = inspect.getsourcelines(fn)
    return start, start + len(lines) - 1


def demo_forward_flow() -> None:
    torch.manual_seed(101)
    this_file = Path(__file__).resolve()
    start, end = _line_range(forward_probe)

    model = TinyClassifier(in_dim=16, hidden=24, classes=4)
    x = torch.randn(20, 16)

    monitorch.reset()
    monitorch.lines(start, end, file=this_file)
    monitorch.on(show=False, save=True, save_dir="monitorch_outputs", max_nodes=24, max_matrix=10)
    probs = forward_probe(model, x)
    result = monitorch.off(filename="nn_forward_flow.png")

    print("demo_forward_flow -> probs:", tuple(probs.shape))
    if result and result.saved_path:
        print("Saved nn forward:", result.saved_path)
    monitorch.lines()


def demo_train_step_flow() -> None:
    torch.manual_seed(202)
    this_file = Path(__file__).resolve()
    start, end = _line_range(train_step)

    model = TinyClassifier(in_dim=16, hidden=24, classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(24, 16)
    target = torch.randint(0, 4, (24,))

    monitorch.reset()
    monitorch.lines(start, end, file=this_file)
    monitorch.on(show=False, save=True, save_dir="monitorch_outputs", max_nodes=28, max_matrix=10)
    metrics = train_step(model, optimizer, x, target)
    result = monitorch.off(filename="nn_train_step_flow.png")

    print("demo_train_step_flow ->", metrics)
    if result and result.saved_path:
        print("Saved nn train step:", result.saved_path)
    monitorch.lines()


def main() -> None:
    demo_forward_flow()
    demo_train_step_flow()


if __name__ == "__main__":
    main()
