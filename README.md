# monitorch

`monitorch` is a lightweight PyTorch debugging helper.
It captures Tensor changes while your code runs and renders them as one flow chart with arrows.

## Highlights

- Graph view: Tensor updates are rendered as table nodes, and arrows connect real data dependencies (for example, `x,w -> y`).
- Argument inference: `monitorch.run()` can infer missing args from `overrides -> monitorch.args() -> local/global scope -> defaults -> heuristics`.
- Shape-faithful tables: `(n,)` is rendered as `1 x n`; 3D/4D tensors are rendered as indexed 2D slice tables.
- Downsampling: large tensors are sampled for readability (`max_vector`, `max_matrix`, `max_slices`).
- Lightweight workflow: `on()/off()` can be dropped into notebook or script code without changing your model logic.

## Install

```bash
pip install -e .
```

## Quick Start

### Notebook-style

```python
import torch
import monitorch

monitorch.on(show=True, save=False)

x = torch.randn(8, 8)
w = torch.randn(8, 8)
y = x @ w
z = torch.relu(y)

monitorch.off()
```

### VS Code / script-style

```python
import torch
import monitorch

monitorch.lines(10, 40)  # only monitor these lines in this file
monitorch.on(show=False, save=True, save_dir="monitorch_outputs")

a = torch.randn(16, 16)
b = a.softmax(dim=-1)
c = b[:, :8]

monitorch.off(filename="block.png")
```

### Inferred arguments + explicit overrides

```python
import torch
import monitorch

def block(x, scale: float, bias):
    y = x * scale
    return y + bias

monitorch.args(scale=0.5, bias=torch.ones(8, 8))
monitorch.on(show=False, save=True)
out = monitorch.run(block, x=torch.randn(8, 8))
monitorch.off(filename="inferred.png")
```

## API

- `monitorch.on(...)`: enable tracing.
- Useful `on()` knobs: `max_vector`, `max_matrix`, `max_slices`.
- `monitorch.off(...)`: disable tracing and optionally render.
- `monitorch.flush(...)`: render current captured nodes without disabling.
- `monitorch.lines(start, end, file=...)`: constrain monitoring by line range.
- `monitorch.args(**kwargs)`: provide argument hints for `run()`.
- `monitorch.run(fn, **overrides)`: call function with inferred arguments.
- `monitorch.watch(...)`: context manager around `on/off`.
- `monitorch.reset()`: clear internal runtime state.

## Current limits

- Granularity is Python line-level Tensor delta tracking, not ATen op-level tracing.
- Default history size is `12` nodes (`on(max_nodes=...)` to change).
