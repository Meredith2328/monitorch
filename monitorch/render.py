from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class TensorSlice:
    label: str
    data: np.ndarray  # 2D matrix


@dataclass
class TensorView:
    kind: str
    slices: Tuple[TensorSlice, ...]
    original_shape: Tuple[int, ...]
    displayed_shape: Tuple[int, ...]
    dtype: str
    device: str
    sampled: bool
    stats: Dict[str, float]


@dataclass
class RenderNode:
    name: str
    filename: str
    lineno: int
    code: str
    view: TensorView
    dependencies: Tuple[str, ...] = ()


@dataclass
class RenderOutput:
    saved_path: Optional[str]
    displayed: bool
    node_count: int


def build_tensor_view(
    tensor: torch.Tensor,
    *,
    max_vector: int = 32,
    max_matrix: int = 8,
    max_slices: int = 6,
) -> TensorView:
    with torch.no_grad():
        detached = tensor.detach()
        original_shape = tuple(detached.shape)
        dtype = str(detached.dtype).replace("torch.", "")
        device = str(detached.device)
        sampled = False

        # Fallback for >4D: keep center slices until 4D.
        while detached.dim() > 4:
            middle = max(detached.shape[0] // 2, 0)
            detached = detached.select(0, middle)
            sampled = True

        slices: List[TensorSlice] = []
        displayed_shape: Tuple[int, ...]

        if detached.dim() == 0:
            scalar = detached.to("cpu").to(torch.float32).reshape(1, 1).numpy()
            slices.append(TensorSlice(label="", data=scalar))
            kind = "scalar"
            displayed_shape = (1, 1)
        elif detached.dim() == 1:
            indices = _sample_indices(detached.shape[0], max_vector, detached.device)
            sampled = sampled or detached.shape[0] > max_vector
            vector = detached.index_select(0, indices).to("cpu").to(torch.float32).reshape(1, -1).numpy()
            slices.append(TensorSlice(label="", data=vector))
            kind = "vector"
            displayed_shape = tuple(vector.shape)
        elif detached.dim() == 2:
            matrix, matrix_sampled = _sample_matrix(detached, max_matrix)
            sampled = sampled or matrix_sampled
            slices.append(TensorSlice(label="", data=matrix))
            kind = "matrix"
            displayed_shape = tuple(matrix.shape)
        elif detached.dim() == 3:
            indices = _sample_indices(detached.shape[0], max_slices, detached.device)
            sampled = sampled or detached.shape[0] > max_slices
            per_slice_limit = min(max_matrix, 4) if int(indices.numel()) > 4 else max_matrix
            for index in indices.tolist():
                matrix, matrix_sampled = _sample_matrix(detached[index], per_slice_limit)
                sampled = sampled or matrix_sampled
                slices.append(TensorSlice(label=f"[{index}]", data=matrix))
            kind = "tensor3"
            displayed_shape = _stacked_shape(slices)
        else:
            dim0, dim1 = int(detached.shape[0]), int(detached.shape[1])
            total = dim0 * dim1
            linear_indices = _sample_indices(total, max_slices, detached.device)
            sampled = sampled or total > max_slices
            per_slice_limit = min(max_matrix, 4) if int(linear_indices.numel()) > 4 else max_matrix
            for linear in linear_indices.tolist():
                i = int(linear // dim1)
                j = int(linear % dim1)
                matrix, matrix_sampled = _sample_matrix(detached[i, j], per_slice_limit)
                sampled = sampled or matrix_sampled
                slices.append(TensorSlice(label=f"[{i},{j}]", data=matrix))
            kind = "tensor4"
            displayed_shape = _stacked_shape(slices)

        stats = _stats_from_slices(slices)
        return TensorView(
            kind=kind,
            slices=tuple(slices),
            original_shape=original_shape,
            displayed_shape=displayed_shape,
            dtype=dtype,
            device=device,
            sampled=sampled,
            stats=stats,
        )


def render_chain(
    nodes: Sequence[RenderNode],
    *,
    show: bool,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> RenderOutput:
    if not nodes:
        return RenderOutput(saved_path=None, displayed=False, node_count=0)

    edges = _build_edges(nodes)
    positions, box_width, box_height, layout_cols, layout_rows = _layout_nodes(len(nodes), edges)
    figure_width, figure_height = _figure_size(
        nodes,
        layout_cols,
        layout_rows,
        has_edges=bool(edges),
    )
    figure = plt.figure(figsize=(figure_width, figure_height))
    edge_axis = figure.add_axes([0.0, 0.0, 1.0, 1.0], zorder=6, frameon=False)
    edge_axis.patch.set_alpha(0.0)
    edge_axis.set_axis_off()

    for index, node in enumerate(nodes):
        center_x, center_y = positions[index]
        left = center_x - box_width / 2.0
        bottom = center_y - box_height / 2.0
        node_axis = figure.add_axes([left, bottom, box_width, box_height], zorder=2)
        _draw_node(node_axis, node)

    _draw_edges(edge_axis, edges, positions, box_width)

    if title:
        figure.suptitle(title, fontsize=14, y=0.985, color="#0F172A")

    output_path: Optional[str] = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        output_path = str(save_path)

    if show:
        plt.show()
    plt.close(figure)
    return RenderOutput(saved_path=output_path, displayed=show, node_count=len(nodes))


def _build_edges(nodes: Sequence[RenderNode]) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    last_by_name: Dict[str, int] = {}

    for index, node in enumerate(nodes):
        parents: List[int] = []
        for dep in node.dependencies:
            parent_index = last_by_name.get(dep)
            if parent_index is not None and parent_index != index:
                parents.append(parent_index)

        if not parents:
            parent_index = last_by_name.get(node.name)
            if parent_index is not None and parent_index != index:
                parents.append(parent_index)

        for parent in _unique_ints(parents):
            edges.append((parent, index))

        last_by_name[node.name] = index

    return edges


def _layout_nodes(
    node_count: int,
    edges: Sequence[Tuple[int, int]],
) -> Tuple[Dict[int, Tuple[float, float]], float, float, int, int]:
    parents_by_node: Dict[int, List[int]] = defaultdict(list)
    for src, dst in edges:
        parents_by_node[dst].append(src)

    levels = [0 for _ in range(node_count)]
    for index in range(node_count):
        parents = parents_by_node.get(index, [])
        if parents:
            levels[index] = max(levels[parent] + 1 for parent in parents)

    level_groups: Dict[int, List[int]] = defaultdict(list)
    for index, level in enumerate(levels):
        level_groups[level].append(index)

    max_level = max(levels) if levels else 0
    max_rows = max(len(nodes_at_level) for nodes_at_level in level_groups.values()) if level_groups else 1

    left = 0.04
    right = 0.96
    bottom = 0.07
    top = 0.88
    # No dependency levels: use centered grid to avoid left-clustered layout.
    if max_level == 0:
        if node_count <= 1:
            grid_cols = 1
        elif node_count <= 4:
            grid_cols = 2
        else:
            grid_cols = 3
        grid_rows = int(np.ceil(node_count / grid_cols))

        gap_x = 0.06
        gap_y = 0.08
        usable_w = right - left
        usable_h = top - bottom

        box_width = min(0.36, (usable_w - gap_x * (grid_cols - 1)) / max(1, grid_cols))
        box_width = max(0.24, box_width)
        box_height = min(0.40, (usable_h - gap_y * (grid_rows - 1)) / max(1, grid_rows))
        box_height = max(0.28, box_height)

        total_w = grid_cols * box_width + gap_x * (grid_cols - 1)
        start_x = 0.5 - total_w / 2.0 + box_width / 2.0
        x_positions = np.array([start_x + i * (box_width + gap_x) for i in range(grid_cols)], dtype=float)

        total_h = grid_rows * box_height + gap_y * (grid_rows - 1)
        start_y = 0.5 + total_h / 2.0 - box_height / 2.0
        y_positions = np.array([start_y - i * (box_height + gap_y) for i in range(grid_rows)], dtype=float)

        positions: Dict[int, Tuple[float, float]] = {}
        for index in range(node_count):
            row = index // grid_cols
            col = index % grid_cols
            positions[index] = (float(x_positions[col]), float(y_positions[row]))

        return positions, box_width, box_height, grid_cols, grid_rows

    num_levels = max_level + 1
    if num_levels <= 2:
        gap_x = 0.12
        width_cap = 0.34
    elif num_levels == 3:
        gap_x = 0.09
        width_cap = 0.27
    elif num_levels <= 5:
        gap_x = 0.06
        width_cap = 0.21
    else:
        gap_x = 0.04
        width_cap = 0.16

    usable_w = right - left
    available_w = usable_w - gap_x * max(0, num_levels - 1)
    if available_w <= 0:
        gap_x = 0.01
        available_w = usable_w - gap_x * max(0, num_levels - 1)
    max_box_width = available_w / max(1, num_levels)
    box_width = min(width_cap, max_box_width)
    box_width = min(max_box_width, max(0.07, box_width))

    if max_rows <= 1:
        box_height = min(0.40, (top - bottom) * 0.70)
    else:
        box_height = min(0.32, (top - bottom) / (max_rows + 0.35))

    if num_levels == 1:
        x_positions = np.array([(left + right) / 2.0], dtype=float)
    else:
        total_w = num_levels * box_width + (num_levels - 1) * gap_x
        start = 0.5 - total_w / 2.0 + box_width / 2.0
        x_positions = np.array([start + i * (box_width + gap_x) for i in range(num_levels)], dtype=float)
    positions = {}

    for level in range(num_levels):
        members = level_groups.get(level, [])
        if not members:
            continue
        if len(members) == 1:
            y_coords = np.array([(top + bottom) / 2.0], dtype=float)
        else:
            y_coords = np.linspace(
                top - box_height / 2.0,
                bottom + box_height / 2.0,
                num=len(members),
            )
        for node_index, y_pos in zip(members, y_coords):
            positions[node_index] = (float(x_positions[level]), float(y_pos))

    return positions, box_width, box_height, num_levels, max_rows


def _figure_size(
    nodes: Sequence[RenderNode],
    layout_cols: int,
    layout_rows: int,
    *,
    has_edges: bool,
) -> Tuple[float, float]:
    max_cols = 1
    has_multi_slice = False
    max_complexity = 1.0
    for node in nodes:
        if len(node.view.slices) > 1:
            has_multi_slice = True
        for tensor_slice in node.view.slices:
            if tensor_slice.data.ndim == 2:
                max_cols = max(max_cols, int(tensor_slice.data.shape[1]))

        node_max_rows = 1
        node_max_cols = 1
        for tensor_slice in node.view.slices:
            if tensor_slice.data.ndim == 2:
                node_max_rows = max(node_max_rows, int(tensor_slice.data.shape[0]))
                node_max_cols = max(node_max_cols, int(tensor_slice.data.shape[1]))
        node_code_lines = len(_wrap_text_full(node.code.strip(), width=30)) if node.code else 0
        node_complexity = (
            1.0
            + 0.07 * max(0, node_max_rows - 6)
            + 0.04 * max(0, node_max_cols - 6)
            + 0.05 * max(0, len(node.view.slices) - 1)
            + 0.04 * max(0, node_code_lines - 1)
        )
        max_complexity = max(max_complexity, node_complexity)

    width_scale = 1.0 + max(0, max_cols - 6) * 0.05
    if has_multi_slice:
        width_scale += 0.20
    height_scale = 1.12 if has_multi_slice else 1.0

    width = max(4.2, (2.5 * max(1, layout_cols) + 1.1) * width_scale)
    height = max(3.8, (2.35 * max(1, layout_rows) + 1.25) * height_scale)
    height = height * min(1.45, max_complexity)
    if len(nodes) <= 2 and layout_cols <= 2 and not has_multi_slice and not has_edges:
        width = max(4.4, width * 0.78)
    if has_edges and layout_cols <= 2 and layout_rows <= 1 and not has_multi_slice:
        width = max(width, 8.0)
        height = max(height, 5.3)
    return width, height


def _draw_edges(
    axis: plt.Axes,
    edges: Sequence[Tuple[int, int]],
    positions: Dict[int, Tuple[float, float]],
    box_width: float,
) -> None:
    for src, dst in edges:
        src_x, src_y = positions[src]
        dst_x, dst_y = positions[dst]
        direction = 1.0 if dst_x >= src_x else -1.0
        margin = 0.008
        start = (src_x + direction * (box_width / 2.0 + margin), src_y)
        end = (dst_x - direction * (box_width / 2.0 + margin), dst_y)
        delta_y = dst_y - src_y
        curvature = 0.0 if abs(delta_y) < 0.05 else 0.18 * np.sign(delta_y)

        axis.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.8,
                "color": "#334155",
                "shrinkA": 0,
                "shrinkB": 0,
                "connectionstyle": f"arc3,rad={curvature}",
            },
            zorder=10,
        )


def _draw_node(axis: plt.Axes, node: RenderNode) -> None:
    view = node.view
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("#F8FAFC")
    for spine in axis.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(1.2)

    axis.text(
        0.03,
        0.97,
        f"{node.name}  L{node.lineno}",
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color="#0F172A",
        clip_on=True,
    )
    meta = _fit_text_lines(f"shape={view.original_shape}  {view.dtype}@{view.device}", width=26, max_lines=2)
    meta_lines = meta.splitlines() if meta else []
    axis.text(
        0.03,
        0.885,
        meta,
        va="top",
        ha="left",
        fontsize=8,
        color="#334155",
        clip_on=True,
    )

    sampled_text = "sampled" if view.sampled else "full"
    stats_text = (
        f"{sampled_text} | mn={view.stats['min']:.2g} "
        f"mx={view.stats['max']:.2g} av={view.stats['mean']:.2g}"
    )
    code_lines = _wrap_text_full(node.code.strip(), width=30) if node.code else []
    stats_lines = _wrap_text_full(stats_text, width=30)

    code_font = 6.1 if len(code_lines) > 2 else 6.3
    stats_font = 5.9
    line_h_code = 0.035 if len(code_lines) <= 2 else 0.031
    line_h_stats = 0.029
    line_h_meta = 0.038

    code_block_h = line_h_code * max(1, len(code_lines)) if code_lines else 0.0
    stats_block_h = line_h_stats * max(1, len(stats_lines))
    gap_between = 0.045 if code_lines else 0.0
    text_bottom = 0.012
    stats_y = text_bottom + code_block_h + gap_between
    text_top = stats_y + stats_block_h
    meta_bottom = 0.885 - line_h_meta * max(1, len(meta_lines))

    if len(view.slices) <= 1:
        table_top_nominal = 0.79
        min_table_bottom = 0.28
        min_table_height = 0.30
    else:
        table_top_nominal = 0.76
        min_table_bottom = 0.24
        min_table_height = 0.26

    table_top = min(table_top_nominal, meta_bottom - 0.022)
    table_bottom = max(min_table_bottom, text_top + 0.045)
    if table_top - table_bottom < min_table_height:
        table_bottom = table_top - min_table_height
    table_bottom = max(0.18, table_bottom)
    _draw_slice_tables(axis, view, bbox=(0.04, table_bottom, 0.92, table_top - table_bottom))

    axis.text(
        0.03,
        stats_y,
        "\n".join(stats_lines),
        va="bottom",
        ha="left",
        fontsize=stats_font,
        color="#334155",
        clip_on=True,
    )

    if code_lines:
        axis.text(
            0.03,
            text_bottom,
            "\n".join(code_lines),
            va="bottom",
            ha="left",
            fontsize=code_font,
            color="#475569",
            clip_on=True,
        )


def _draw_slice_tables(axis: plt.Axes, view: TensorView, bbox: Tuple[float, float, float, float]) -> None:
    x0, y0, width, height = bbox
    slices = list(view.slices)
    if not slices:
        return

    if len(slices) == 1:
        sub_axis = axis.inset_axes([x0, y0, width, height], transform=axis.transAxes)
        _draw_single_table(sub_axis, slices[0], show_label=False, total_slices=1)
        return

    count = len(slices)
    if count <= 2:
        cols = count
    elif count <= 6:
        cols = 2
    else:
        cols = 3
    rows = int(np.ceil(count / cols))

    col_gap = 0.02
    row_gap = 0.02
    cell_w = (width - col_gap * (cols - 1)) / cols
    cell_h = (height - row_gap * (rows - 1)) / rows

    for idx, slice_view in enumerate(slices):
        row = idx // cols
        col = idx % cols
        left = x0 + col * (cell_w + col_gap)
        bottom = y0 + (rows - 1 - row) * (cell_h + row_gap)
        sub_axis = axis.inset_axes([left, bottom, cell_w, cell_h], transform=axis.transAxes)
        _draw_single_table(sub_axis, slice_view, show_label=True, total_slices=count)


def _draw_single_table(axis: plt.Axes, tensor_slice: TensorSlice, *, show_label: bool, total_slices: int) -> None:
    axis.set_axis_off()
    axis.set_facecolor("#F8FAFC")

    if show_label:
        axis.text(
            0.03,
            0.97,
            tensor_slice.label,
            va="top",
            ha="left",
            fontsize=6.1,
            color="#334155",
            clip_on=True,
        )
        table_bbox = [0.04, 0.08, 0.92, 0.64]
    else:
        table_bbox = [0.04, 0.05, 0.92, 0.90]

    matrix = tensor_slice.data
    rows = int(matrix.shape[0])
    cols = int(matrix.shape[1]) if matrix.ndim > 1 else 1
    compact_numbers = total_slices > 1

    formatted_rows = [[_format_value(float(v), compact=compact_numbers) for v in row] for row in matrix]
    max_chars = max((len(text) for row in formatted_rows for text in row), default=1)
    table = axis.table(
        cellText=formatted_rows,
        cellLoc="center",
        loc="center",
        bbox=table_bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(_table_font_size(rows, cols, total_slices, max_chars))

    for (row, col), cell in table.get_celld().items():
        if row < 0 or col < 0:
            continue
        shade = "#F1F5F9" if (row + col) % 2 else "#FFFFFF"
        cell.set_facecolor(shade)
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.45)
        cell.get_text().set_color("#0F172A")
        cell.get_text().set_clip_on(True)
        cell.set_clip_on(True)


def _table_font_size(rows: int, cols: int, total_slices: int, max_chars: int) -> float:
    largest = max(rows, cols)
    if largest <= 4:
        size = 8.2
    elif largest <= 6:
        size = 6.9
    elif largest <= 8:
        size = 4.8
    elif largest <= 12:
        size = 4.3
    else:
        size = 3.9

    if max_chars >= 7:
        size -= 0.8
    elif max_chars >= 6:
        size -= 0.4

    if total_slices <= 1:
        return max(3.4, size)
    if total_slices <= 4:
        size -= 1.8
    else:
        size -= 2.3
    return max(2.7, size)


def _fit_text_lines(text: str, *, width: int, max_lines: int) -> str:
    wrapped = textwrap.wrap(
        text,
        width=max(8, width),
        break_long_words=True,
        break_on_hyphens=False,
    )
    if not wrapped:
        return ""
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)

    head = wrapped[:max_lines]
    last = head[-1]
    if len(last) > 3:
        head[-1] = last[:-3] + "..."
    else:
        head[-1] = "..."
    return "\n".join(head)


def _wrap_text_full(text: str, *, width: int) -> List[str]:
    if not text:
        return []
    wrapped = textwrap.wrap(
        text,
        width=max(8, width),
        break_long_words=True,
        break_on_hyphens=False,
    )
    return wrapped if wrapped else [text]


def _format_value(value: float, *, compact: bool = False) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"

    abs_value = abs(value)
    if compact:
        if 0 < abs_value < 0.01:
            text = f"{value:.2f}"
        elif abs_value >= 100:
            text = f"{value:.1e}"
        elif abs_value >= 10:
            text = f"{value:.0f}"
        else:
            text = f"{value:.1f}"
        return _normalize_numeric(text, max_len=6)

    if abs_value >= 1000:
        text = f"{value:.2e}"
    else:
        text = f"{value:.2f}"
    return _normalize_numeric(text, max_len=8)


def _normalize_numeric(text: str, *, max_len: int) -> str:
    if "e" not in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    if len(text) <= max_len:
        return text

    value = float(text)
    scientific = f"{value:.1e}"
    if len(scientific) <= max_len:
        return scientific
    return f"{value:.0e}"


def _sample_matrix(matrix: torch.Tensor, limit: int) -> Tuple[np.ndarray, bool]:
    row_idx = _sample_indices(int(matrix.shape[0]), limit, matrix.device)
    col_idx = _sample_indices(int(matrix.shape[1]), limit, matrix.device)
    sampled = int(matrix.shape[0]) > limit or int(matrix.shape[1]) > limit
    reduced = matrix.index_select(0, row_idx).index_select(1, col_idx)
    return reduced.to("cpu").to(torch.float32).numpy(), sampled


def _stacked_shape(slices: Sequence[TensorSlice]) -> Tuple[int, ...]:
    if not slices:
        return (0, 0, 0)
    first = slices[0].data
    return (len(slices), int(first.shape[0]), int(first.shape[1]))


def _sample_indices(length: int, limit: int, device: torch.device) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if length <= limit:
        return torch.arange(length, dtype=torch.long, device=device)
    values = np.linspace(0, length - 1, num=limit, dtype=np.int64)
    unique = np.unique(values)
    return torch.as_tensor(unique, dtype=torch.long, device=device)


def _stats_from_slices(slices: Sequence[TensorSlice]) -> Dict[str, float]:
    if not slices:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    flattened = np.concatenate([tensor_slice.data.reshape(-1) for tensor_slice in slices], axis=0)
    return {
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
        "mean": float(np.mean(flattened)),
    }


def _unique_ints(values: Sequence[int]) -> List[int]:
    seen = set()
    unique: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
