from __future__ import annotations

import ast
import contextlib
import datetime as dt
import inspect
import linecache
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .infer import infer_call_arguments
from .render import RenderNode, RenderOutput, build_tensor_view, render_chain


@dataclass
class FrameSnapshot:
    signatures: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class FrameState:
    snapshot: FrameSnapshot
    prev_lineno: int
    prev_code: str


class MonitorRuntime:
    def __init__(self) -> None:
        self.enabled = False
        self.arg_hints: Dict[str, Any] = {}
        self._nodes: list[RenderNode] = []
        self._frame_states: Dict[int, FrameState] = {}
        self._previous_trace: Any = None
        self._workspace_root = Path.cwd().resolve()
        self._package_root = Path(__file__).resolve().parent

        self._line_file_key: Optional[str] = None
        self._line_start: Optional[int] = None
        self._line_end: Optional[int] = None

        self._show = False
        self._save = False
        self._save_dir = Path("monitorch_outputs")
        self._file_prefix = "tensor_flow"
        self._max_nodes = 12
        self._max_vector = 32
        self._max_matrix = 8
        self._max_slices = 6

    def on(
        self,
        *,
        show: Optional[bool] = None,
        save: bool = False,
        save_dir: str | Path = "monitorch_outputs",
        file_prefix: str = "tensor_flow",
        max_nodes: int = 12,
        max_vector: int = 32,
        max_matrix: int = 8,
        max_slices: int = 6,
    ) -> "MonitorRuntime":
        self._workspace_root = Path.cwd().resolve()
        self._show = _in_notebook() if show is None else bool(show)
        self._save = bool(save)
        self._save_dir = Path(save_dir)
        self._file_prefix = file_prefix
        self._max_nodes = max(1, int(max_nodes))
        self._max_vector = max(4, int(max_vector))
        self._max_matrix = max(4, int(max_matrix))
        self._max_slices = max(1, int(max_slices))

        if self.enabled:
            return self

        self.enabled = True
        self._install_trace()
        self._bootstrap_current_frame()
        return self

    def off(
        self,
        *,
        render: bool = True,
        filename: str | Path | None = None,
        show: Optional[bool] = None,
        save: Optional[bool] = None,
        clear: bool = True,
    ) -> Optional[RenderOutput]:
        if not self.enabled:
            if render and self._nodes:
                return self.flush(filename=filename, show=show, save=save, clear=clear)
            return None

        self.enabled = False
        self._uninstall_trace()
        self._frame_states.clear()

        if not render:
            return None
        return self.flush(filename=filename, show=show, save=save, clear=clear)

    def flush(
        self,
        *,
        filename: str | Path | None = None,
        show: Optional[bool] = None,
        save: Optional[bool] = None,
        clear: bool = True,
    ) -> Optional[RenderOutput]:
        if not self._nodes:
            return None

        should_show = self._show if show is None else bool(show)
        should_save = self._save if save is None else bool(save)
        save_path: Optional[Path] = None

        if should_save:
            if filename is None:
                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self._save_dir / f"{self._file_prefix}_{stamp}.png"
            else:
                provided = Path(filename)
                save_path = provided if provided.is_absolute() else self._save_dir / provided

        output = render_chain(
            self._nodes,
            show=should_show,
            save_path=save_path,
            title="monitorch Tensor Flow",
        )

        if clear:
            self._nodes.clear()
        return output

    def lines(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        *,
        file: str | Path | None = None,
    ) -> "MonitorRuntime":
        if start is None and end is None and file is None:
            self._line_file_key = None
            self._line_start = None
            self._line_end = None
            return self

        if start is None or end is None:
            raise ValueError("Both start and end must be provided, or both omitted to clear filter.")
        if start <= 0 or end <= 0 or end < start:
            raise ValueError("Invalid line range.")

        if file is None:
            caller = self._external_caller_frame()
            if caller is None:
                raise RuntimeError("Unable to infer caller file for lines().")
            filename = caller.f_code.co_filename
        else:
            filename = str(file)

        self._line_file_key = self._file_key(filename)
        self._line_start = int(start)
        self._line_end = int(end)
        return self

    def args(self, **kwargs: Any) -> "MonitorRuntime":
        self.arg_hints.update(kwargs)
        return self

    def run(self, fn: Callable[..., Any], /, **overrides: Any) -> Any:
        caller = self._external_caller_frame()
        local_scope = caller.f_locals if caller is not None else {}
        global_scope = caller.f_globals if caller is not None else {}
        positional, keyword = infer_call_arguments(
            fn,
            overrides=overrides,
            hints=self.arg_hints,
            local_scope=local_scope,
            global_scope=global_scope,
        )
        return fn(*positional, **keyword)

    def reset(self) -> None:
        self.off(render=False)
        self._nodes.clear()
        self._frame_states.clear()
        self.arg_hints.clear()
        self.lines()

    def _install_trace(self) -> None:
        import sys

        self._previous_trace = sys.gettrace()
        sys.settrace(self._trace)
        threading.settrace(self._trace)

    def _uninstall_trace(self) -> None:
        import sys

        sys.settrace(self._previous_trace)
        threading.settrace(None)

    def _bootstrap_current_frame(self) -> None:
        frame = inspect.currentframe()
        while frame is not None:
            frame = frame.f_back
            if frame is None:
                return
            if self._is_internal_frame(frame):
                continue
            if not self._should_track_frame(frame):
                continue

            frame.f_trace = self._trace
            self._frame_states[id(frame)] = FrameState(
                snapshot=self._collect_snapshot(frame),
                prev_lineno=frame.f_lineno,
                prev_code=self._source_line(frame.f_code.co_filename, frame.f_lineno),
            )

    def _trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        if not self.enabled:
            return None

        if event == "call":
            if self._should_track_frame(frame):
                self._frame_states[id(frame)] = FrameState(
                    snapshot=self._collect_snapshot(frame),
                    prev_lineno=frame.f_lineno,
                    prev_code=self._source_line(frame.f_code.co_filename, frame.f_lineno),
                )
                return self._trace
            return None

        state = self._frame_states.get(id(frame))
        if state is None:
            if event == "line" and self._should_track_frame(frame):
                self._frame_states[id(frame)] = FrameState(
                    snapshot=self._collect_snapshot(frame),
                    prev_lineno=frame.f_lineno,
                    prev_code=self._source_line(frame.f_code.co_filename, frame.f_lineno),
                )
                return self._trace
            return None

        if event == "line":
            current = self._collect_snapshot(frame)
            self._record_deltas(
                frame=frame,
                lineno=state.prev_lineno,
                code=state.prev_code,
                before=state.snapshot,
                after=current,
            )
            state.snapshot = current
            state.prev_lineno = frame.f_lineno
            state.prev_code = self._source_line(frame.f_code.co_filename, frame.f_lineno)
            return self._trace

        if event == "return":
            current = self._collect_snapshot(frame)
            self._record_deltas(
                frame=frame,
                lineno=state.prev_lineno,
                code=state.prev_code,
                before=state.snapshot,
                after=current,
            )
            self._frame_states.pop(id(frame), None)
            return self._trace

        return self._trace

    def _record_deltas(
        self,
        *,
        frame: FrameType,
        lineno: int,
        code: str,
        before: FrameSnapshot,
        after: FrameSnapshot,
    ) -> None:
        if lineno <= 0:
            return
        if not self._line_allowed(frame.f_code.co_filename, lineno):
            return

        changed = []
        for name, signature in after.signatures.items():
            if before.signatures.get(name) != signature:
                changed.append(name)

        if not changed:
            return

        before_tensor_names = tuple(before.tensors.keys())
        dependencies_by_target = _line_target_dependencies(code, before_tensor_names)
        line_reads = _line_tensor_reads(code, before_tensor_names)

        for name in sorted(changed):
            tensor = after.tensors[name]
            dependencies = dependencies_by_target.get(name)
            if dependencies is None:
                if line_reads and (len(changed) == 1 or name in line_reads):
                    dependencies = line_reads
                elif name in before.tensors:
                    dependencies = (name,)
                else:
                    dependencies = ()
            self._append_node(
                name=name,
                filename=frame.f_code.co_filename,
                lineno=lineno,
                code=code,
                tensor=tensor,
                dependencies=dependencies,
            )

    def _append_node(
        self,
        *,
        name: str,
        filename: str,
        lineno: int,
        code: str,
        tensor: torch.Tensor,
        dependencies: Sequence[str] = (),
    ) -> None:
        node = RenderNode(
            name=name,
            filename=filename,
            lineno=lineno,
            code=code.strip(),
            view=build_tensor_view(
                tensor,
                max_vector=self._max_vector,
                max_matrix=self._max_matrix,
                max_slices=self._max_slices,
            ),
            dependencies=tuple(_unique_strs(dependencies)),
        )
        self._nodes.append(node)
        if len(self._nodes) > self._max_nodes:
            self._nodes.pop(0)

    def _collect_snapshot(self, frame: FrameType) -> FrameSnapshot:
        snapshot = FrameSnapshot()
        for name, value in frame.f_locals.items():
            if isinstance(value, torch.Tensor):
                snapshot.tensors[name] = value
                snapshot.signatures[name] = _tensor_signature(value)
        return snapshot

    def _line_allowed(self, filename: str, lineno: int) -> bool:
        file_key = self._file_key(filename)
        if self._line_file_key is not None and file_key != self._line_file_key:
            return False
        if self._line_start is not None and lineno < self._line_start:
            return False
        if self._line_end is not None and lineno > self._line_end:
            return False
        return True

    def _should_track_frame(self, frame: FrameType) -> bool:
        filename = frame.f_code.co_filename
        file_key = self._file_key(filename)

        if self._line_file_key is not None and file_key != self._line_file_key:
            return False

        if filename.startswith("<ipython-input"):
            return True
        if filename.startswith("<"):
            return False

        path = self._safe_resolve(filename)
        if path is None:
            return False
        if self._is_own_file(path):
            return False
        if self._line_file_key is None and not self._is_under_workspace(path):
            return False
        return True

    def _is_internal_frame(self, frame: FrameType) -> bool:
        filename = frame.f_code.co_filename
        if filename.startswith("<"):
            return False
        path = self._safe_resolve(filename)
        return bool(path and self._is_own_file(path))

    def _external_caller_frame(self) -> Optional[FrameType]:
        frame = inspect.currentframe()
        while frame is not None:
            frame = frame.f_back
            if frame is None:
                return None
            if self._is_internal_frame(frame):
                continue
            return frame
        return None

    def _is_own_file(self, path: Path) -> bool:
        try:
            path.relative_to(self._package_root)
            return True
        except ValueError:
            return False

    def _is_under_workspace(self, path: Path) -> bool:
        try:
            path.relative_to(self._workspace_root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _safe_resolve(filename: str) -> Optional[Path]:
        try:
            return Path(filename).resolve()
        except OSError:
            return None

    @staticmethod
    def _file_key(filename: str) -> str:
        if filename.startswith("<"):
            return filename
        path = MonitorRuntime._safe_resolve(filename)
        if path is None:
            return filename.lower()
        return str(path).lower()

    @staticmethod
    def _source_line(filename: str, lineno: int) -> str:
        if lineno <= 0:
            return ""
        return linecache.getline(filename, lineno).strip()


def _line_target_dependencies(code: str, tensor_names: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    parsed = _safe_parse_line(code)
    if parsed is None:
        return {}

    tensor_name_set = set(tensor_names)
    dependencies_by_target: Dict[str, Tuple[str, ...]] = {}

    for statement in parsed.body:
        if isinstance(statement, ast.Assign):
            dependencies = _expression_tensor_names(statement.value, tensor_name_set)
            for target in statement.targets:
                for name in _target_names(target):
                    dependencies_by_target[name] = dependencies
        elif isinstance(statement, ast.AnnAssign):
            dependencies = _expression_tensor_names(statement.value, tensor_name_set) if statement.value else ()
            for name in _target_names(statement.target):
                dependencies_by_target[name] = dependencies
        elif isinstance(statement, ast.AugAssign):
            target_names = _target_names(statement.target)
            dependencies = _expression_tensor_names(statement.value, tensor_name_set)
            for name in target_names:
                merged = list(dependencies)
                if name in tensor_name_set:
                    merged.insert(0, name)
                dependencies_by_target[name] = tuple(_unique_strs(merged))

    return dependencies_by_target


def _line_tensor_reads(code: str, tensor_names: Sequence[str]) -> Tuple[str, ...]:
    parsed = _safe_parse_line(code)
    if parsed is None:
        return ()
    return _expression_tensor_names(parsed, set(tensor_names))


def _expression_tensor_names(node: ast.AST, tensor_name_set: set[str]) -> Tuple[str, ...]:
    collector = _NameCollector()
    collector.visit(node)
    return tuple(name for name in _unique_strs(collector.names) if name in tensor_name_set)


def _target_names(node: ast.AST) -> Tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        names: List[str] = []
        for element in node.elts:
            names.extend(_target_names(element))
        return tuple(names)
    return ()


def _safe_parse_line(code: str) -> Optional[ast.Module]:
    stripped = code.strip()
    if not stripped:
        return None
    try:
        return ast.parse(stripped, mode="exec")
    except SyntaxError:
        return None


class _NameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: List[str] = []

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.append(node.id)
        self.generic_visit(node)


def _unique_strs(values: Sequence[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _tensor_signature(tensor: torch.Tensor) -> Tuple[Any, ...]:
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False

    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


_RUNTIME = MonitorRuntime()


def on(**kwargs: Any) -> MonitorRuntime:
    return _RUNTIME.on(**kwargs)


def off(**kwargs: Any) -> Optional[RenderOutput]:
    return _RUNTIME.off(**kwargs)


def flush(**kwargs: Any) -> Optional[RenderOutput]:
    return _RUNTIME.flush(**kwargs)


def lines(start: Optional[int] = None, end: Optional[int] = None, *, file: str | Path | None = None) -> MonitorRuntime:
    return _RUNTIME.lines(start=start, end=end, file=file)


def args(**kwargs: Any) -> MonitorRuntime:
    return _RUNTIME.args(**kwargs)


def run(fn: Callable[..., Any], /, **overrides: Any) -> Any:
    return _RUNTIME.run(fn, **overrides)


def reset() -> None:
    _RUNTIME.reset()


@contextlib.contextmanager
def watch(**kwargs: Any):
    runtime = on(**kwargs)
    try:
        yield runtime
    finally:
        off()
