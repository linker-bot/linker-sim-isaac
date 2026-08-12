from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KALEIDOSCOPE = ROOT / "src" / "linkerbot_sim" / "kaleidoscope"
TRAINING = ROOT / "src" / "linkerbot_sim" / "training" / "skrl"
NEWTON_MANAGER = (
    ROOT / "src" / "linkerbot_sim" / "isaac" / "physics" / "newton" / "manager.py"
)


def _calls(path: Path) -> list[tuple[str, str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node
        while owner in parents and not isinstance(
            owner, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            owner = parents[owner]
        function = owner.name if isinstance(owner, ast.FunctionDef) else "<module>"
        result.append((function, node.func.attr, node.lineno))
    return result


def test_native_hot_path_has_no_host_array_conversion() -> None:
    excluded = {
        KALEIDOSCOPE / "adapters" / "gymnasium.py",
        KALEIDOSCOPE / "checkpoint.py",
    }
    violations = []
    for path in KALEIDOSCOPE.rglob("*.py"):
        if path in excluded:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                if any(name == "numpy" or name.startswith("numpy.") for name in names):
                    violations.append((str(path), "numpy import", node.lineno))
        for function, attribute, line in _calls(path):
            if attribute not in {"cpu", "numpy", "tolist", "item", "nonzero"}:
                continue
            # Native/debug step 用一次同步 scalar guard 表达可恢复的 pending-reset
            # 生命周期错误；训练 tokenized_step 不经过这个入口。下方独立测试将豁免
            # 收紧为恰好一次 item()，防止未来在同一函数偷渡第二个 host readback。
            if path.name == "runtime.py" and function == "step" and attribute == "item":
                continue
            violations.append((str(path), f"{function}.{attribute}", line))
    assert violations == []


def test_native_step_has_exactly_one_pending_reset_host_readback() -> None:
    calls = [
        attribute
        for function, attribute, _line in _calls(KALEIDOSCOPE / "runtime.py")
        if function == "step"
    ]
    assert calls.count("item") == 1
    assert "_assert_async" not in calls
    assert set(calls).isdisjoint({"cpu", "numpy", "tolist", "nonzero"})


def test_tokenized_step_has_no_host_scalar_or_array_readback() -> None:
    path = KALEIDOSCOPE / "runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "tokenized_step"
    )
    attributes = {
        node.func.attr
        for node in ast.walk(target)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    named_calls = {
        node.func.id
        for node in ast.walk(target)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert attributes.isdisjoint({"cpu", "numpy", "tolist", "item", "nonzero"})
    assert "bool" not in named_calls


def test_skrl_hot_path_only_syncs_in_explicit_metric_export() -> None:
    violations = []
    for path in TRAINING.rglob("*.py"):
        for function, attribute, line in _calls(path):
            if attribute not in {"cpu", "numpy", "tolist", "item"}:
                continue
            if (
                path.name == "final_observation_ppo.py"
                and function == "export_training_metrics"
            ):
                continue
            violations.append((str(path), f"{function}.{attribute}", line))
    assert violations == []


def test_newton_state_flush_does_not_build_masks_through_host_arrays() -> None:
    tree = ast.parse(
        NEWTON_MANAGER.read_text(encoding="utf-8"), filename=str(NEWTON_MANAGER)
    )
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_flush_cold_state_updates"
    )
    violations: list[tuple[str, int]] = []
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in {
            "_host_array",
            "make_selected_world_mask",
        }:
            violations.append((node.func.id, node.lineno))
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "array",
            "numpy",
            "isin",
        }:
            violations.append((node.func.attr, node.lineno))
    assert violations == []
