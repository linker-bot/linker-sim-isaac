#!/usr/bin/env python3
"""重建并校验最终架构清单、双语模块图和配置审计指纹。

这个脚本刻意把“人工决定”和“机械事实”分开：产品边界、facade、依赖方向与禁止项
保存在 ``architecture/module_disposition.yaml`` 的手写区；文件列表、数量和 SHA-256
则从工作树确定性生成。这样移动或删除模块后不会靠人工抄写 300 多行 inventory，也不会
因为只更新了其中一份文档而让中英文事实漂移。
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib.util import resolve_name
from pathlib import Path
import re
import sys

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "linkerbot_sim"
MANIFEST_PATH = REPO_ROOT / "architecture" / "module_disposition.yaml"
ALLOWLIST_PATH = (
    REPO_ROOT / "tests" / "data" / "config_audit" / "hardcoded_allowlist.yaml"
)
MODULE_MAP_PATHS = {
    "en": REPO_ROOT / "docs" / "en" / "development" / "module-map.md",
    "zh-CN": REPO_ROOT / "docs" / "zh-CN" / "development" / "module-map.md",
}
GENERATED_START = "# generated-inventory:start"
GENERATED_END = "# generated-inventory:end"
RUNTIME_LABELS = frozenset({"pure", "Isaac main thread", "cuRobo/CUDA"})
PRODUCT_PHYSICS_SELECTIONS = {
    "mirror": frozenset({("physx", "cpu"), ("newton", "cpu"), ("newton", "cuda")}),
    "kaleidoscope": frozenset({("physx", "cuda"), ("newton", "cuda")}),
}
KIT_EXPERIENCE_PATHS = frozenset(
    {
        "apps/linkerbot_sim.mirror.physx.python.kit",
        "apps/linkerbot_sim.mirror.newton.python.kit",
        "apps/linkerbot_sim.mirror.newton_render.python.kit",
        "apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit",
        "apps/linkerbot_sim.kaleidoscope.newton.python.kit",
        "apps/linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit",
        "apps/linkerbot_sim.kaleidoscope.newton_viewport.python.kit",
    }
)
MODE_PROFILE_PATHS = {
    "mirror": frozenset(
        {
            "configs/modes/mirror/physx_cpu.yaml",
            "configs/modes/mirror/physx_cpu_hybrid.yaml",
            "configs/modes/mirror/newton_cpu.yaml",
            "configs/modes/mirror/newton_cuda.yaml",
        }
    ),
    "kaleidoscope": frozenset(
        {
            "configs/modes/kaleidoscope/physx_cuda.yaml",
            "configs/modes/kaleidoscope/newton_cuda.yaml",
        }
    ),
}
# 这些目录保存公开配置 catalog 的 canonical leaf。多数 leaf 由产品 composition 引用；
# 少数尚未进入产品能力矩阵的严格 leaf也在这里冻结，避免用“缺文件”隐式表达能力状态。
# 逐目录冻结文件集合，防止已删除的 alias 或第二套事实源只靠刷新 inventory 重新混入。
CANONICAL_CONFIGURATION_FILE_SETS = {
    # 两种 scene schema 不兼容，目录 namespace 与产品边界保持一致；mode 必须使用
    # mirror/... 或 kaleidoscope/... 引用，根目录不得恢复平铺的弱归属命名。
    "configs/scenes": frozenset(),
    "configs/scenes/mirror": frozenset(
        {
            "configs/scenes/mirror/scene3.yaml",
            "configs/scenes/mirror/scene3_hybrid.yaml",
        }
    ),
    "configs/scenes/kaleidoscope": frozenset(
        {"configs/scenes/kaleidoscope/tblock_push.yaml"}
    ),
    "configs/physics/physx": frozenset(
        {"configs/physics/physx/cpu.yaml", "configs/physics/physx/cuda.yaml"}
    ),
    # physics profile 目录按 engine/execution 正交展开。Mirror 使用 Newton CPU/CUDA，
    # Kaleidoscope 仍严格限制为 CUDA；产品支持矩阵在上方独立冻结，不能只从目录对称性
    # 推断某个 mode 的合法 execution。
    "configs/physics/newton": frozenset(
        {"configs/physics/newton/cpu.yaml", "configs/physics/newton/cuda.yaml"}
    ),
    # Kaleidoscope 的 control writer 是固定产品合同，不再用零自由度 leaf 重复声明；
    # Mirror 的交互控制策略与物理后端正交，因此两套 composition 共用一份 profile。
    "configs/control": frozenset(
        {
            "configs/control/hybrid_force_position.yaml",
            "configs/control/mirror.yaml",
        }
    ),
    "configs/controllers/physx": frozenset(
        {
            "configs/controllers/physx/arm_controller.yaml",
            "configs/controllers/physx/hand_controller.yaml",
        }
    ),
    "configs/controllers/newton": frozenset(
        {
            "configs/controllers/newton/arm_controller.yaml",
            "configs/controllers/newton/hand_controller.yaml",
        }
    ),
    # cuRobo 数值参数只有这一套 canonical schema：Mirror 同时装配 IK/planner，
    # Kaleidoscope 只装配 batch IK；模板不得重新进入可运行的 profile namespace。
    "configs/curobo": frozenset(
        {
            "configs/curobo/kaleidoscope_batch_ik.yaml",
            "configs/curobo/mirror.yaml",
        }
    ),
    # Mirror 请求缺省值与 cuRobo solver 容量分属产品策略和数值后端，不能重新混入
    # configs/planning/curobo 子树。
    "configs/planning": frozenset({"configs/planning/mirror.yaml"}),
    # Kaleidoscope viewport 是 launch-only 配置，但仍是 configuration root 下受审计的
    # canonical leaf；单文件不再额外套一层 kaleidoscope/viewport 子目录。
    "configs/visualization": frozenset({"configs/visualization/kaleidoscope.yaml"}),
}
FORBIDDEN_PUBLIC_CONFIGURATION_PATHS = frozenset(
    {
        "configs/kinematics",
        "configs/logging",
        "configs/planning/curobo",
        "configs/replication",
        "configs/visualization/kaleidoscope",
        "configs/controllers/example.yaml",
        "configs/curobo/example.yaml",
        "configs/objects/example.yaml",
        "configs/robots/example.yaml",
    }
)
CANONICAL_CONFIGURATION_DIRECTORY_SETS = {
    "configs/controllers": frozenset({"newton", "physx"}),
    "configs/physics": frozenset({"newton", "physx"}),
    "configs/scenes": frozenset({"kaleidoscope", "mirror"}),
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """拒绝重复 key，避免后出现的架构规则静默覆盖前一个规则。"""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing architecture mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ModuleFact:
    """模块地图的一行；中英文只允许 responsibility 与链接标签不同。"""

    path: Path
    module: str
    group: str
    layer: str
    runtime: str
    classification: str
    documentation_target: str


def _manifest() -> dict[str, object]:
    value = yaml.load(
        MANIFEST_PATH.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
    )
    if not isinstance(value, dict):
        raise ValueError(f"{MANIFEST_PATH} top level must be a mapping")
    if value.get("schema_version") != 2:
        raise ValueError("only module disposition schema_version=2 is supported")
    return value


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    suffix = ".".join(parts)
    return "linkerbot_sim" + (f".{suffix}" if suffix else "")


def _source_modules() -> dict[str, Path]:
    modules = {_module_name(path): path for path in SOURCE_ROOT.rglob("*.py")}
    if len(modules) != len(tuple(SOURCE_ROOT.rglob("*.py"))):
        raise ValueError("multiple Python files resolved to the same module path")
    return dict(sorted(modules.items()))


def _module_group(module: str) -> str:
    parts = module.split(".")
    return "root" if len(parts) == 1 else parts[1]


def _prefix_matches(module: str, prefix: str) -> bool:
    # 末尾带点表示“只匹配后代，不匹配 package 自身”，用于约束 training 只能从
    # Kaleidoscope facade 导入，而不能直连其实现模块。
    if prefix.endswith("."):
        return module.startswith(prefix)
    return module == prefix or module.startswith(prefix + ".")


def _classification(
    module: str, *, facades: Mapping[str, object], owner_paths: frozenset[str]
) -> str:
    if module in facades:
        return "documented facade"
    if module in owner_paths:
        return "owner path"
    return "internal"


def _runtime(module: str, module_map: Mapping[str, object]) -> str:
    facades = _mapping(module_map.get("_facades"), label="facades")
    facade = facades.get(module)
    if isinstance(facade, Mapping):
        value = facade.get("runtime")
        if isinstance(value, str):
            return value

    for prefix in _string_sequence(
        module_map.get("pure_module_prefixes", ()), label="pure_module_prefixes"
    ):
        if _prefix_matches(module, prefix):
            return "pure"

    runtime_prefixes = _mapping(
        module_map.get("runtime_prefixes"), label="runtime_prefixes"
    )
    for label, prefixes in runtime_prefixes.items():
        if label not in RUNTIME_LABELS:
            raise ValueError(f"unknown runtime label: {label!r}")
        for prefix in _string_sequence(prefixes, label=f"runtime_prefixes.{label}"):
            if _prefix_matches(module, prefix):
                return str(label)
    return "pure"


def _module_facts(manifest: Mapping[str, object]) -> tuple[ModuleFact, ...]:
    module_map = dict(_mapping(manifest.get("module_map"), label="module_map"))
    facades = _mapping(manifest.get("public_facades"), label="public_facades")
    module_map["_facades"] = facades
    owner_paths = frozenset(
        _string_sequence(module_map.get("owner_paths", ()), label="owner_paths")
    )
    documentation_targets = _mapping(
        module_map.get("documentation_targets"), label="documentation_targets"
    )
    configured_groups = _string_sequence(
        module_map.get("group_order"), label="group_order"
    )
    configured_group_set = set(configured_groups)
    group_layers = _mapping(module_map.get("group_layers"), label="group_layers")
    allowed_layers = set(
        _string_sequence(
            _mapping(manifest.get("layers"), label="layers").get("order"),
            label="layers.order",
        )
    )
    if set(group_layers) != configured_group_set:
        raise ValueError(
            "module_map.group_layers must use exactly the same groups as group_order"
        )
    unknown_layers = sorted(set(group_layers.values()) - allowed_layers)
    if unknown_layers:
        raise ValueError(f"group_layers contains unknown layer(s): {unknown_layers}")

    facts: list[ModuleFact] = []
    for module, path in _source_modules().items():
        group = _module_group(module)
        if group not in configured_group_set:
            raise ValueError(
                f"new top-level package {group!r} is not yet in module_map.group_order"
            )
        target = documentation_targets.get(group)
        if not isinstance(target, str) or not target:
            raise ValueError(f"module group {group!r} is missing a documentation target")
        runtime = _runtime(module, module_map)
        if runtime not in RUNTIME_LABELS:
            raise ValueError(f"{module} uses unknown runtime {runtime!r}")
        facts.append(
            ModuleFact(
                path=path,
                module=module,
                group=group,
                layer=str(group_layers[group]),
                runtime=runtime,
                classification=_classification(
                    module, facades=facades, owner_paths=owner_paths
                ),
                documentation_target=target,
            )
        )

    order_index = {group: index for index, group in enumerate(configured_groups)}
    return tuple(sorted(facts, key=lambda item: (order_index[item.group], item.module)))


_RESPONSIBILITY_LEAVES = {
    "actions": (
        "fixed-shape action validation and application",
        "定形 action 校验与写入",
    ),
    "adapters": ("external API adapter namespace", "外部 API adapter 命名空间"),
    "admission": (
        "bounded request admission and response ownership",
        "有界请求准入与响应所有权",
    ),
    "app": ("process lifecycle and service composition", "进程生命周期与服务组合"),
    "assets": ("typed asset profile contracts", "强类型资产 profile 合同"),
    "bootstrap": (
        "composition root and resource ownership transfer",
        "组合根与资源所有权移交",
    ),
    "checkpoint": (
        "explicit cold persistent checkpoint boundary",
        "显式冷持久化 checkpoint 边界",
    ),
    "cli": ("command-line parsing and process startup", "命令行解析与进程启动"),
    "catalog": (
        "sole project profile YAML I/O and composition owner",
        "项目 profile YAML I/O 与组合的唯一 owner",
    ),
    "common": ("shared immutable configuration primitives", "共享不可变配置原语"),
    "controller": (
        "owner-thread command dispatch and safety controls",
        "owner 线程指令分派与安全控制",
    ),
    "env": ("training environment lifecycle", "训练环境生命周期"),
    "training_port": (
        "public CUDA training environment protocol",
        "公开 CUDA 训练环境 protocol",
    ),
    "extensions": (
        "Isaac extension enablement and exclusivity audit",
        "Isaac extension 启用与排他审计",
    ),
    "factory": ("validated concrete runtime construction", "验收后的具体 runtime 构造"),
    "ik": ("device-native batched inverse kinematics", "设备原生批量逆运动学"),
    "interface": ("product interface namespace", "产品接口命名空间"),
    "isaac_adapter": ("Isaac runtime adapter boundary", "Isaac runtime adapter 边界"),
    "isaac_views": ("fixed-shape Isaac tensor views", "定形 Isaac tensor view"),
    "kaleidoscope": (
        "Kaleidoscope strict configuration root",
        "Kaleidoscope 严格配置根",
    ),
    "linear_motion": (
        "synchronous device-native linear motion",
        "同步设备原生线性运动",
    ),
    "loader": (
        "the sole project YAML loading and composition owner",
        "项目唯一 YAML 加载与组合 owner",
    ),
    "manager": (
        "physics owner registry and lifecycle coordination",
        "物理 owner 注册与生命周期协调",
    ),
    "memory": ("CUDA rollout memory integration", "CUDA rollout memory 集成"),
    "mirror": ("Mirror strict configuration root", "Mirror 严格配置根"),
    "observations": ("device-native observation assembly", "设备原生 observation 组装"),
    "outputs": ("Mirror output configuration contracts", "Mirror 输出配置合同"),
    "physx": ("PhysX runtime ownership adapter", "PhysX runtime 所有权 adapter"),
    "physx_ports": ("PhysX CUDA tensor port contracts", "PhysX CUDA tensor port 合同"),
    "ports": ("injectable product boundary protocols", "可注入产品边界 protocol"),
    "protocol": ("strict versioned wire protocol", "严格版本化 wire protocol"),
    "registration": ("explicit Gymnasium registration", "显式 Gymnasium 注册"),
    "rendering": ("render and camera resource coordination", "渲染与相机资源协调"),
    "resets": ("batched reset and autoreset semantics", "批量 reset 与 autoreset 语义"),
    "reset": ("transactional runtime reset orchestration", "事务式 runtime reset 编排"),
    "rewards": ("device-native reward computation", "设备原生 reward 计算"),
    "runtime": (
        "resource lifecycle and simulation-step orchestration",
        "资源生命周期与仿真步进编排",
    ),
    "scenes": ("typed scene profile contracts", "强类型 scene profile 合同"),
    "session": (
        "SimulationApp, stage, and physics runtime owner",
        "SimulationApp、stage 与物理 runtime owner",
    ),
    "snapshot": (
        "owned snapshot schema and restore semantics",
        "自有 snapshot schema 与恢复语义",
    ),
    "state": ("Mirror state access and mutation", "Mirror state 读取与写入"),
    "state_api": (
        "batched state, snapshot, and clone API",
        "批量 state、snapshot 与 clone API",
    ),
    "task": ("vector task contract and step semantics", "vector task 合同与 step 语义"),
    "task_buffers": (
        "owned fixed-shape task state buffers",
        "自有定形 task 状态 buffer",
    ),
    "tasks": ("registered task implementation namespace", "已注册 task 实现命名空间"),
    "tensors": (
        "CUDA tensor validation and allocation invariants",
        "CUDA tensor 校验与分配不变量",
    ),
    "terminations": (
        "device-native termination and truncation rules",
        "设备原生终止与截断规则",
    ),
    "transport": (
        "stdin, TCP JSONL, and WebSocket ingress",
        "stdin、TCP JSONL 与 WebSocket ingress",
    ),
}


def _responsibility(fact: ModuleFact, language: str) -> str:
    if fact.module == "linkerbot_sim":
        return (
            "lightweight repository metadata facade"
            if language == "en"
            else "轻量仓库 metadata facade"
        )
    if fact.classification == "documented facade":
        return (
            f"stable lazy {fact.group} public facade"
            if language == "en"
            else f"稳定 lazy {fact.group} public facade"
        )
    leaf = fact.module.rsplit(".", maxsplit=1)[-1]
    description = _RESPONSIBILITY_LEAVES.get(leaf)
    if description is not None:
        return description[0 if language == "en" else 1]
    if fact.module.count(".") == 1:
        return (
            f"{fact.group} implementation namespace"
            if language == "en"
            else f"{fact.group} 实现命名空间"
        )
    readable = leaf.replace("_", " ")
    return (
        f"{fact.group} implementation owner for {readable}"
        if language == "en"
        else f"{fact.group} 层 {readable} 实现 owner"
    )


def _module_map(manifest: Mapping[str, object], language: str) -> str:
    facts = _module_facts(manifest)
    counts = Counter(fact.group for fact in facts)
    configured_groups = _string_sequence(
        _mapping(manifest.get("module_map"), label="module_map").get("group_order"),
        label="group_order",
    )
    groups = tuple(group for group in configured_groups if counts[group])
    facades = tuple(
        fact for fact in facts if fact.classification == "documented facade"
    )
    owners = tuple(fact for fact in facts if fact.classification == "owner path")

    if language == "en":
        title = "# Source module map"
        language_line = "Language: [English](module-map.md) | [中文](../../zh-CN/development/module-map.md)"
        intro = (
            "This generated map covers every Python module under `src/linkerbot_sim`. "
            "The v2 architecture manifest owns facade, runtime, layer, and ordering facts; "
            "run `python scripts/update_architecture_inventory.py --write` after a source move."
        )
        registry_title = "## Interface and owner registry"
        inventory_title = "## Complete inventory"
        table_header = "| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |"
        link_label = "Architecture reference"
        layer_title = "## Layer direction"
        layer_intro = (
            "Dependencies point downward. Each product owns one runtime and one Isaac "
            "session; training consumes only the Kaleidoscope public port."
        )
        runtime_bullets = (
            "- `pure`: no Kit/Isaac application is required.",
            "- `Isaac main thread`: runtime work belongs to the simulation owner thread.",
            "- `cuRobo/CUDA`: numerical work requires the configured CUDA stack.",
        )
    else:
        title = "# 源码模块图"
        language_line = "语言：[中文](module-map.md) | [English](../../en/development/module-map.md)"
        intro = (
            "本文由目标架构清单生成，完整覆盖 `src/linkerbot_sim` 下每个 Python 模块。"
            "facade、运行前提、分层和顺序由 v2 manifest 负责；移动源码后运行 "
            "`python scripts/update_architecture_inventory.py --write` 同步。"
        )
        registry_title = "## 接口与 owner 登记表"
        inventory_title = "## 完整 Inventory"
        table_header = "| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |"
        link_label = "架构参考"
        layer_title = "## 分层方向"
        layer_intro = (
            "依赖只允许向下。每个产品拥有一个 runtime 和一个 Isaac session；训练层只消费 "
            "Kaleidoscope public port。"
        )
        runtime_bullets = (
            "- `pure`：不需要启动 Kit/Isaac application。",
            "- `Isaac main thread`：runtime 操作属于仿真 owner 线程。",
            "- `cuRobo/CUDA`：数值操作需要配置指定的 CUDA stack。",
        )

    lines = [
        title,
        "",
        language_line,
        "",
        intro,
        "",
        *runtime_bullets,
        "",
        layer_title,
        "",
        layer_intro,
        "",
        "```text",
        "product interface / training",
        "            ↓",
        "   Mirror | Kaleidoscope",
        "            ↓",
        "Isaac infrastructure | numerical backends",
        "            ↓",
        "configuration + pure domain",
        "",
        "Controller/Env → Runtime → IsaacSession → concrete PhysicsRuntime",
        "```",
        "",
        registry_title,
        "",
        "<!-- module-interface-registry:start -->",
        "| Module | Classification | Runtime |",
        "| --- | --- | --- |",
    ]
    for fact in (*facades, *owners):
        lines.append(f"| `{fact.module}` | {fact.classification} | {fact.runtime} |")
    lines.extend(
        [
            "<!-- module-interface-registry:end -->",
            "",
            inventory_title,
            "",
            "<!-- module-inventory:start -->",
            "",
        ]
    )
    for group in groups:
        lines.extend(
            [
                f"### {group} ({counts[group]})",
                "",
                table_header,
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for fact in (item for item in facts if item.group == group):
            responsibility = _responsibility(fact, language)
            lines.append(
                f"| {group} | `{fact.module}` | {fact.layer} | {responsibility} | "
                f"{fact.runtime} | {fact.classification} | "
                f"[{link_label}]({fact.documentation_target}) |"
            )
        lines.append("")
    lines.extend(["<!-- module-inventory:end -->", ""])
    return "\n".join(lines)


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        return {name for item in target.elts for name in _assigned_names(item)}
    return set()


def _defined_symbol_paths(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(f"{node.name}.{member.name}")
                elif isinstance(member, ast.Assign):
                    for target in member.targets:
                        symbols.update(
                            f"{node.name}.{name}" for name in _assigned_names(target)
                        )
                elif isinstance(member, ast.AnnAssign):
                    symbols.update(
                        f"{node.name}.{name}" for name in _assigned_names(member.target)
                    )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                symbols.update(_assigned_names(target))
        elif isinstance(node, ast.AnnAssign):
            symbols.update(_assigned_names(node.target))
    return symbols


def _replace_scalar(text: str, key: str, value: object) -> str:
    pattern = re.compile(rf"^(?P<indent>\s*){re.escape(key)}:\s*.*$", re.MULTILINE)
    matches = tuple(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected {key!r} to appear exactly once in the allowlist inventory")
    return pattern.sub(rf"\g<indent>{key}: {value}", text, count=1)


def _hardcoded_allowlist() -> str:
    text = ALLOWLIST_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    inventory = _mapping(data.get("inventory"), label="allowlist.inventory")
    source_files = sorted(REPO_ROOT.glob(str(inventory["source_python_glob"])))
    scan = _mapping(inventory.get("candidate_scan"), label="candidate_scan")
    pattern = re.compile(str(scan["name_regex"]))
    candidates = sorted(
        {
            f"{path.relative_to(SOURCE_ROOT).as_posix()}:{symbol}"
            for path in source_files
            for symbol in _defined_symbol_paths(path)
            if "." not in symbol and pattern.fullmatch(symbol)
        }
    )
    candidate_sha = hashlib.sha256("\n".join(candidates).encode("utf-8")).hexdigest()
    project_files = sorted(REPO_ROOT.glob(str(inventory["project_yaml_glob"])))
    task_files = sorted(REPO_ROOT.glob(str(inventory["third_party_task_glob"])))

    for key, value in (
        ("source_python_count", len(source_files)),
        ("reviewed_candidate_count", len(candidates)),
        ("reviewed_candidate_sha256", candidate_sha),
        ("project_yaml_count", len(project_files)),
        ("third_party_task_count", len(task_files)),
    ):
        text = _replace_scalar(text, key, value)
    return text


def _relative_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(
        sorted(path.relative_to(REPO_ROOT) for path in paths if path.is_file())
    )


def _path_digest(paths: Sequence[Path]) -> str:
    payload = "".join(f"{path.as_posix()}\n" for path in paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_content_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _inventory_group(
    paths: Sequence[Path], *, with_module_facts: bool, manifest: Mapping[str, object]
) -> dict[str, object]:
    facts_by_path: dict[Path, ModuleFact] = {}
    facades: frozenset[str] = frozenset()
    owner_paths: frozenset[str] = frozenset()
    if with_module_facts:
        facts_by_path = {
            fact.path.relative_to(REPO_ROOT): fact for fact in _module_facts(manifest)
        }
        facades = frozenset(
            _mapping(manifest.get("public_facades"), label="public_facades")
        )
        module_map = _mapping(manifest.get("module_map"), label="module_map")
        owner_paths = frozenset(
            _string_sequence(module_map.get("owner_paths"), label="owner_paths")
        )
    files: list[dict[str, object]] = []
    for relative in paths:
        entry: dict[str, object] = {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(),
        }
        fact = facts_by_path.get(relative)
        if fact is not None:
            entry.update(
                {
                    "module": fact.module,
                    "group": fact.group,
                    "layer": fact.layer,
                    "owner": _nearest_owner(
                        fact.module,
                        facades=facades,
                        owner_paths=owner_paths,
                    ),
                    "runtime": fact.runtime,
                    "classification": fact.classification,
                }
            )
        files.append(entry)
    return {
        "count": len(paths),
        "path_sha256": _path_digest(paths),
        "path_content_sha256": _path_content_digest(paths),
        "files": files,
    }


def _nearest_owner(
    module: str, *, facades: frozenset[str], owner_paths: frozenset[str]
) -> str:
    """返回最长的显式 owner/facade；没有时由一级 package 自持有。"""

    candidates = [
        candidate
        for candidate in (*facades, *owner_paths)
        if _prefix_matches(module, candidate)
    ]
    if candidates:
        return max(candidates, key=len)
    parts = module.split(".")
    return module if len(parts) == 1 else ".".join(parts[:2])


def _generated_inventory(manifest: Mapping[str, object]) -> str:
    facades = _mapping(manifest.get("public_facades"), label="public_facades")
    pending = sorted(
        module
        for module, settings in facades.items()
        if not isinstance(settings, Mapping)
        or settings.get("freeze_status") != "frozen"
    )
    groups = {
        "production_python": _relative_paths(SOURCE_ROOT.rglob("*.py")),
        "test_python": _relative_paths((REPO_ROOT / "tests").rglob("*.py")),
        "configuration_yaml": _relative_paths(
            path
            for suffix in ("*.yaml", "*.yml")
            for path in (REPO_ROOT / "configs").rglob(suffix)
        ),
        "kit_experiences": _relative_paths((REPO_ROOT / "apps").glob("*.kit")),
        "entry_scripts": _relative_paths((REPO_ROOT / "scripts").glob("*.py")),
        "automation_scripts": _relative_paths((REPO_ROOT / "ci").glob("*.sh")),
        "maintained_docs": _relative_paths((REPO_ROOT / "docs").rglob("*.md")),
    }
    inventory: dict[str, object] = {
        "status": "final" if not pending else "provisional",
        "generated_by": "scripts/update_architecture_inventory.py",
        "hash_serialization": (
            "path_sha256 uses UTF-8 POSIX path plus LF; path_content_sha256 uses "
            "UTF-8 POSIX path, NUL, raw content, NUL"
        ),
        "pending_facades": pending,
    }
    for name, paths in groups.items():
        inventory[name] = _inventory_group(
            paths,
            with_module_facts=name == "production_python",
            manifest=manifest,
        )
    payload = yaml.safe_dump(
        {"generated_inventory": inventory},
        allow_unicode=True,
        sort_keys=False,
        width=120,
    ).rstrip()
    return f"{GENERATED_START}\n{payload}\n{GENERATED_END}"


def _manifest_with_generated_inventory(manifest: Mapping[str, object]) -> str:
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    if text.count(GENERATED_START) != 1 or text.count(GENERATED_END) != 1:
        raise ValueError("each manifest must contain exactly one generated inventory marker")
    before, remainder = text.split(GENERATED_START, maxsplit=1)
    _old, after = remainder.split(GENERATED_END, maxsplit=1)
    return before + _generated_inventory(manifest) + after


def _module_imports(path: Path, module: str) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            relative = "." * node.level + (node.module or "")
            try:
                base = resolve_name(relative, package)
            except (ImportError, ValueError):
                base = relative
        else:
            base = node.module or ""
        if base:
            imports.add(base)
        if node.module is None and base:
            imports.update(f"{base}.{alias.name}" for alias in node.names)
    return frozenset(imports)


def _static_exports(module: str, path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dictionaries: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        value = node.value
        if value is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or not isinstance(value, ast.Dict):
                continue
            keys = tuple(
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
            if len(keys) == len(value.keys):
                dictionaries[target.id] = keys

    values: list[tuple[str, ...]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            continue
        value = node.value
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError):
            literal = None
        if isinstance(literal, (list, tuple)) and all(
            isinstance(item, str) for item in literal
        ):
            values.append(tuple(literal))
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "sorted"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in dictionaries
        ):
            values.append(tuple(sorted(dictionaries[value.args[0].id])))
    if len(values) != 1:
        raise ValueError(f"{module} must have a statically resolvable __all__")
    if len(values[0]) != len(set(values[0])):
        raise ValueError(f"{module}.__all__ contains duplicate exports")
    return values[0]


def _facade_path(module: str) -> Path:
    suffix = module.split(".")[1:]
    return SOURCE_ROOT.joinpath(*suffix, "__init__.py")


def _architecture_violations(
    manifest: Mapping[str, object], *, require_final: bool
) -> tuple[str, ...]:
    violations: list[str] = []
    modules = _source_modules()

    # 这些集合是产品合同，不是可随 generated inventory 一起变化的机械事实。若有人同时
    # 修改目录和 manifest，普通文件存在性检查会一起“漂移成功”，因此在生成器中保留精确锚点。
    breaking_migration = _mapping(
        manifest.get("breaking_migration"), label="breaking_migration"
    )
    products = _mapping(
        breaking_migration.get("products"), label="breaking_migration.products"
    )
    for product_name, expected_selections in PRODUCT_PHYSICS_SELECTIONS.items():
        product = _mapping(
            products.get(product_name),
            label=f"breaking_migration.products.{product_name}",
        )
        selections = _physics_selection_sequence(
            product.get("physics"), label=f"{product_name}.physics"
        )
        if len(selections) != len(expected_selections) or set(selections) != set(
            expected_selections
        ):
            violations.append(
                f"{product_name} physics selection contract drift: "
                f"expected={sorted(expected_selections)}, actual={sorted(selections)}"
            )

    required_targets = _mapping(
        manifest.get("required_targets"), label="required_targets"
    )
    kit_targets = _mapping(
        required_targets.get("kit_experiences"),
        label="required_targets.kit_experiences",
    )
    kit_paths = _string_sequence(
        kit_targets.get("exact"), label="required_targets.kit_experiences.exact"
    )
    if len(kit_paths) != len(KIT_EXPERIENCE_PATHS) or set(kit_paths) != set(
        KIT_EXPERIENCE_PATHS
    ):
        violations.append(
            "formal Kit experience contract drift: "
            f"expected={sorted(KIT_EXPERIENCE_PATHS)}, actual={sorted(kit_paths)}"
        )

    configuration_targets = _mapping(
        required_targets.get("canonical_configuration"),
        label="required_targets.canonical_configuration",
    )
    configuration_paths = _string_sequence(
        configuration_targets.get("paths"),
        label="required_targets.canonical_configuration.paths",
    )
    for product_name, expected_profiles in MODE_PROFILE_PATHS.items():
        parent = Path("configs/modes") / product_name
        declared_profiles = {
            path for path in configuration_paths if Path(path).parent == parent
        }
        actual_profiles = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / parent).glob("*.yaml")
            if path.is_file()
        }
        if declared_profiles != set(expected_profiles):
            violations.append(
                f"declared {product_name} mode profile contract drift: "
                f"expected={sorted(expected_profiles)}, "
                f"actual={sorted(declared_profiles)}"
            )
        if actual_profiles != set(expected_profiles):
            violations.append(
                f"{product_name} mode profile directory drift: "
                f"expected={sorted(expected_profiles)}, actual={sorted(actual_profiles)}"
            )

    for relative_parent, expected_files in CANONICAL_CONFIGURATION_FILE_SETS.items():
        parent = Path(relative_parent)
        declared_files = {
            path for path in configuration_paths if Path(path).parent == parent
        }
        actual_files = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / parent).glob("*.yaml")
            if path.is_file()
        }
        if declared_files != set(expected_files):
            violations.append(
                f"declared canonical config set drift under {relative_parent}: "
                f"expected={sorted(expected_files)}, actual={sorted(declared_files)}"
            )
        if actual_files != set(expected_files):
            violations.append(
                f"canonical config directory drift under {relative_parent}: "
                f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
            )

    for relative in FORBIDDEN_PUBLIC_CONFIGURATION_PATHS:
        if (REPO_ROOT / relative).exists():
            violations.append(f"forbidden public configuration path exists: {relative}")

    for (
        relative_parent,
        expected_names,
    ) in CANONICAL_CONFIGURATION_DIRECTORY_SETS.items():
        actual_names = {
            path.name
            for path in (REPO_ROOT / relative_parent).iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        if actual_names != set(expected_names):
            violations.append(
                f"canonical config subdirectory drift under {relative_parent}: "
                f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
            )

    for group_name, raw_group in required_targets.items():
        group = _mapping(raw_group, label=f"required_targets.{group_name}")
        raw_paths = group.get("exact", group.get("paths"))
        paths = _string_sequence(raw_paths, label=f"required_targets.{group_name}")
        for relative in paths:
            if not (REPO_ROOT / relative).is_file():
                violations.append(f"required target missing: {relative}")
        if "exact" in group:
            parents = {Path(relative).parent for relative in paths}
            if len(parents) != 1:
                raise ValueError(f"{group_name}.exact must reside in the same directory")
            parent = REPO_ROOT / next(iter(parents))
            actual = {
                path.relative_to(REPO_ROOT).as_posix()
                for path in parent.glob("*")
                if path.is_file()
            }
            expected = set(paths)
            if actual != expected:
                violations.append(
                    f"{group_name} exact inventory drift: "
                    f"missing={sorted(expected - actual)}, "
                    f"unexpected={sorted(actual - expected)}"
                )

    facades = _mapping(manifest.get("public_facades"), label="public_facades")
    pending = []
    for module, raw_settings in facades.items():
        settings = _mapping(raw_settings, label=f"public_facades.{module}")
        status = settings.get("freeze_status")
        if status != "frozen":
            pending.append(str(module))
            continue
        path = _facade_path(str(module))
        if not path.is_file():
            violations.append(f"frozen facade is missing file: {path.relative_to(REPO_ROOT)}")
            continue
        actual = set(_static_exports(str(module), path))
        expected = set(
            _string_sequence(settings.get("exports"), label=f"{module}.exports")
        )
        if actual != expected:
            violations.append(
                f"{module} facade export drift: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
    if require_final and pending:
        violations.append(f"facade not yet frozen: {', '.join(sorted(pending))}")

    layers = _mapping(manifest.get("layers"), label="layers")
    rules = layers.get("dependency_rules")
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise ValueError("layers.dependency_rules must be a sequence")
    imports_by_module = {
        module: _module_imports(path, module) for module, path in modules.items()
    }
    for raw_rule in rules:
        rule = _mapping(raw_rule, label="dependency_rule")
        name = str(rule.get("name"))
        source_prefix = str(rule.get("source_prefix"))
        forbidden = _string_sequence(
            rule.get("forbidden_import_prefixes"), label=f"{name}.forbidden"
        )
        for module, imports in imports_by_module.items():
            if not _prefix_matches(module, source_prefix):
                continue
            for imported in sorted(imports):
                if any(_prefix_matches(imported, prefix) for prefix in forbidden):
                    violations.append(f"dependency rule {name}: {module} -> {imported}")

    retirement = _mapping(manifest.get("legacy_retirement"), label="legacy_retirement")
    forbidden_imports = _string_sequence(
        retirement.get("forbidden_import_prefixes"),
        label="legacy_retirement.forbidden_import_prefixes",
    )
    for module, imports in imports_by_module.items():
        for imported in sorted(imports):
            if any(_prefix_matches(imported, prefix) for prefix in forbidden_imports):
                violations.append(f"retired shim import: {module} -> {imported}")

    regexes = tuple(
        re.compile(pattern)
        for pattern in _string_sequence(
            retirement.get("forbidden_path_or_text_regexes"),
            label="legacy_retirement.forbidden_path_or_text_regexes",
        )
    )
    roots = retirement.get("scan_roots")
    if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)):
        raise ValueError("legacy_retirement.scan_roots must be a sequence")
    for raw_root in roots:
        settings = _mapping(raw_root, label="legacy scan root")
        root = REPO_ROOT / str(settings.get("root"))
        suffixes = set(
            _string_sequence(settings.get("suffixes"), label=f"{root}.suffixes")
        )
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.suffix not in suffixes:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            for pattern in regexes:
                if pattern.search(relative) or pattern.search(text):
                    violations.append(
                        f"retired product name {pattern.pattern!r}: {relative}"
                    )
    return tuple(sorted(set(violations)))


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value  # type: ignore[return-value]


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a string sequence")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} may only contain non-empty strings")
    return tuple(value)


def _physics_selection_sequence(
    value: object, *, label: str
) -> tuple[tuple[str, str], ...]:
    """解析架构清单中的公开 engine/execution 组合，不接受内部 runtime kind。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a physics selection sequence")
    result: list[tuple[str, str]] = []
    for index, raw_selection in enumerate(value):
        selection = _mapping(raw_selection, label=f"{label}[{index}]")
        if set(selection) != {"engine", "execution"}:
            raise ValueError(f"{label}[{index}] must contain exactly engine/execution")
        engine = selection["engine"]
        execution = selection["execution"]
        if not isinstance(engine, str) or not engine:
            raise ValueError(f"{label}[{index}].engine must be a non-empty string")
        if not isinstance(execution, str) or not execution:
            raise ValueError(f"{label}[{index}].execution must be a non-empty string")
        result.append((engine, execution))
    return tuple(result)


def _desired_files(manifest: Mapping[str, object]) -> dict[Path, str]:
    return {
        MODULE_MAP_PATHS["en"]: _module_map(manifest, "en"),
        MODULE_MAP_PATHS["zh-CN"]: _module_map(manifest, "zh-CN"),
        ALLOWLIST_PATH: _hardcoded_allowlist(),
        MANIFEST_PATH: _manifest_with_generated_inventory(manifest),
    }


def _write(desired: Mapping[Path, str]) -> tuple[Path, ...]:
    changed: list[Path] = []
    for path, content in desired.items():
        if path.read_text(encoding="utf-8") == content:
            continue
        path.write_text(content, encoding="utf-8")
        changed.append(path)
    return tuple(changed)


def _drift(desired: Mapping[Path, str]) -> tuple[Path, ...]:
    return tuple(
        path
        for path, content in desired.items()
        if path.read_text(encoding="utf-8") != content
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="only check the generated output and the architecture rules",
    )
    mode.add_argument("--write", action="store_true", help="rewrite all mechanical inventory")
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="require all facades to be frozen; used as a release gate after legacy paths are removed",
    )
    arguments = parser.parse_args(argv)

    try:
        manifest = _manifest()
        desired = _desired_files(manifest)
        if arguments.write:
            # module map 本身属于 maintained_docs inventory，必须先落盘再计算 manifest
            # 的 content hash；否则一次 write 会把写入前的文档摘要冻结进去，必须运行
            # 第二遍才能收敛。
            changed = list(
                _write(
                    {
                        path: content
                        for path, content in desired.items()
                        if path != MANIFEST_PATH
                    }
                )
            )
            final_manifest = _manifest_with_generated_inventory(_manifest())
            changed.extend(_write({MANIFEST_PATH: final_manifest}))
            for path in changed:
                print(f"updated {path.relative_to(REPO_ROOT)}")
            # write 不能成为绕过架构规则的入口。机械清单落盘后重新读取最终 manifest，
            # 再执行与 check 相同的 facade、依赖方向、退役路径和精确产品合同检查。
            written_manifest = _manifest()
            violations = _architecture_violations(
                written_manifest, require_final=arguments.require_final
            )
            for violation in violations:
                print(f"architecture violation: {violation}")
            return 1 if violations else 0

        drift = _drift(desired)
        violations = _architecture_violations(
            manifest, require_final=arguments.require_final
        )
        for path in drift:
            print(f"generated inventory drift: {path.relative_to(REPO_ROOT)}")
        for violation in violations:
            print(f"architecture violation: {violation}")
        return 1 if drift or violations else 0
    except (
        KeyError,
        OSError,
        SyntaxError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        print(f"architecture inventory error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
