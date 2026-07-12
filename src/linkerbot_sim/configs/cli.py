"""运行时配置的纯 Python 校验命令行入口。

该入口在不导入 Isaac Sim、torch 或 cuRobo 运行库的情况下加载运行 profile、解析最终值并
遍历完整配置依赖图。普通输出只包含 profile 名和指纹，可安全写入启动日志；包含配置路径等
敏感或冗长信息的完整值，仅在用户显式传入 ``--dump-effective-config`` 时输出。

本模块只负责参数解析、调用校验边界和格式化 JSON，不启动仿真，也不修改配置文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.configs.runtime import (
    ResolvedRuntimeConfig,
    RuntimeProfileConfig,
    load_runtime_profile,
    resolve_runtime_config,
)
from linkerbot_sim.configs.validator import validate_profile_graph


def build_parser() -> argparse.ArgumentParser:
    """构造独立配置校验命令的参数解析器。

    返回:
        只包含运行 profile 选择和有效配置转储开关的 ``ArgumentParser``。
    副作用:
        无；不会读取参数、配置文件或进程环境。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Validate and resolve a runtime YAML profile without starting Isaac Sim."
        )
    )
    parser.add_argument(
        "--runtime-profile",
        default="default",
        help="runtime profile name under configs/runtime (default: default)",
    )
    parser.add_argument(
        "--dump-effective-config",
        action="store_true",
        help="print resolved values and the source of every field as JSON",
    )
    return parser


def resolve_runtime_profile(
    runtime_profile: str,
    *,
    expected_mode: str | None = None,
) -> tuple[RuntimeProfileConfig, ResolvedRuntimeConfig]:
    """加载并完整解析一个运行 profile，用于无仿真预检。

    参数:
        runtime_profile: ``configs/runtime`` 下的稳定 profile 名称。
        expected_mode: 可选入口模式约束；不匹配时立即失败。
    返回:
        严格原 profile 与最终解析结果组成的 tuple。
    异常:
        FileNotFoundError: profile 或任一依赖文件不存在。
        TypeError: YAML 节点类型不符合 schema。
        ValueError: 字段、模式、依赖绑定或路径所有权校验失败。
    副作用:
        读取运行、环境及其依赖 profile；不创建 Isaac 应用或输出文件。
    """

    profile = load_runtime_profile(runtime_profile)
    env_config = load_profile_yaml("env", profile.profiles.env)
    resolved = resolve_runtime_config(
        profile,
        cli_overrides={},
        env_config=env_config,
        expected_mode=expected_mode,
    )
    validate_profile_graph(
        runtime_profile=runtime_profile,
        profile=profile,
        resolved=resolved,
        env_config=env_config,
    )
    return profile, resolved


def validation_summary(
    runtime_profile: str,
    resolved: ResolvedRuntimeConfig,
) -> dict[str, object]:
    """生成不含配置路径和值的普通启动诊断摘要。

    返回新字典，包含固定事件名、运行 profile 名和最终配置指纹，不修改 ``resolved``。
    """

    return {
        "event": "config_validated",
        "runtime_profile": runtime_profile,
        "fingerprint": resolved.fingerprint,
    }


def effective_config_dump(
    runtime_profile: str,
    resolved: ResolvedRuntimeConfig,
) -> dict[str, object]:
    """生成显式请求的完整有效配置与逐字段来源转储。

    结果包含路径等原始配置值，只应在用户主动诊断时输出。返回结构是可序列化副本，调用方
    修改它不会改变 ``resolved``。
    """

    return {
        "runtime_profile": runtime_profile,
        "fingerprint": resolved.fingerprint,
        "effective": resolved.as_dict(),
        "sources": dict(resolved.sources),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """执行配置校验并返回进程风格退出码。

    参数:
        argv: 可选参数序列；``None`` 时由 argparse 读取 ``sys.argv``。
        stdout: 成功 JSON 的目标文本流，默认 ``sys.stdout``。
        stderr: 校验错误的目标文本流，默认 ``sys.stderr``。
    返回:
        校验成功返回 ``0``，可预期的文件、类型或值错误返回 ``1``。
    副作用:
        读取配置依赖，并向指定输出流写入一条结果；不会吞掉非配置类程序错误。
    """

    output = sys.stdout if stdout is None else stdout
    error_output = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)

    try:
        _profile, resolved = resolve_runtime_profile(args.runtime_profile)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(
            f"CONFIG_INVALID {type(exc).__name__}: {exc}",
            file=error_output,
            flush=True,
        )
        return 1

    payload: Mapping[str, object]
    if args.dump_effective_config:
        payload = effective_config_dump(args.runtime_profile, resolved)
    else:
        payload = validation_summary(args.runtime_profile, resolved)
    print(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        file=output,
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - 由外层薄脚本覆盖执行路径。
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "effective_config_dump",
    "main",
    "resolve_runtime_profile",
    "validation_summary",
]
