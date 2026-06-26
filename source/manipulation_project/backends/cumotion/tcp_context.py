"""带可选临时 TCP URDF 的 cuMotion context 装配入口。

cuMotion 的 FK/IK/planner 只能使用 robot description 中已经存在的 frame。也就是说，
如果任务层计算出了 pinch center 这类临时 TCP，必须在 ``CuMotionContext`` 创建之前把
该 TCP 作为 fixed link 写进待加载的 URDF。

本模块把这个后端约束封装成 context manager：任务层只需要传入 ``TcpFrame``，即可获得已经
识别该 TCP 的 ``CuMotionContext``，不用直接管理临时目录、临时 URDF 和
``CuMotionConfig`` 替换。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Iterator
import xml.etree.ElementTree as ET

import numpy as np

from manipulation_project.backends.cumotion.context import (
    CuMotionConfig,
    CuMotionContext,
)
from manipulation_project.backends.cumotion.tcp_urdf_builder import write_tcp_urdf
from manipulation_project.tcp.tcp_frame import TcpFrame


@contextmanager
def make_cumotion_context(
    config: CuMotionConfig,
    *,
    tcp: TcpFrame | None = None,
    output_dir: str | Path | None = None,
) -> Iterator[CuMotionContext]:
    """按需创建普通或带自定义 TCP 的 ``CuMotionContext``。

    参数:
        config: 基础 cuMotion 配置。``xrdf_path`` 不会被修改；``urdf_path`` 只有在需要追加
            TCP link 时才会替换成临时 URDF。
        tcp: 可选 TCP 描述。为空时保持原始 ``CuMotionContext(config)`` 语义。
        output_dir: 可选输出目录。为空时本函数创建并持有临时目录；非空时调用方负责该目录
            后续清理，便于调试或测试检查生成的 URDF。

    关键语义:
        * 不为了判断 frame 是否存在而先创建一个普通 ``CuMotionContext``，避免重复加载
          cuMotion robot description。
        * 已存在的非 flange frame 默认报错，因为传入的 ``TcpFrame.xyz/rpy`` 无法覆盖基础
          URDF 中已有 link 的真实位姿，静默忽略会非常危险。
        * 临时目录必须活到 ``CuMotionContext`` 使用结束；这正是本函数使用 context manager
          的原因。
    """

    if tcp is None:
        # 没有自定义 TCP 时完全走原始路径，确保现有调用方不因新入口改变行为。
        yield CuMotionContext(config)
        return

    base_urdf_path = Path(config.urdf_path)
    # 先解析基础 URDF 的 link 列表做前置判断。这里不用 cuMotion 加载模型，因为最终如果要
    # 追加 TCP 仍然需要再创建一次带临时 URDF 的 context；提前加载会多一次昂贵初始化。
    link_names = _urdf_link_names(base_urdf_path)
    if tcp.parent_frame not in link_names:
        # parent frame 不存在时，即使写出 URDF 也会得到断开的或非法的运动树，直接给出清晰错误。
        raise ValueError(
            f"Parent frame {tcp.parent_frame!r} not found in {base_urdf_path}"
        )

    if tcp.frame_name in link_names:
        if _is_zero_offset_flange_tcp(config, tcp):
            # “TCP 就是法兰本身”是唯一允许复用已有 link 的特殊情况。这里显式清空
            # custom_tcp_frame，避免基础 config 中残留的工具 frame 继续成为 IK/planner 默认值。
            context = CuMotionContext(replace(config, custom_tcp_frame=None))
            _validate_context_tcp(context, tcp, expect_custom_tcp=False)
            yield context
            return
        # 已有工具 link 应通过基础 URDF/XRDF + custom_tcp_frame 或单次 tcp_frame_name 表达。
        # 如果这里接受 TcpFrame，调用方很容易误以为 xyz/rpy 会覆盖已有 link 位姿。
        raise ValueError(
            f"TCP frame {tcp.frame_name!r} already exists in {base_urdf_path}; "
            "use CuMotionConfig.custom_tcp_frame or an explicit tcp_frame_name for "
            "existing tool frames"
        )

    if output_dir is None:
        # 临时目录由 context manager 持有，保证 URDF 文件在 context 及其派生 IK/planner 使用期内存在。
        with tempfile.TemporaryDirectory(prefix="cumotion_tcp_") as temp_dir:
            context = _make_context_with_written_tcp(config, tcp, Path(temp_dir))
            yield context
    else:
        # 显式 output_dir 主要用于测试、诊断或离线查看生成 URDF。该目录不是本函数创建的
        # TemporaryDirectory，因此退出 context 后不清理。
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        context = _make_context_with_written_tcp(config, tcp, output_path)
        yield context


def _make_context_with_written_tcp(
    config: CuMotionConfig, tcp: TcpFrame, output_dir: Path
) -> CuMotionContext:
    """写出附加 TCP link 的 URDF，并用它创建 ``CuMotionContext``。"""

    base_urdf_path = Path(config.urdf_path)
    tcp_urdf_path = output_dir / f"{base_urdf_path.stem}_{tcp.frame_name}.urdf"
    write_tcp_urdf(base_urdf_path, tcp_urdf_path, tcp)
    # ``custom_tcp_frame`` 写进 config 后，context.make_inverse_kinematics() 和
    # context.make_motion_planner() 在调用方未显式传 tcp_frame_name 时会默认使用该 TCP。
    context = CuMotionContext(
        replace(
            config,
            urdf_path=tcp_urdf_path,
            custom_tcp_frame=tcp.frame_name,
        )
    )
    _validate_context_tcp(context, tcp, expect_custom_tcp=True)
    return context


def _validate_context_tcp(
    context: CuMotionContext, tcp: TcpFrame, *, expect_custom_tcp: bool
) -> None:
    """校验 cuMotion 最终暴露的 frame 集合与装配意图一致。

    URDF 中出现 link 不代表 cuMotion kinematics 一定会把它暴露成可查询 frame，因此创建
    context 后仍要以后端实际 ``frame_names()`` 为准做一次后验检查。
    """

    if not context.has_frame(tcp.frame_name):
        raise ValueError(
            f"cuMotion frame {tcp.frame_name!r} was not found after loading the URDF"
        )
    if expect_custom_tcp and context.config.custom_tcp_frame != tcp.frame_name:
        raise ValueError(
            "cuMotion context custom_tcp_frame mismatch: "
            f"expected {tcp.frame_name!r}, got {context.config.custom_tcp_frame!r}"
        )


def _is_zero_offset_flange_tcp(config: CuMotionConfig, tcp: TcpFrame) -> bool:
    """判断传入 TCP 是否只是“法兰 frame 本身”。

    使用 ``np.allclose`` 而不是严格相等，是为了兼容配置解析或几何计算产生的浮点微小误差。
    """

    return (
        tcp.frame_name == config.flange_frame
        and tcp.parent_frame == config.flange_frame
        and np.allclose(tcp.xyz, 0.0)
        and np.allclose(tcp.rpy, 0.0)
    )


def _urdf_link_names(urdf_path: str | Path) -> set[str]:
    """读取 URDF 中所有 link 名称，用于 context 创建前的快速装配判断。"""

    root = ET.parse(Path(urdf_path)).getroot()
    return {str(link.get("name")) for link in root.findall("link") if link.get("name")}
