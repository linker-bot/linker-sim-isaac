"""Kaleidoscope 显式 human viewport 的纯启动配置。

该配置只决定 Kit/RTX 启动、窗口尺寸、可见环境和显示节奏，不属于训练任务、物理状态
或 episode snapshot 的语义。它是独立 launch root，不嵌入 ``KaleidoscopeConfig``。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common import (
    ConfigurationError,
    as_bool,
    as_int,
    as_string,
    require_keys,
    strict_mapping,
)
from ..scenes import SceneVisualSettings


@dataclass(frozen=True)
class KaleidoscopeViewportSettings:
    """单个选中环境的 human viewport 启动参数。

    ``selected_env`` 只选择 renderer-facing world；最终环境数仍由训练配置和调用时
    ``num_envs`` override 决定，并在 scene assembly 前完成交叉校验。
    """

    selected_env: int
    render_every_n_steps: int
    width: int
    height: int
    window_width: int
    window_height: int
    renderer: str
    anti_aliasing: int
    samples_per_pixel_per_frame: int
    denoiser: bool
    visuals: SceneVisualSettings

    def __post_init__(self) -> None:
        """也约束直接构造/``dataclasses.replace``，避免绕过 YAML parser。"""

        if type(self.selected_env) is not int or self.selected_env < 0:
            raise ConfigurationError(
                "viewport.selected_env must be a non-negative integer"
            )
        if type(self.render_every_n_steps) is not int or self.render_every_n_steps < 1:
            raise ConfigurationError(
                "viewport.render_every_n_steps must be a positive integer"
            )
        for name in ("width", "height", "window_width", "window_height"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ConfigurationError(f"viewport.{name} must be a positive integer")
        if (
            not isinstance(self.renderer, str)
            or not self.renderer
            or self.renderer.strip() != self.renderer
        ):
            raise ConfigurationError(
                "viewport.renderer must be a non-empty string with no leading or trailing whitespace"
            )
        if type(self.anti_aliasing) is not int or self.anti_aliasing < 0:
            raise ConfigurationError(
                "viewport.anti_aliasing must be a non-negative integer"
            )
        if (
            type(self.samples_per_pixel_per_frame) is not int
            or self.samples_per_pixel_per_frame < 1
        ):
            raise ConfigurationError(
                "viewport.samples_per_pixel_per_frame must be a positive integer"
            )
        if type(self.denoiser) is not bool:
            raise ConfigurationError("viewport.denoiser must be a boolean")
        if not isinstance(self.visuals, SceneVisualSettings):
            raise ConfigurationError("viewport.visuals must be a SceneVisualSettings")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        label: str = "viewport",
    ) -> "KaleidoscopeViewportSettings":
        """从一个 exact-key mapping 构造 viewport 设置。"""

        mapping = strict_mapping(value, label=label)
        required = {
            "selected_env",
            "render_every_n_steps",
            "width",
            "height",
            "window_width",
            "window_height",
            "renderer",
            "anti_aliasing",
            "samples_per_pixel_per_frame",
            "denoiser",
            "visuals",
        }
        require_keys(mapping, required=required, label=label)
        visuals_mapping = strict_mapping(mapping["visuals"], label=f"{label}.visuals")
        try:
            visuals = SceneVisualSettings.from_scene_mapping(
                {"visuals": visuals_mapping}
            )
        except ValueError as exc:
            raise ConfigurationError(
                f"{label} contains invalid visual settings: {exc}"
            ) from exc
        return cls(
            selected_env=as_int(
                mapping["selected_env"],
                label=f"{label}.selected_env",
                minimum=0,
            ),
            render_every_n_steps=as_int(
                mapping["render_every_n_steps"],
                label=f"{label}.render_every_n_steps",
                minimum=1,
            ),
            width=as_int(mapping["width"], label=f"{label}.width", minimum=1),
            height=as_int(mapping["height"], label=f"{label}.height", minimum=1),
            window_width=as_int(
                mapping["window_width"],
                label=f"{label}.window_width",
                minimum=1,
            ),
            window_height=as_int(
                mapping["window_height"],
                label=f"{label}.window_height",
                minimum=1,
            ),
            renderer=as_string(mapping["renderer"], label=f"{label}.renderer"),
            anti_aliasing=as_int(
                mapping["anti_aliasing"],
                label=f"{label}.anti_aliasing",
                minimum=0,
            ),
            samples_per_pixel_per_frame=as_int(
                mapping["samples_per_pixel_per_frame"],
                label=f"{label}.samples_per_pixel_per_frame",
                minimum=1,
            ),
            denoiser=as_bool(mapping["denoiser"], label=f"{label}.denoiser"),
            visuals=visuals,
        )


__all__ = ["KaleidoscopeViewportSettings"]
