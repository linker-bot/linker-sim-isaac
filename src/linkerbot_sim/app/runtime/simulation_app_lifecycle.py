"""``SimulationApp`` 生命周期收口工具。

Isaac 在关闭 ``SimulationApp`` 时有时会通过 ``SystemExit`` 表示正常退出。脚本和测试不应把
这种正常关闭误判为失败，因此统一通过本模块封装 close 行为。
"""

from __future__ import annotations


def close_simulation_app(app) -> None:
    """关闭 Isaac ``SimulationApp``，但不把正常 shutdown 转成进程失败。"""

    # Isaac 5.1 的 native shutdown 不会把控制流交还给 app.close() 的调用者，
    # 因此 file-backed importer 资源必须在进入 native shutdown 前释放。
    from linkerbot_sim.assets.robot_import import release_imported_asset_files

    release_imported_asset_files()
    try:
        app.close()
    except SystemExit as exc:
        # Isaac 正常关闭可能抛 SystemExit(0/None)；非零退出码仍然保留为真实错误。
        if exc.code not in (None, 0):
            raise
