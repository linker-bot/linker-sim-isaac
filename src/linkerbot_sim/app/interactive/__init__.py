"""交互式仿真入口、协议、队列、transport 和 runtime loop。

Single Scene 交互与 Tiled Scene step-control 都放在本包下；它们共享“交互入口”这个职责，
但 TiledSceneRuntime 不复用 Single Scene motion queue / async planner 语义。
"""
