from __future__ import annotations

import json
from io import StringIO
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

import linkerbot_sim.mirror.cli as mirror_cli
from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.mirror.cli import build_parser
from linkerbot_sim.mirror.interface.protocol import (
    MIRROR_PROTOCOL,
    MIRROR_PROTOCOL_V2,
    MirrorResponse,
    decode_request,
    encode_response,
)
from linkerbot_sim.mirror.interface.transport import (
    MirrorTransportHub,
    StdinJsonlTransport,
    TcpJsonlTransport,
    TransportCloseReport,
    WebSocketTransport,
    make_json_handler,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tcp_jsonl_transport_keeps_request_response_ownership() -> None:
    def handler(payload: str) -> str:
        request = decode_request(payload)
        return encode_response(
            MirrorResponse.success(request.request_id, {"operation": request.operation})
        )

    server = TcpJsonlTransport(handler, host="127.0.0.1", port=0)
    server.start()
    try:
        with socket.create_connection(
            ("127.0.0.1", server.bound_port), timeout=2.0
        ) as client:
            stream = client.makefile("rw", encoding="utf-8", newline="\n")
            request = {
                "protocol": MIRROR_PROTOCOL,
                "request_id": "tcp-1",
                "operation": "runtime.status",
                "arguments": {},
            }
            stream.write(json.dumps(request) + "\n")
            stream.flush()
            response = json.loads(stream.readline())
    finally:
        report = server.close()

    assert response["request_id"] == "tcp-1"
    assert response["result"] == {"operation": "runtime.status"}
    assert report.stopped is True


def test_stdin_eof_policy_and_message_limit_are_explicit() -> None:
    quit_events: list[str] = []
    output = StringIO()
    transport = StdinJsonlTransport(
        lambda payload: payload.strip(),
        input_stream=StringIO("too-large\n"),
        output_stream=output,
        eof_requests_quit=lambda: quit_events.append("quit"),
        max_message_bytes=3,
    )
    transport.start()
    assert transport._thread is not None
    transport._thread.join(timeout=1.0)
    assert transport.close(timeout_s=1.0).stopped

    response = json.loads(output.getvalue().splitlines()[0])
    assert response["error"]["code"] == "message_too_large"
    assert quit_events == ["quit"]

    keep_alive = StdinJsonlTransport(
        lambda payload: payload,
        input_stream=StringIO(""),
        output_stream=StringIO(),
        eof_requests_quit=None,
    )
    keep_alive.start()
    assert keep_alive._thread is not None
    keep_alive._thread.join(timeout=1.0)
    assert keep_alive.close(timeout_s=1.0).stopped


def test_tcp_oversized_frame_returns_failure_and_connection_remains_usable() -> None:
    server = TcpJsonlTransport(
        lambda payload: json.dumps({"payload": payload.strip()}),
        host="127.0.0.1",
        port=0,
        max_message_bytes=8,
    )
    server.start()
    try:
        with socket.create_connection(
            ("127.0.0.1", server.bound_port), timeout=2.0
        ) as client:
            stream = client.makefile("rw", encoding="utf-8", newline="\n")
            stream.write("123456789\n")
            stream.write("ok\n")
            stream.flush()
            oversized = json.loads(stream.readline())
            accepted = json.loads(stream.readline())
    finally:
        report = server.close()

    assert oversized["error"]["code"] == "message_too_large"
    assert accepted == {"payload": "ok"}
    assert report.stopped


def test_websocket_transport_uses_same_text_handler() -> None:
    from websockets.sync.client import connect

    server = WebSocketTransport(
        lambda payload: json.dumps({"payload": payload}),
        host="127.0.0.1",
        port=0,
    )
    server.start()
    try:
        with connect(f"ws://127.0.0.1:{server.bound_port}") as client:
            client.send("status")
            response = json.loads(client.recv())
    finally:
        report = server.close()

    assert response == {"payload": "status"}
    assert report.stopped


def test_json_handler_turns_owner_wait_timeout_into_protocol_failure() -> None:
    request = json.dumps(
        {
            "protocol": MIRROR_PROTOCOL,
            "request_id": "slow",
            "operation": "runtime.status",
            "arguments": {},
        }
    )

    def timeout(*_args: object, **_kwargs: object) -> MirrorResponse:
        raise TimeoutError("owner busy")

    response = json.loads(make_json_handler(timeout, timeout_s=0.01)(request))
    assert response["request_id"] == "slow"
    assert response["error"]["code"] == "response_timeout"


def test_json_handler_timeout_echoes_v2_protocol() -> None:
    request = json.dumps(
        {
            "protocol": MIRROR_PROTOCOL_V2,
            "request_id": "slow-v2",
            "operation": "control.get_mode",
            "arguments": {},
        }
    )

    def timeout(*_args: object, **_kwargs: object) -> MirrorResponse:
        raise TimeoutError("owner busy")

    response = json.loads(make_json_handler(timeout, timeout_s=0.01)(request))
    assert response["protocol"] == MIRROR_PROTOCOL_V2
    assert response["error"]["code"] == "response_timeout"


def test_transport_hub_rolls_back_started_endpoints() -> None:
    events: list[str] = []

    class Endpoint:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def start(self) -> None:
            events.append(f"start:{self.name}")
            if self.fail:
                raise RuntimeError(self.name)

        def close(self) -> bool:
            events.append(f"close:{self.name}")
            return True

    hub = MirrorTransportHub((Endpoint("first"), Endpoint("second", fail=True)))
    try:
        hub.start()
    except RuntimeError:
        pass

    assert events == [
        "start:first",
        "start:second",
        "close:second",
        "close:first",
    ]


def test_transport_hub_retains_timed_out_endpoint_for_close_retry() -> None:
    class Endpoint:
        def __init__(self) -> None:
            self.close_calls = 0

        def start(self) -> None:
            pass

        def close(self) -> TransportCloseReport:
            self.close_calls += 1
            if self.close_calls == 1:
                return TransportCloseReport(False, ("endpoint-worker",))
            return TransportCloseReport(True)

    endpoint = Endpoint()
    hub = MirrorTransportHub((endpoint,))
    hub.start()

    first = hub.close()
    second = hub.close()

    assert first == TransportCloseReport(False, ("endpoint-worker",))
    assert second == TransportCloseReport(True)
    assert endpoint.close_calls == 2


def test_mirror_cli_has_only_new_profile_spelling() -> None:
    parser = build_parser()
    assert parser.parse_args([]).profile == "physx_cpu"
    args = parser.parse_args(["--profile", "newton_cpu", "--no-stdin"])
    assert args.profile == "newton_cpu"
    assert args.stdin is False
    assert parser.parse_args(["--profile", "physx_cpu_hybrid"]).profile == (
        "physx_cpu_hybrid"
    )
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }
    assert "--runtime-profile" not in option_strings
    assert "--mirror-profile" not in option_strings
    assert "--env" not in option_strings
    assert "--curobo-profile" not in option_strings


def test_cli_projects_strict_interface_settings_to_every_transport(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_mirror_config()
    calls: dict[str, object] = {}

    class Controller:
        def submit_and_wait(self, *_args: object, **_kwargs: object) -> MirrorResponse:
            raise AssertionError("run_mirror is replaced in this test")

        def request_quit(self) -> None:
            calls["quit"] = True

    runtime = SimpleNamespace(controller=Controller(), is_closed=False)
    monkeypatch.setattr(mirror_cli, "load_mirror_config", lambda _profile: config)
    monkeypatch.setattr(mirror_cli, "create_mirror_runtime", lambda _config: runtime)
    monkeypatch.setattr(
        mirror_cli,
        "make_json_handler",
        lambda _submit, *, timeout_s: calls.setdefault("response_timeout_s", timeout_s),
    )

    def endpoint(kind: str):
        def factory(_handler: object, **kwargs: object) -> object:
            calls[kind] = kwargs
            return SimpleNamespace(kind=kind)

        return factory

    monkeypatch.setattr(mirror_cli, "StdinJsonlTransport", endpoint("stdin"))
    monkeypatch.setattr(mirror_cli, "TcpJsonlTransport", endpoint("tcp"))
    monkeypatch.setattr(mirror_cli, "WebSocketTransport", endpoint("websocket"))

    def run(_runtime: object, **kwargs: object) -> object:
        calls["run"] = kwargs
        kwargs["before_session_close"](
            SimpleNamespace(completed_phases=("outputs_camera_planner",))
        )
        return SimpleNamespace(close_report=SimpleNamespace(stopped=True))

    monkeypatch.setattr(mirror_cli, "run_mirror", run)

    assert (
        mirror_cli.main(
            [
                "--tcp-jsonl",
                "127.0.0.1:43001",
                "--websocket",
                "127.0.0.1:43002",
            ]
        )
        == 0
    )

    interface = config.control.interface
    assert calls["response_timeout_s"] == interface.response_timeout_s
    assert calls["run"]["poll_timeout_s"] == interface.queue_poll_timeout_s  # type: ignore[index]
    for kind in ("stdin", "tcp", "websocket"):
        values = calls[kind]
        assert values["max_message_bytes"] == interface.max_message_bytes  # type: ignore[index]
        assert values["shutdown_timeout_s"] == interface.shutdown_timeout_s  # type: ignore[index]
    assert calls["tcp"]["max_connections"] == interface.max_connections  # type: ignore[index]
    assert calls["websocket"]["startup_timeout_s"] == interface.startup_timeout_s  # type: ignore[index]
    assert capsys.readouterr().out.splitlines() == ["MIRROR_INTERACTIVE_EXIT"]


def test_importing_mirror_facade_does_not_load_heavy_runtime_modules() -> None:
    script = """
import sys
import linkerbot_sim.mirror as mirror
for name in mirror.__all__:
    getattr(mirror, name)
for prefix in ('omni', 'isaacsim', 'torch', 'curobo', 'websockets'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
