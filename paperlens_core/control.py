from __future__ import annotations

import json
import sys
import threading


class ControlState:
    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._paused = threading.Event()
        self._resume_signal = threading.Event()
        self._resume_signal.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def cancel(self) -> None:
        self._cancelled.set()
        self._resume_signal.set()

    def pause(self) -> None:
        self._paused.set()
        self._resume_signal.clear()

    def resume(self) -> None:
        self._paused.clear()
        self._resume_signal.set()

    def wait_if_paused(self) -> None:
        while self.paused and not self.cancelled:
            self._resume_signal.wait()

    def require_not_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("Run cancelled")


def start_control_listener(control: ControlState) -> threading.Thread:
    def listen() -> None:
        for raw in sys.stdin:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            command = message.get("type") or message.get("command")
            if command == "cancel":
                control.cancel()
            elif command == "pause":
                control.pause()
            elif command == "resume":
                control.resume()

    thread = threading.Thread(target=listen, name="control-listener", daemon=True)
    thread.start()
    return thread
