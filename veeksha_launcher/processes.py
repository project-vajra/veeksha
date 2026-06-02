"""Subprocess lifecycle helpers for veeksha_launcher."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Optional


class ProcessTerminator:
    """Terminate a subprocess and its process group when possible."""

    def __init__(self, terminate_timeout: float = 30.0, kill_timeout: float = 10.0):
        self.terminate_timeout = terminate_timeout
        self.kill_timeout = kill_timeout

    def terminate(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return

        pgid = self._process_group_id(process)
        if pgid is not None:
            self._signal_group(pgid, signal.SIGTERM)
        else:
            process.terminate()

        try:
            process.wait(timeout=self.terminate_timeout)
            return
        except subprocess.TimeoutExpired:
            pass

        if pgid is not None:
            self._signal_group(pgid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=self.kill_timeout)

    @staticmethod
    def _process_group_id(process: subprocess.Popen) -> Optional[int]:
        pid = getattr(process, "pid", None)
        if pid is None:
            return None
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return None
        if pgid == os.getpgrp():
            return None
        return pgid

    @staticmethod
    def _signal_group(pgid: int, sig: signal.Signals) -> None:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
