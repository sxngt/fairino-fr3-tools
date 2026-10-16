"""Cooperative motion lock: two scripts never move the arm at once. / 협조적 모션 잠금.

Every entry point that commands motion takes the same advisory file lock (``flock``) for the
controller it talks to. A second script fails fast with the holder's PID and command line instead
of fighting the first one for the servo stream. The lock is released by the kernel when the
process exits, even on a crash, so a stale lock file is harmless.

It is cooperative: it protects scripts that use it, not the teach pendant or foreign tools.

모션을 내는 모든 진입점이 제어기별 같은 파일 잠금(flock)을 잡는다. 두 번째 스크립트는 보유자 PID 와
명령행을 보여 주며 바로 실패한다. 프로세스가 끝나면(비정상 종료 포함) 커널이 잠금을 푼다. 협조적
잠금이라 티치 펜던트나 다른 도구는 막지 못한다.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import tempfile
import time


class MotionLockBusyError(RuntimeError):
    """Another process holds the motion lock. / 다른 프로세스가 모션 잠금을 보유 중."""


def default_lock_dir() -> str:
    """``$FR3_LOCK_DIR``, else ``$XDG_RUNTIME_DIR``, else the temp dir. / 잠금 디렉터리."""
    return os.environ.get("FR3_LOCK_DIR") or os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()


class MotionLock:
    """``with MotionLock("192.168.58.2"): ...`` / 모션 잠금 컨텍스트.

    ``name`` is usually the controller IP; ``timeout`` > 0 waits for the holder to finish.
    """

    def __init__(self, name: str = "fr3", lock_dir: str | None = None, timeout: float = 0.0):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        self.path = os.path.join(lock_dir or default_lock_dir(), f"fr3tools-motion-{safe}.lock")
        self.timeout = timeout
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def holder(self) -> dict:
        """Who wrote the lock file last (may be stale if nobody holds it). / 마지막 보유자 정보."""
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.loads(f.read() or "{}")
        except (OSError, ValueError):
            return {}

    def acquire(self) -> MotionLock:
        import fcntl

        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - t0 >= self.timeout:
                    os.close(fd)
                    h = self.holder()
                    raise MotionLockBusyError(
                        f"another process is moving the arm (pid {h.get('pid', '?')}: "
                        f"{h.get('cmd', '?')}); lock {self.path}"
                    ) from None
                time.sleep(0.05)
        info = {"pid": os.getpid(), "cmd": " ".join(sys.argv)[:200], "since": time.strftime("%H:%M:%S")}
        os.ftruncate(fd, 0)
        os.pwrite(fd, json.dumps(info).encode(), 0)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        import fcntl

        try:
            os.ftruncate(self._fd, 0)
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> MotionLock:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def exclusive_motion(name: str = "fr3", **kw):
    """Decorator: run the function under :class:`MotionLock`. / 함수를 모션 잠금 아래에서 실행."""

    def wrap(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            with MotionLock(name, **kw):
                return fn(*args, **kwargs)

        return run

    return wrap
