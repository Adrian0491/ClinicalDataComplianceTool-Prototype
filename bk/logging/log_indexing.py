from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(
        self,
        log_dir:  str | Path = "logs",
        log_file: str = "cdct.log",
        level:    int | None = None,
    ) -> None:
        if level is None:
            level = logging.DEBUG if os.getenv("ENV") == "dev" else logging.INFO

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / log_file

        # Named logger so multiple instances don't stack handlers
        self._logger = logging.getLogger(f"cdct.{log_file}")
        self._logger.setLevel(level)

        # Some Windows consoles default to a legacy codepage (e.g. cp1252) that
        # can't encode the arrows/emoji used in log and summary messages.
        # Reconfiguring to UTF-8 lets them render properly; "replace" is a
        # last-resort fallback so an odd character never crashes logging.
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

        if not self._logger.handlers:
            fmt = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

    def info(self,    msg: str) -> None: self._logger.info(msg)
    def warning(self, msg: str) -> None: self._logger.warning(msg)
    def error(self,   msg: str) -> None: self._logger.error(msg)
    def debug(self,   msg: str) -> None: self._logger.debug(msg)

    def push_logs_to_github(
        self,
        repo_root:  str | Path | None = None,
        branch:     str = "log-archive",
        commit_msg: str | None = None,
    ) -> str:
        """Push the log file to a dedicated git branch (uses cwd=, never os.chdir)."""
        if repo_root is None:
            repo_root = self.log_path.parent.parent
        repo_root  = Path(repo_root).resolve()
        commit_msg = commit_msg or f"CDCT log update — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        def _run(cmd: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)

        try:
            status = _run(["git", "status", "--porcelain"])
            if self.log_path.name not in status.stdout:
                return "[INFO] No log changes to commit."
            _run(["git", "checkout", branch])
            _run(["git", "add", str(self.log_path)])
            result = _run(["git", "commit", "-m", commit_msg])
            if result.returncode != 0:
                return f"[INFO] Nothing to commit: {result.stderr.strip()}"
            _run(["git", "push", "origin", branch])
            _run(["git", "checkout", "-"])
            self.info("Log pushed to GitHub.")
            return "[INFO] Log pushed to GitHub log-archive branch."
        except Exception as e:
            self.error(f"Git push failed: {e}")
            return f"[ERROR] Git push failed: {e}"
