"""
Tests for app.validators.log_utils.Logger

Covers:
  - log methods write to the log file with correct level tags
  - debug is only written when level=DEBUG is passed explicitly
  - push_logs_to_github returns correct status strings
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.validators.log_utils import Logger


class TestLogger(unittest.TestCase):
    def setUp(self) -> None:
        # Each test gets its own temp dir + fresh Logger so handlers don't stack.
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name)
        # Force DEBUG level so all methods can be observed.
        self.logger = Logger(
            log_dir=self.log_dir,
            log_file="test.log",
            level=logging.DEBUG,
        )
        self.log_file = self.log_dir / "test.log"

    def tearDown(self) -> None:
        # Close file handlers to release the log file on Windows.
        for handler in self.logger._logger.handlers[:]:
            handler.close()
            self.logger._logger.removeHandler(handler)
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # Log method behaviour
    # ------------------------------------------------------------------

    def test_info_writes_info_tag(self) -> None:
        self.logger.info("hello info")
        content = self.log_file.read_text()
        self.assertIn("[INFO]", content)
        self.assertIn("hello info", content)

    def test_warning_writes_warning_tag(self) -> None:
        self.logger.warning("hello warning")
        content = self.log_file.read_text()
        self.assertIn("[WARNING]", content)
        self.assertIn("hello warning", content)

    def test_error_writes_error_tag(self) -> None:
        self.logger.error("hello error")
        content = self.log_file.read_text()
        self.assertIn("[ERROR]", content)
        self.assertIn("hello error", content)

    def test_debug_writes_debug_tag(self) -> None:
        self.logger.debug("hello debug")
        content = self.log_file.read_text()
        self.assertIn("[DEBUG]", content)
        self.assertIn("hello debug", content)

    def test_log_methods_return_none(self) -> None:
        """Logger methods are fire-and-forget; they must return None."""
        self.assertIsNone(self.logger.info("x"))
        self.assertIsNone(self.logger.warning("x"))
        self.assertIsNone(self.logger.error("x"))
        self.assertIsNone(self.logger.debug("x"))

    def test_log_file_created_on_first_write(self) -> None:
        new_logger = Logger(
            log_dir=self.log_dir,
            log_file="created.log",
            level=logging.INFO,
        )
        new_logger.info("trigger creation")
        self.assertTrue((self.log_dir / "created.log").exists())
        for h in new_logger._logger.handlers[:]:
            h.close(); new_logger._logger.removeHandler(h)

    # ------------------------------------------------------------------
    # push_logs_to_github
    # ------------------------------------------------------------------

    def test_push_no_changes_returns_info(self) -> None:
        """Returns '[INFO] No log changes to commit.' when the log file is not
        reported as modified by git status."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""   # log file NOT in status output
            )
            result = self.logger.push_logs_to_github(repo_root=self.log_dir)

        self.assertIn("[INFO]", result)
        self.assertIn("No log changes", result)

    def test_push_successful_returns_github_string(self) -> None:
        """Returns '[INFO] Log pushed to GitHub log-archive branch.' on success."""
        log_name = "test.log"
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                # git status --porcelain  (log file IS listed)
                subprocess.CompletedProcess(args=[], returncode=0, stdout=f"M {log_name}\n"),
                # git checkout log-archive
                subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
                # git add ...
                subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
                # git commit -m ...  (returncode 0 = success)
                subprocess.CompletedProcess(args=[], returncode=0, stdout="[log-archive abc1234]"),
                # git push origin log-archive
                subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
                # git checkout -
                subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
            ]
            result = self.logger.push_logs_to_github(repo_root=self.log_dir)

        self.assertIn("[INFO]", result)
        self.assertIn("GitHub", result)

    def test_push_commit_fails_returns_nothing_to_commit(self) -> None:
        """When git commit exits non-zero, returns '[INFO] Nothing to commit: ...'"""
        log_name = "test.log"
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout=f"M {log_name}\n"),
                subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
                # commit fails
                subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr="nothing to commit"
                ),
            ]
            result = self.logger.push_logs_to_github(repo_root=self.log_dir)

        self.assertIn("[INFO]", result)
        self.assertIn("Nothing to commit", result)

    def test_push_git_exception_returns_error(self) -> None:
        """When subprocess raises an exception, returns '[ERROR] Git push failed: ...'"""
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            result = self.logger.push_logs_to_github(repo_root=self.log_dir)

        self.assertIn("[ERROR]", result)
        self.assertIn("Git push failed", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
