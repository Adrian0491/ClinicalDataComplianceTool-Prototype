from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

_LOG_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"\s+\[(?P<level>\w+)\]"
    r"\s+(?P<message>.+)$"
)


class LogParser:
    def __init__(self, log_file: str | Path) -> None:
        self.log_file = Path(log_file)

    def parse_logs(self) -> list[dict]:
        entries = []
        with self.log_file.open(encoding="utf-8") as fh:
            for line in fh:
                entry = self._parse_line(line.rstrip())
                if entry:
                    entries.append(entry)
        return entries

    def _parse_line(self, line: str) -> dict | None:
        m = _LOG_RE.match(line)
        if not m:
            return None
        return {
            "timestamp": datetime.strptime(m.group("timestamp"), "%Y-%m-%d %H:%M:%S"),
            "level":     m.group("level"),
            "message":   m.group("message"),
        }

    def filter_by_level(self, level: str) -> list[dict]:
        return [e for e in self.parse_logs() if e["level"] == level.upper()]
