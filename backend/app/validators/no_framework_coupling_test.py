"""
Guardrail: app.validators must stay a pure pandas-in/pandas-out domain layer.

This is what previously let bk's tests run instantly with no database —
merging bk into backend/app/ (see the docstring history in this repo) made
it easy for that boundary to erode, since a SQLAlchemy model or FastAPI
Depends() is now just one import away. This test fails loudly the day
someone crosses that line, rather than relying on convention/memory.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

_FORBIDDEN = ["sqlalchemy", "fastapi", "app.models", "app.database"]

# Only the actual source modules — not this file, not the *_test.py files
# (a test file importing app.database to set up fixtures would be normal
# and is not what this guard is protecting against).
_SOURCE_FILES = [
    "schemas.py",
    "domain.py",
    "helpers.py",
    "anomaly.py",
    "runner.py",
    "log_utils.py",
    "log_parser.py",
]

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w\.]+)", re.MULTILINE)


class TestNoFrameworkCoupling(unittest.TestCase):

    def test_validator_modules_have_no_framework_imports(self):
        here = Path(__file__).parent
        violations: dict[str, list[str]] = {}

        for filename in _SOURCE_FILES:
            path = here / filename
            if not path.is_file():
                continue
            source = path.read_text(encoding="utf-8")
            for match in _IMPORT_RE.finditer(source):
                module = match.group(1)
                for forbidden in _FORBIDDEN:
                    if module == forbidden or module.startswith(forbidden + "."):
                        violations.setdefault(filename, []).append(module)

        self.assertEqual(
            violations, {},
            "app.validators must stay framework-independent (pure pandas in, "
            "pandas out) so its tests can run instantly without a database. "
            f"Found forbidden imports: {violations}"
        )


if __name__ == "__main__":
    unittest.main()
