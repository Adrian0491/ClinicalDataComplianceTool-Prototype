from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from bk.logging.log_indexing import Logger
from bk.schemas import concat_findings
from bk.validator.domain import (
    validate_ae, validate_cm, validate_dm, validate_ds, validate_eg,
    validate_ex, validate_lb, validate_pr, validate_qs, validate_rs,
    validate_dm_link, validate_irt_consistency, validate_vs, validate_vs_ae, validate_vs_cm,
)
from bk.validator.anomaly import (
    apply_rules, build_frame_from_domains, detect_anomalies, load_generic, to_findings,
)

_log = Logger()


def run_sdtm_validation(
    data_dir:   str | Path = "mock_data",
    output_dir: str | Path = "output",
) -> pd.DataFrame:
    """Run full SDTM domain + cross-domain validation pipeline."""
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load(name: str) -> pd.DataFrame:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            _log.warning(f"{path} not found — skipping {name.upper()} domain.")
            return pd.DataFrame()
        df = pd.read_csv(path)
        _log.info(f"Loaded {name.upper()}: {len(df)} rows, {len(df.columns)} cols.")
        return df

    dm = _load("dm")
    vs = _load("vs")
    ae = _load("ae")
    cm = _load("cm")
    ex = _load("ex")
    lb = _load("lb")
    ds = _load("ds")
    eg = _load("eg")
    qs = _load("qs")
    rs = _load("rs")
    pr = _load("pr")

    parts: list[pd.DataFrame] = []

    if len(dm): _log.info("Running DM validation..."),  parts.append(validate_dm(dm))
    if len(vs): _log.info("Running VS validation..."),  parts.append(validate_vs(vs))
    if len(ae): _log.info("Running AE validation..."),  parts.append(validate_ae(ae))
    if len(cm): _log.info("Running CM validation..."),  parts.append(validate_cm(cm))
    if len(ex): _log.info("Running EX validation..."),  parts.append(validate_ex(ex))
    if len(lb): _log.info("Running LB validation..."),  parts.append(validate_lb(lb))
    if len(ds): _log.info("Running DS validation..."),  parts.append(validate_ds(ds))
    if len(eg): _log.info("Running EG validation..."),  parts.append(validate_eg(eg))
    if len(qs): _log.info("Running QS validation..."),  parts.append(validate_qs(qs))
    if len(rs): _log.info("Running RS validation..."),  parts.append(validate_rs(rs))
    if len(pr): _log.info("Running PR validation..."),  parts.append(validate_pr(pr))

    if len(dm):
        for name, df in [("VS", vs), ("AE", ae), ("CM", cm), ("EX", ex), ("LB", lb), ("DS", ds), ("EG", eg), ("QS", qs), ("RS", rs), ("PR", pr)]:
            if len(df):
                _log.info(f"Running DM ↔ {name} link check...")
                parts.append(validate_dm_link(dm, df, name))

    if len(dm) and len(ex):
        _log.info("Running IRT (EX/DM/DS) consistency checks...")
        parts.append(validate_irt_consistency(dm, ex, ds))

    if len(vs) and len(ae):
        _log.info("Running VS ↔ AE cross-domain checks...")
        parts.append(validate_vs_ae(vs, ae))

    if len(vs) and len(cm):
        _log.info("Running VS ↔ CM cross-domain checks...")
        parts.append(validate_vs_cm(vs, cm))

    if len(dm):
        _log.info("Running anomaly detection across DM/VS/EX/LB/EG/QS/PR...")
        generic = build_frame_from_domains(dm, vs, ex, lb=lb, eg=eg, qs=qs, pr=pr)
        if len(generic):
            generic = apply_rules(generic)
            generic = detect_anomalies(generic)
            anomaly_findings = to_findings(generic)
            if len(anomaly_findings):
                parts.append(anomaly_findings)

    findings = concat_findings(parts)
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = output_dir / f"findings_{ts}.csv"
    findings.to_csv(path, index=False)
    _log.info(f"SDTM validation complete — {len(findings)} findings → {path}")
    _print_summary(findings)
    return findings


def run_generic_validation(
    csv_path:   str | Path = "mock_data/mock_data.csv",
    output_dir: str | Path = "output",
) -> pd.DataFrame:
    """Run rule-based + anomaly detection on the generic clinical CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info(f"Loading generic data from {csv_path}...")
    df = load_generic(str(csv_path))
    df = apply_rules(df)
    df = detect_anomalies(df)
    findings = to_findings(df)

    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = output_dir / f"generic_findings_{ts}.csv"
    findings.to_csv(path, index=False)
    _log.info(f"Generic validation complete — {len(findings)} findings → {path}")
    _print_summary(findings)
    return findings


def _print_summary(findings: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  Clinical Data Compliance Tool — Validation Summary")
    print("=" * 60)
    print(f"  Total findings : {len(findings)}")

    if len(findings) == 0:
        print("  ✅  No issues found.")
        print("=" * 60)
        return

    tags = {"CRIT": "🔴", "HIGH": "🟠", "MED": "🟡", "LOW": "🔵"}
    for sev in ["CRIT", "HIGH", "MED", "LOW"]:
        n = len(findings[findings["severity"] == sev])
        if n:
            print(f"  {tags[sev]} {sev:6s}: {n}")

    print("-" * 60)
    by_domain = findings.groupby("domain").size().sort_values(ascending=False)
    for domain, n in by_domain.items():
        print(f"  Domain {domain:8s}: {n} finding(s)")
    print("=" * 60 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="bk — Clinical Data Compliance Tool")
    p.add_argument("--data-dir",     default="mock_data",              help="Directory with domain CSVs")
    p.add_argument("--output",       default="output",                 help="Output directory")
    p.add_argument("--generic",      default="mock_data/mock_data.csv", help="Generic clinical CSV")
    p.add_argument("--skip-sdtm",    action="store_true")
    p.add_argument("--skip-generic", action="store_true")
    args = p.parse_args()

    if not args.skip_sdtm:
        run_sdtm_validation(data_dir=args.data_dir, output_dir=args.output)
    if not args.skip_generic and Path(args.generic).exists():
        run_generic_validation(csv_path=args.generic, output_dir=args.output)


if __name__ == "__main__":
    main()
