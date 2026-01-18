import os
import polars as pl

from edc_validator.sdtm_domains import (
    validate_dm, validate_vs, validate_ae, validate_cm,
    validate_vs_ae, validate_vs_cm, validate_dm_link
)

os.makedirs("output", exist_ok=True)

dm = pl.read_csv("mock_data/dm.csv")
vs = pl.read_csv("mock_data/vs.csv")
ae = pl.read_csv("mock_data/ae.csv")
cm = pl.read_csv("mock_data/cm.csv")

f_dm = validate_dm(dm)
f_vs = validate_vs(vs)
f_ae = validate_ae(ae)
f_cm = validate_cm(cm)

# SDTM-correct anchoring
f_dm_vs = validate_dm_link(dm, vs, "VS")
f_dm_ae = validate_dm_link(dm, ae, "AE")
f_dm_cm = validate_dm_link(dm, cm, "CM")

# Optional extra cross-domain heuristics
f_vsae = validate_vs_ae(vs, ae)
f_vscm = validate_vs_cm(vs, cm)

findings = pl.concat(
    [f_dm, f_vs, f_ae, f_cm, f_dm_vs, f_dm_ae, f_dm_cm, f_vsae, f_vscm],
    how="vertical_relaxed"
)

findings.write_csv("output/findings.csv")
print(findings)
print("\nSaved: output/findings.csv")
