#!/usr/bin/env python3
import json
from pathlib import Path

rows = json.loads((Path(__file__).parent / "expected_results.json").read_text())
for row in rows:
    ratio = row["cce_us"] / row["vmi_us"]
    status = "REGRESSION_REPRODUCED" if ratio < 0.98 else "NO_GAP_REPRODUCED"
    print(f"case={row['case']} CCE_us={row['cce_us']:.4f} VMI_us={row['vmi_us']:.4f} CCE_over_VMI={ratio:.4f} status={status}")
