#!/usr/bin/env python3
"""Print the pinned A5 measurements that selected this reduced reproducer."""
import json
from pathlib import Path

rows = json.loads((Path(__file__).parent / "expected_results.json").read_text())
print("source=pinned_A5_event_medians parity_threshold=0.98")
for r in rows:
    ratio = r["asc_us"] / r["vmi_us"]
    print(f"case={r['case']} ASC_us={r['asc_us']:.4f} VMI_us={r['vmi_us']:.4f} "
          f"ratio_asc_over_vmi={ratio:.4f} status={'PASS' if ratio >= .98 else 'REGRESSION_REPRODUCED'}")
