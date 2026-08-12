#!/usr/bin/env python3
import json
from pathlib import Path
for row in json.loads((Path(__file__).parent / "expected_results.json").read_text()):
    print(f"case={row['case']} status={row['status']} negative_test={row['negative_test']}")
