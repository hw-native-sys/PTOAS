# vmula bf16 output is bit-compared as uint16 bf16 bit patterns: the DUT
# rounds once to bf16 (RNE), so the simulator/NPU result must match the
# golden bits exactly.  A mismatch here means the vldas+vldus stateful load
# (#1374) read wrong lanes from the misaligned address.

import sys

import numpy as np


def main() -> None:
    golden = np.fromfile("golden_v4.bin", dtype=np.uint16)
    output = np.fromfile("v4.bin", dtype=np.uint16)
    if golden.shape != output.shape or not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else -1
        print(f"[ERROR] compare failed idx={idx} golden={int(golden[idx]) if idx >= 0 else 'n/a'} output={int(output[idx]) if idx >= 0 else 'n/a'}")
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
