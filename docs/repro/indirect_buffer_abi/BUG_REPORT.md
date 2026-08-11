# Feature request

PTODSL cannot express a typed GM pointer table. The checked negative fixture
fails at the nested pointer type, forcing fixed-argument staging. The live
corrected one-launch device-0 run measures direct CCE at 36.992 us and fixed-argument VMI at 31.588 us (ratio 1.1711), with both paths host-golden checked. The previous 159 us value included an invalid host-side launch loop.
Add the bounds-checkable pointer-table load and launcher support specified in
`README.md`.
