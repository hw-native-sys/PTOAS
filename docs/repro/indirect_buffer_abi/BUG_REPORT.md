# Feature request

PTODSL cannot express a typed GM pointer table. The checked negative fixture
fails at the nested pointer type, forcing fixed-argument staging. The live
device-0 run measures direct CCE at 37.686 us and the fixed-argument VMI
control at 159.406 us (ratio 0.2364), with both paths host-golden checked.
Add the bounds-checkable pointer-table load and launcher support specified in
`README.md`.
