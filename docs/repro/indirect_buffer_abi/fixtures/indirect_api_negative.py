"""Machine-checked negative fixture for the missing generic pointer-table ABI."""
try:
    from ptodsl import pto
    @pto.jit(target='a5', backend='vpto', mode='explicit', insert_sync=False)
    def indirect(table: pto.ptr(pto.ptr(pto.f32, 'gm'), 'gm'), dst: pto.ptr(pto.f32, 'gm')):
        pto.load_ptr(table, 0)
    indirect.compile()
except Exception as exc:
    print('EXPECTED_NEGATIVE', type(exc).__name__, str(exc).splitlines()[0])
else:
    raise SystemExit('pointer-table API unexpectedly accepted')
