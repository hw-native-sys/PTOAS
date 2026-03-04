if __name__ == "__main__":
    # Regression for zhangstevenunity/PTOAS#185:
    # barrier_sync should support any SyncOpType that can be mapped to a PIPE
    # (not just TMATMUL/TVEC).
    print(
        r"""module {
  func.func @test_barrier_sync_py() {
    pto.barrier_sync[<TLOAD>]
    pto.barrier_sync[<TSTORE_VEC>]
    pto.barrier_sync[<TVEC>]
    return
  }
}
"""
    )
