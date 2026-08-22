# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression tests for the PTODSL TileLib daemon lifecycle.

Each test uses a **dedicated socket path** (``--daemon-socket-path``) so it
does not interfere with daemons started by other tests running in parallel
(``ctest -j4``).  Only the test's own socket/PID is checked — no global
``pgrep`` or ``/tmp/tilelib_daemon_*.sock`` scanning.

Covers:

* Two consecutive ``_core.main`` calls in the **same process**: the second
  ``start()`` must ``stop()`` the first daemon (singleton processInfo) before
  launching a new one.
* A normally-terminated daemon exits via SIGTERM, not force-killed.
* After ptoas exits, the test's own socket is removed.
* ``stop()`` does not steal the exit status of an unrelated child in the
  same ptoas process.
* Startup-timeout path reaps the PID when the daemon never opens its socket.

Known test gap (not covered):
  The ``stop()``-returns-false path (``SyscallError``) that prevents
  ``start()`` from overwriting the old PID is not exercised here because it
  requires a non-deterministic syscall failure to trigger.  Adding a test
  seam for process operations would allow injecting this; tracked as a
  follow-up.
"""

import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]

_MINIMAL_PTO = """\
module {
  func.func @daemon_lifecycle_probe() {
    %t = pto.alloc_tile : !pto.tile_buf<vec, 8x8xf32>
    pto.tadd ins(%t, %t : !pto.tile_buf<vec, 8x8xf32>, !pto.tile_buf<vec, 8x8xf32>)
         outs(%t : !pto.tile_buf<vec, 8x8xf32>)
    return
  }
}
"""


def _run_ptoas_subprocess(
    pto_text: str,
    socket_path: str | None = None,
    extra_args: list[str] | None = None,
) -> tuple[int, str]:
    """Run ptoas as a subprocess and return (returncode, stderr_text)."""
    ptoas = shutil.which("ptoas")
    assert ptoas is not None, "ptoas must be on PATH"
    with TemporaryDirectory() as td:
        pto_file = Path(td) / "probe.pto"
        pto_file.write_text(pto_text)
        out_file = str(Path(td) / "out.mlir")
        args = [
            ptoas,
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--tile-lib-backend=ptodsl",
            f"--ptodsl-python-exe={sys.executable}",
            "--emit-vpto",
            str(pto_file),
            "-o", out_file,
        ]
        if socket_path:
            args.append(f"--daemon-socket-path={socket_path}")
        if extra_args:
            args.extend(extra_args)
        r = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return r.returncode, r.stderr


class TestDaemonLifecycle(unittest.TestCase):

    def test_double_start_reaps_first_daemon_in_process(self):
        """Two consecutive _core.main calls in the same process.

        This exercises the singleton processInfo path: the second start()
        must stop() the first daemon before launching a new one.  If stop()
        fails with a syscall error it must NOT overwrite the old PID.

        Verifies:
        - Both compilations succeed (rc == 0).
        - Both daemons start and stop.
        - Both sockets are cleaned up after the process exits.

        Note: this test does not directly observe the first daemon's PID
        between the two calls; that would require a test seam in the C++
        process manager.  The singleton overwrite path is covered indirectly
        by the fact that the second start succeeds (it would fail if stop()
        returned false and start() aborted).
        """
        sock1 = f"/tmp/test_daemon_lc_{os.getpid()}_dbl1.sock"
        sock2 = f"/tmp/test_daemon_lc_{os.getpid()}_dbl2.sock"
        for s in (sock1, sock2):
            if os.path.exists(s):
                os.unlink(s)

        driver = f"""
import os, sys, time
from ptoas import _core
import tempfile
from pathlib import Path

PTO = '''{_MINIMAL_PTO}'''

with tempfile.TemporaryDirectory() as td:
    pto = Path(td) / "probe.pto"
    pto.write_text(PTO)

    # First call: starts daemon on sock1, stops it on exit.
    out1 = str(Path(td) / "out1.mlir")
    rc1 = _core.main([
        "ptoas", "--pto-arch=a5", "--pto-backend=vpto",
        "--tile-lib-backend=ptodsl",
        f"--ptodsl-python-exe={{sys.executable}}",
        f"--daemon-socket-path={sock1}",
        "--emit-vpto", str(pto), "-o", out1,
    ])
    if rc1 != 0:
        print("FIRST_RUN_FAILED", file=sys.stderr)
        sys.exit(1)

    # Second call in the same process: start() must stop() the first
    # daemon (singleton processInfo) before starting a new one on sock2.
    out2 = str(Path(td) / "out2.mlir")
    rc2 = _core.main([
        "ptoas", "--pto-arch=a5", "--pto-backend=vpto",
        "--tile-lib-backend=ptodsl",
        f"--ptodsl-python-exe={{sys.executable}}",
        f"--daemon-socket-path={sock2}",
        "--emit-vpto", str(pto), "-o", out2,
    ])
    if rc2 != 0:
        print("SECOND_RUN_FAILED", file=sys.stderr)
        sys.exit(1)

    # Socket cleanup happens at process exit (atexit).  We check sockets
    # from the parent after this driver exits.
    print("OK")
"""
        r = subprocess.run(
            [sys.executable, "-c", driver],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(r.returncode, 0,
                         f"in-process double-start failed: {r.stderr}")
        self.assertIn("OK", r.stdout,
                      f"double-start check failed: {r.stderr}")
        # After the driver exits, atexit has run and cleaned up both sockets.
        time.sleep(0.3)
        self.assertFalse(os.path.exists(sock1), f"sock1 leaked: {sock1}")
        self.assertFalse(os.path.exists(sock2), f"sock2 leaked: {sock2}")

    def test_graceful_exit_no_force_kill(self):
        """Daemon must exit via SIGTERM, not SIGKILL."""
        sock = f"/tmp/test_daemon_lc_{os.getpid()}_grace.sock"
        if os.path.exists(sock):
            os.unlink(sock)
        try:
            rc, err = _run_ptoas_subprocess(_MINIMAL_PTO, socket_path=sock)
            self.assertEqual(rc, 0, f"ptoas failed (rc={rc}): {err}")
            self.assertIn("daemon started", err)
            self.assertIn("daemon stopped", err)
            self.assertNotIn("force-killed", err,
                              f"daemon was force-killed: {err}")
            self.assertFalse(os.path.exists(sock),
                              f"socket leaked: {sock}")
        finally:
            if os.path.exists(sock):
                os.unlink(sock)

    def test_no_socket_after_stop(self):
        """After ptoas exits, the test's own socket must be removed."""
        sock = f"/tmp/test_daemon_lc_{os.getpid()}_nosock.sock"
        if os.path.exists(sock):
            os.unlink(sock)
        try:
            rc, err = _run_ptoas_subprocess(_MINIMAL_PTO, socket_path=sock)
            self.assertEqual(rc, 0, f"ptoas failed (rc={rc}): {err}")
            self.assertFalse(os.path.exists(sock),
                              f"socket remains after stop: {sock}")
        finally:
            if os.path.exists(sock):
                os.unlink(sock)

    def test_stop_does_not_reap_unrelated_child_in_process(self):
        """stop() must not steal exit status of a child in the same ptoas process.

        We write a small Python driver that forks a child, then calls
        ``ptoas._core.main`` in-process (so the daemon and the forked child
        share the same parent), then reaps the child.  If stop() used
        waitpid(-1) it would steal the child's status.
        """
        driver = """
import os, sys, time
from ptoas import _core

child_pid = os.fork()
if child_pid == 0:
    time.sleep(0.5)
    os._exit(42)

import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as td:
    pto = Path(td) / "probe.pto"
    pto.write_text('''module {
  func.func @daemon_lifecycle_probe() {
    %t = pto.alloc_tile : !pto.tile_buf<vec, 8x8xf32>
    pto.tadd ins(%t, %t : !pto.tile_buf<vec, 8x8xf32>, !pto.tile_buf<vec, 8x8xf32>)
         outs(%t : !pto.tile_buf<vec, 8x8xf32>)
    return
  }
}
''')
    sock = f"/tmp/test_daemon_lc_driver_{os.getpid()}.sock"
    rc = _core.main([
        "ptoas", "--pto-arch=a5", "--pto-backend=vpto",
        "--tile-lib-backend=ptodsl",
        f"--ptodsl-python-exe={sys.executable}",
        f"--daemon-socket-path={sock}",
        "--emit-vpto", str(pto), "-o", str(Path(td) / "out.mlir"),
    ])
    if rc != 0:
        sys.exit(1)
    _, status = os.waitpid(child_pid, 0)
    if not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 42):
        print(f"UNRELATED_CHILD_STOLEN status={status}", file=sys.stderr)
        sys.exit(2)
    print("OK")
"""
        r = subprocess.run(
            [sys.executable, "-c", driver],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(r.returncode, 0,
                         f"driver failed (rc={r.returncode}): {r.stderr}")
        self.assertIn("OK", r.stdout,
                      f"child status check failed: {r.stderr}")
        self.assertNotIn("UNRELATED_CHILD_STOLEN", r.stderr,
                         f"stop() stole unrelated child: {r.stderr}")

    def test_timeout_path_reaps_pid(self):
        """Startup-timeout path must reap PID when daemon never opens socket.

        Use a fake Python interpreter (a script that writes its own PID to a
        file then sleeps) as ``--ptodsl-python-exe``.  ptoas starts it, but
        it never creates the socket.  The startup timeout fires and
        terminateAndReap must clean up the PID.
        """
        ptoas = shutil.which("ptoas")
        assert ptoas is not None
        sock = f"/tmp/test_daemon_lc_{os.getpid()}_timeout.sock"
        with TemporaryDirectory() as td:
            pid_file = Path(td) / "child_pid"
            fake_python = Path(td) / "fake_python"
            fake_python.write_text(
                f"#!/bin/sh\necho $$ > {pid_file}\nsleep 30\n"
            )
            fake_python.chmod(0o755)

            pto_file = Path(td) / "probe.pto"
            pto_file.write_text(_MINIMAL_PTO)
            out_file = str(Path(td) / "out.mlir")
            r = subprocess.run(
                [
                    ptoas,
                    "--pto-arch=a5",
                    "--pto-backend=vpto",
                    "--tile-lib-backend=ptodsl",
                    f"--ptodsl-python-exe={fake_python}",
                    f"--daemon-socket-path={sock}",
                    "--emit-vpto",
                    str(pto_file),
                    "-o", out_file,
                ],
                capture_output=True, text=True, timeout=30,
            )
            err = r.stderr

            self.assertIn("socket not created", err,
                          f"should report socket timeout: {err}")
            self.assertNotEqual(r.returncode, 0,
                                f"ptoas should fail (non-zero rc): {err}")

            if pid_file.exists():
                fake_pid = int(pid_file.read_text().strip())
                alive = True
                try:
                    os.kill(fake_pid, 0)
                except ProcessLookupError:
                    alive = False
                self.assertFalse(alive,
                                 f"fake interpreter (pid={fake_pid}) still alive; "
                                 "timeout path did not reap the PID")
            else:
                self.fail("fake interpreter did not write its PID")
            self.assertFalse(os.path.exists(sock),
                              f"socket leaked after timeout: {sock}")


if __name__ == "__main__":
    unittest.main()
