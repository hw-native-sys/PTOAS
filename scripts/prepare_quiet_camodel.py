#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Prepare a shared quiet camodel directory.

The output directory is a simulator lib directory view: non-config entries are
symlinked from the original camodel lib directory, and config.json is copied and
patched to reduce simulator log/dump I/O.
"""

import argparse
import fcntl
import json
import os
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Original camodel simulator lib directory.")
    parser.add_argument("--output-dir", required=True, help="Quiet camodel output directory.")
    return parser.parse_args()


def patch_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("LOG", {})["flush_level"] = 6
    config.setdefault("LOG", {})["core_enable_mask"] = ["0x0"]
    wrapper = config.setdefault("WRAPPER", {})
    wrapper["adapter_log_file_level"] = 6
    wrapper["aic_wrap_log_file_level"] = 6
    wrapper["cosim_log_file_level"] = 6
    wrapper["cosim_log_flush_level"] = 6
    wrapper["cosim_log_scr_level"] = 6
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
        handle.write("\n")


def prepare_quiet_camodel(source_dir, quiet_dir):
    source_dir = os.path.realpath(source_dir)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"camodel source dir is invalid: {source_dir}")

    os.makedirs(quiet_dir, exist_ok=True)
    lock_path = os.path.join(quiet_dir, ".quiet-camodel.lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return prepare_quiet_camodel_locked(source_dir, quiet_dir)


def prepare_quiet_camodel_locked(source_dir, quiet_dir):
    source_marker = os.path.join(quiet_dir, ".quiet-camodel-source")
    config_path = os.path.join(quiet_dir, "config.json")
    if os.path.isfile(source_marker):
        with open(source_marker, "r", encoding="utf-8") as handle:
            existing_source = handle.read().strip()
        if existing_source != source_dir:
            raise RuntimeError(
                f"output dir already points to {existing_source}, cannot reuse it for {source_dir}"
            )
        if os.path.isfile(config_path):
            return os.path.abspath(quiet_dir)

    for name in os.listdir(source_dir):
        src = os.path.join(source_dir, name)
        dst = os.path.join(quiet_dir, name)
        if name == "config.json":
            shutil.copy2(src, dst)
            os.chmod(dst, os.stat(dst).st_mode | 0o200)
            continue
        if os.path.lexists(dst):
            continue
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"camodel config.json not found under {source_dir}")
    patch_config(config_path)
    with open(source_marker, "w", encoding="utf-8") as handle:
        handle.write(source_dir + "\n")
    return os.path.abspath(quiet_dir)


def main():
    args = parse_args()
    quiet_dir = os.path.abspath(args.output_dir)
    print(prepare_quiet_camodel(args.source_dir, quiet_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
