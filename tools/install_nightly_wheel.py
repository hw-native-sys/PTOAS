#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Install the latest wheel published by the PTOAS nightly GitHub release."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_REPOSITORY = "hw-native-sys/PTOAS"
DEFAULT_TAG = "nightly"
NETWORK_TIMEOUT_SECONDS = 30
STALE_WHEEL_AGE = datetime.timedelta(hours=48)


class WheelSelection:
    def __init__(
        self,
        name: str,
        url: str,
        updated_at: datetime.datetime | None,
        digest: str | None,
    ) -> None:
        self.name = name
        self.url = url
        self.updated_at = updated_at
        self.digest = digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the latest compatible wheel from a PTOAS GitHub release."
    )
    parser.add_argument(
        "--repository",
        default=DEFAULT_REPOSITORY,
        help=f"GitHub repository (default: {DEFAULT_REPOSITORY})",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"GitHub release tag (default: {DEFAULT_TAG})",
    )
    parser.add_argument(
        "--package",
        default="ptoas",
        help="Distribution to install (default: ptoas)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the wheel without installing it",
    )
    parser.add_argument(
        "--sha256",
        help="Expected SHA-256 digest of the selected wheel",
    )
    return parser.parse_args()


def github_request(url: str) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ptoas-nightly-wheel-installer",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = ""
        if error.code in (403, 429):
            detail = " Set GITHUB_TOKEN if GitHub API rate limiting is suspected."
        try:
            response_body = error.read().decode("utf-8", errors="replace")
            response_json = json.loads(response_body)
            message = response_json.get("message") if isinstance(response_json, dict) else None
            if message:
                detail += f" GitHub message: {message}."
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        raise RuntimeError(
            f"GitHub API request failed with HTTP {error.code}: {error.reason}.{detail}"
        ) from error
    except TimeoutError as error:
        raise RuntimeError(
            f"GitHub API request timed out after {NETWORK_TIMEOUT_SECONDS} seconds"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"unable to reach GitHub API: {error.reason}") from error


def parse_updated_at(value: object) -> datetime.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
        return timestamp
    except ValueError:
        return None


def parse_digest(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value.removeprefix("sha256:")
    if len(digest) == 64 and all(character in "0123456789abcdefABCDEF" for character in digest):
        return digest.lower()
    return None


def packaging_modules():
    try:
        from packaging.tags import sys_tags
        from packaging.utils import canonicalize_name, parse_wheel_filename
        return sys_tags, canonicalize_name, parse_wheel_filename
    except ImportError:
        try:
            from pip._vendor.packaging.tags import sys_tags
            from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
            return sys_tags, canonicalize_name, parse_wheel_filename
        except ImportError as error:
            raise RuntimeError(
                "packaging support is required; use a Python environment with pip "
                "or install it with 'python -m pip install packaging'"
            ) from error


def _wheel_candidate(asset, canonical_package, tag_rank, parse_wheel_filename, canonicalize_name):
    if not isinstance(asset, dict):
        return None
    name = asset.get("name")
    url = asset.get("browser_download_url")
    if not isinstance(name, str) or not name.endswith(".whl") or not isinstance(url, str):
        return None
    try:
        distribution, version, build, wheel_tags = parse_wheel_filename(name)
    except (TypeError, ValueError):
        return None
    if canonicalize_name(str(distribution)) != canonical_package:
        return None
    matching_ranks = [tag_rank[tag] for tag in wheel_tags if tag in tag_rank]
    if not matching_ranks:
        return None
    return (
        parse_updated_at(asset.get("updated_at")),
        version,
        build or (-1, ""),
        min(matching_ranks),
        name,
        url,
        parse_digest(asset.get("digest")),
    )


def _wheel_sort_key(item):
    return (
        item[0] or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
        item[1],
        item[2],
        -item[3],
        item[4],
    )


def select_wheel(release: object, package: str = "ptoas") -> WheelSelection:
    try:
        sys_tags, canonicalize_name, parse_wheel_filename = packaging_modules()
    except RuntimeError as error:
        raise RuntimeError(
            str(error)
        ) from error

    if not isinstance(release, dict) or not isinstance(release.get("assets"), list):
        raise RuntimeError("GitHub release response does not contain wheel assets")

    supported_tags = list(sys_tags())
    tag_rank = {tag: rank for rank, tag in enumerate(supported_tags)}
    canonical_package = canonicalize_name(package)
    candidates = [
        candidate
        for asset in release["assets"]
        if (candidate := _wheel_candidate(
            asset,
            canonical_package,
            tag_rank,
            parse_wheel_filename,
            canonicalize_name,
        )) is not None
    ]

    if not candidates:
        raise RuntimeError(
            f"no compatible {package} wheel found in the {release.get('tag_name', 'requested')} release"
        )
    selected = max(candidates, key=_wheel_sort_key)
    return WheelSelection(selected[4], selected[5], selected[0], selected[6])


def download(url: str, destination: Path, expected_sha256: str | None = None) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ptoas-nightly-wheel-installer"},
    )
    try:
        with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response, destination.open(
            "wb"
        ) as output:
            digest = hashlib.sha256()
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
    except TimeoutError as error:
        raise RuntimeError(
            f"wheel download timed out after {NETWORK_TIMEOUT_SECONDS} seconds"
        ) from error
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(f"failed to download wheel: {error}") from error
    if expected_sha256 and digest.hexdigest() != expected_sha256.lower().removeprefix("sha256:"):
        raise RuntimeError(f"SHA-256 mismatch for downloaded wheel {destination.name}")


def main() -> int:
    args = parse_args()
    try:
        release_url = (
            f"https://api.github.com/repos/{args.repository}/releases/tags/{args.tag}"
        )
        release = github_request(release_url)
        selection = select_wheel(release, args.package)
        print(f"Selected wheel: {selection.name}")
        if selection.updated_at:
            age = datetime.datetime.now(datetime.timezone.utc) - selection.updated_at
            if age > STALE_WHEEL_AGE:
                print(
                    f"warning: selected wheel was last updated {age.total_seconds() / 3600:.1f} hours ago",
                    file=sys.stderr,
                )
        if args.dry_run:
            print(selection.url)
            return 0

        with tempfile.TemporaryDirectory(prefix="ptoas-nightly-") as directory:
            wheel_path = Path(directory) / selection.name
            expected_sha256 = args.sha256 or selection.digest
            download(selection.url, wheel_path, expected_sha256)
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheel_path),
            ]
            subprocess.run(command, check=True)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
