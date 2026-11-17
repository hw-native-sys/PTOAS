#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Shared Feishu notifier for the A5 CI scripts. Source it, then call
# a5_notify_text "message".
#
# Two environment variables drive it (set them in /etc/ptoas-a5-ci.env):
#   A5_FEISHU_WEBHOOK_URL  https://open.feishu.cn/open-apis/bot/v2/hook/<id>
#   A5_FEISHU_SECRET       only needed when the bot enables 签名校验
#
# Without a webhook every call is a silent no-op (return 0). When a webhook is
# configured, a delivery failure returns non-zero so the caller can keep the
# "already reported" flag false and let the watchdog retry the alert.

a5_notify_text() {
  local text="$1"
  [[ -n "${A5_FEISHU_WEBHOOK_URL:-}" ]] || return 0
  command -v curl >/dev/null 2>&1 || return 0
  local py payload
  py="$(command -v python3 || true)"
  [[ -n "${py}" ]] || return 0
  payload="$("${py}" - "${text}" <<'PY'
import base64
import hashlib
import hmac
import json
import os
import sys
import time

text = sys.argv[1]
payload = {"msg_type": "text", "content": {"text": text}}
secret = os.environ.get("A5_FEISHU_SECRET", "").strip()
if secret:
    ts = str(int(time.time()))
    key = ("{}\n{}".format(ts, secret)).encode("utf-8")
    digest = hmac.new(key, digestmod=hashlib.sha256).digest()
    payload["timestamp"] = ts
    payload["sign"] = base64.b64encode(digest).decode("ascii")
print(json.dumps(payload, ensure_ascii=False))
PY
)" || return 0
  [[ -n "${payload}" ]] || return 0
  local code
  code="$(curl -sS -m 10 -o /dev/null -w '%{http_code}' \
    -X POST -H 'Content-Type: application/json' \
    -d "${payload}" "${A5_FEISHU_WEBHOOK_URL}" 2>/dev/null)" || return 1
  [[ "${code}" == "200" ]] || return 1
  return 0
}

# Classify a failure from the tail of a log so an alert says what broke and what
# to do next, instead of only "failed (rc=1)". Prints four tab-separated fields:
#   KIND  PHASE_HINT  ROOT_CAUSE  ACTION
# Unknown input still produces evidence (the last matching error line), because
# "unknown" without the line is useless to whoever is on call.
a5_classify_failure() {
  local log="${1:-}"
  local text="" line="" fails="" pattern=""
  if [[ -n "${log}" && -f "${log}" ]]; then
    text="$(tail -n 400 "${log}" 2>/dev/null)"
    pattern="No space left on device|insufficient free space|command not found|"
    pattern+="could not read Username|Authentication failed|CH\\.[0-9]+|"
    pattern+="no healthy idle|Temporary failure in name resolution|Could not resolve host|"
    pattern+="Connection timed out|Segmentation fault|workspace does not exist|"
    pattern+="not a PTOAS source tree|sample payload generation failed|"
    pattern+="cannot merge .* into|FAIL  |FAIL\\t"
    line="$(printf '%s\n' "${text}" | grep -E "${pattern}" | tail -n 1)"
    # Name the samples that failed, not the driver's summary line about them:
    # sample-build.log prints "<Name>(<file>)\tFAIL\t<reason>", and those lines
    # can sit far above the tail window on a full sweep.
    fails="$(grep -E $'\tFAIL\t' "${log}" 2>/dev/null | cut -f1 | tail -n 3 | paste -sd' ' -)"
  fi
  case "${text}" in
    *"sample payload generation failed"*)
      printf 'sample-build\tpayload\t样例生成失败，板测没有开始（0 例上板）：%s\t' \
             '看 sample-build.log 里标 FAIL 的样例，修好后重跑；这是源码/样例问题，不是板子\n' \
             "${fails:-${line:-见日志}}" ;;
    *"cannot merge"*)
      printf 'merge-conflict\tpreflight\t无法把主线合入 CI 分支，本次没有运行：%s\t' \
             '在 CI 分支上解决冲突并推送后重跑；在 checkout 里手工合并会被下次覆盖\n' \
             "${line:-见日志}" ;;
    *"No space left on device"*)
      printf 'disk-full\tbuild\t磁盘写满（ENOSPC）\t已清理旧运行的重目录；若再次出现，需要主机 owner 释放 /data（共享盘）\n' ;;
    *"insufficient free space"*)
      printf 'disk-low\tpreflight\t共享盘 /data 剩余空间不足 30 GiB，这次一行都没跑：%s\t' \
             'PTOAS CI 自身只占约 4 GiB，是其他租户把它写满了；' \
             '需要主机 owner 清理 /data 后重跑\n' "${line:-见日志}" ;;
    *"command not found"*)
      printf 'script-error\tstartup\t脚本调用了未定义的命令/函数：%s\t修 CI 脚本（先 shellcheck -S error），修好后重跑\n' "${line:-见日志}" ;;
    *"could not read Username"*|*"Authentication failed"*|*"CH."[0-9]*)
      printf 'git-auth\tsync\tgit 认证失败：%s\t检查 /data/ptoas-ci/.git-credentials（CI 专属、chmod 600）\n' "${line:-见日志}" ;;
    *"no healthy idle"*)
      printf 'device\tdevice\t没有健康空闲的 A5 卡：%s\t确认卡 2 未被其他租户占用；必要时 npu-smi info -t health\n' "${line:-见日志}" ;;
    *"Temporary failure in name resolution"*|*"Could not resolve host"*|*"Connection timed out"*)
      printf 'network\tsync\t网络/域名解析失败：%s\t稍后重试；持续失败需检查主机网络\n' "${line:-见日志}" ;;
    *"workspace does not exist"*|*"not a PTOAS source tree"*)
      printf 'config	startup	路径配置错误：%s	检查 --workspace 与 checkout 是否被清理
' "${line:-见日志}" ;;
    *"Segmentation fault"*)
      printf 'crash\tboard\t板测进程崩溃（SIGSEGV）：%s\t重跑一次确认是否偶发；稳定复现则转给对应 owner\n' "${line:-见日志}" ;;
    *)
      printf 'unknown\t?\t未识别的失败\t看日志定位：%s\n' "${line:-nightly.log 末尾 20 行}" ;;
  esac
}

# Convenience wrapper: format the classification as alert lines.
a5_failure_summary() {
  local log="${1:-}" kind phase cause action
  IFS=$'\t' read -r kind phase cause action < <(a5_classify_failure "${log}")
  printf 'kind=%s phase=%s\n原因：%s\n处置：%s' "${kind}" "${phase}" "${cause}" "${action}"
}
