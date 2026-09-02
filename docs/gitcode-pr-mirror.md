# GitCode PR mirror

This repository-specific workflow is designed for personal GitHub and GitCode
Forks. It mirrors pull requests from a personal GitHub Fork into the GitCode
upstream project `cann/pto-as`; GitHub `test` maps to GitCode `master` by
default while this workflow is being validated. Set the repository variable
`GITHUB_UPSTREAM_BRANCH=main` when the workflow is ready for production use.

## How it works

```text
Personal GitHub Fork branch push
        ↓
Find the matching open PR in hw-native-sys/PTOAS
        ↓
Compute the PR diff against the configured upstream branch
        ↓
Push mirror/github-pr-<number> to the personal GitCode Fork
        ↓
Create or reuse a PR to cann/pto-as:master
```

The workflow synchronizes the final diff instead of cherry-picking commits, so
the two repositories do not need identical Git histories. Later pushes to the
same GitHub branch update the same GitCode mirror branch and PR.

## Personal setup

Each contributor must first create both Forks:

```text
GitHub: <github-user>/PTOAS
GitCode: <gitcode-user>/pto-as
```

Copy this workflow into the personal GitHub Fork's default branch and enable
Actions. Configure these repository variables in that Fork:

- `GITCODE_FORK_OWNER`: the GitCode user or namespace that owns the Fork.
- `GITCODE_FORK_REPOSITORY`: the Fork repository name; normally `pto-as`.
- `GITCODE_COMMIT_EMAIL`: a verified email for the GitCode account.

Configure these repository secrets in the same GitHub Fork:

- `GITCODE_MIRROR_SSH_KEY`: a write-enabled SSH key for the personal GitCode
  Fork; it must not be allowed to push `master`.
- `GITCODE_KNOWN_HOSTS`: the verified SSH host key for `gitcode.com`.
- `GITCODE_MIRROR_TOKEN`: a GitCode token for reading and creating PRs in
  `cann/pto-as`.

The mirror commit author and committer are set to:

```text
<GITCODE_FORK_OWNER> <GITCODE_COMMIT_EMAIL>
```

## Daily usage

1. Create a GitHub PR from the personal Fork to the configured upstream branch
   (`test` by default; set `GITHUB_UPSTREAM_BRANCH=main` for production).
2. Push to the PR branch in the personal GitHub Fork.
3. The `push` workflow finds the matching upstream PR and mirrors it.
4. Continue pushing to that GitHub branch; do not edit the GitCode mirror branch.

For an existing PR or a retry, open Actions in the personal GitHub Fork, run
`Mirror PR to GitCode`, and enter the upstream PR number.

The matching upstream PR must be open, target the configured upstream branch,
and have its head repo equal to the personal GitHub Fork. If no unique match is
found, the workflow stops without changing GitCode.

## Security requirements

This workflow does not check out or execute the pull request source code. It
only reads Git objects, computes a patch, and uses credentials from the
personal Fork's own Actions secrets. Keep all credentials in secrets, rotate
tokens that have been exposed, and never allow the mirror key to push the
GitCode protected branch.
