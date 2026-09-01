# GitCode PR mirror

This repository mirrors GitHub pull requests to the GitCode upstream project
`cann/pto-as`. GitHub `main` maps to GitCode `master`.

## One-time administrator setup

The workflow is `.github/workflows/mirror-pr-to-gitcode.yml`. It runs from the
trusted default branch with `pull_request_target`; it reads pull request Git
objects and never executes code from the pull request.

Configure these GitHub Actions secrets:

- `GITCODE_MIRROR_SSH_KEY`: a write-enabled key for the `lllzzzddd/pto-as`
  fork. It must not be allowed to push `master`.
- `GITCODE_KNOWN_HOSTS`: the verified SSH host key for `gitcode.com`.
- `GITCODE_MIRROR_TOKEN`: a GitCode token owned by `lllzzzddd` with permission
  to read pull requests and create them in `cann/pto-as`.

The GitCode commit identity is intentionally fixed to the required account:

```text
lllzzzddd <liuzidi1@huawei.com>
```

Use the verified email configured for that account if it changes.

## Operation

For each GitHub pull request targeting `main`, the workflow maintains this
branch in the GitCode fork:

```text
mirror/github-pr-<number>
```

It creates one GitCode pull request from that branch to
`cann/pto-as:master`. Later GitHub pushes update the same branch, so the
existing GitCode pull request is updated automatically.

To mirror an already-open pull request, run the workflow manually and enter
its GitHub pull request number.

If applying the patch fails, the workflow stops without pushing a new mirror
commit. Resolve the divergence in the GitHub pull request and run it again.

## Security requirements

Do not change this workflow to check out or execute the pull request head. A
fork pull request is untrusted input, while the workflow has access to the
GitCode credentials. Rotate any token that has been exposed and keep all
credentials in GitHub Actions secrets only.
