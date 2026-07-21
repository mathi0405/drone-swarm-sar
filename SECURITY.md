# Security Policy

## Supported versions

This is a research project. Security fixes are applied to the latest release
(`main`) only.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Use GitHub's [private vulnerability reporting](https://github.com/mathi0405/drone-swarm-sar/security/advisories/new)
or email **mathimanichandan@gmail.com** with:

- a description of the issue and its impact,
- steps to reproduce,
- affected version/commit.

You can expect an acknowledgement within a few days. Because this is a
simulation/research codebase (no network services, no user data), most reports
will concern dependency vulnerabilities or unsafe deserialization of untrusted
checkpoints — note that `torch.load` is used with `weights_only=False` for
trusted checkpoints only; do not load checkpoints from untrusted sources.
