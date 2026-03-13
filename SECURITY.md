# Security Policy

TSDataForge is in active alpha development.

Security fixes are handled on a best-effort basis, with priority given to issues that affect untrusted input handling, generated artifacts, or the local UI surface.

## Supported versions

| Version | Support status |
|---|---|
| `main` | actively developed; best-effort security fixes |
| latest tagged release | best-effort fixes for important issues |
| older tags | not supported |

## How to report a vulnerability

Please report suspected vulnerabilities privately to:

- `zxw365@student.bham.ac.uk`

Use a subject line such as:

- `[TSDataForge security] short summary`

Please include:

- affected version, tag, or commit
- a short impact summary
- reproduction steps or a proof of concept
- whether the issue requires a crafted data file, local filesystem access, or user interaction
- any suggested mitigation if you already have one

Please do not disclose suspected vulnerabilities in a public GitHub issue before the maintainer has had a chance to review them.

## What is especially useful to report

- path traversal, unsafe file writes, or unsafe temp-file behavior
- code execution, command injection, or deserialization issues
- HTML/script injection in generated reports or bundle artifacts
- local UI exposure or unsafe defaults in `python -m tsdataforge ui`
- credential, token, or local-path leakage in generated files
- dependency or supply-chain issues with a concrete impact on this repository

## Response expectations

I will try to acknowledge reports within 5 business days and keep coordinated disclosure as the default path.

If the report is valid, the usual flow is:

1. confirm scope and impact
2. prepare a fix or mitigation
3. publish the patch
4. disclose the issue once users have a reasonable upgrade path
