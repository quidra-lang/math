# Development and release workflow

Math uses two permanent branches:

- `main` is the latest published stable source once Math has published its first release. Before that first release, `main` may contain repository bootstrap history only and is not a release or installation identity.
- `develop` is the long-lived integration branch for the next release.

Ordinary work goes directly to `develop`. Do not create feature branches for
routine work and never force-move either permanent branch.

A release must keep `project.toml`, `quidra.package`, the immutable
`vMAJOR.MINOR.PATCH` tag, and the tested Core compatibility range consistent.
Math uses the exact same `MAJOR.MINOR.PATCH` version as Core; it never chooses
or advances a package version independently.

Every Math release requires the same-version Core tag to exist first. Any
package that explicitly declares a Math dependency is released only after the
same-version Math tag exists. Reusable neural-network kernels and fusion policy belong to
NN, while DNN composes model architectures from NN and Math; neither is a
reason to move generic matrix multiplication or other numerical semantics
back into Core.

After editing `project.toml`, regenerate the compatibility manifest with
`quidra package sync .`; CI uses `quidra package validate .` so the Core
package parser/generator is the single metadata implementation. Before
publishing, run integration tests, the AOT/package-native contract, fake-GPU
contracts, and any available real GPU validation. The release workflow verifies
the immutable same-version Core tag, validates metadata with that released Core,
creates the matching immutable Math tag from tested `main`, and publishes the
GitHub Release.
