# Upstream code

Every baseline in this repository is implemented in its own folder on the
shared harness. **Nothing here imports upstream code at run time.** The
repositories below were used as references when writing or adapting those
implementations. They are not vendored, because their licences travel with
them. To inspect them next to this code, clone them into `third_party/`
(git-ignored). `audited_gtep.upstream_commits()` records their commits in
`protocol.json` whenever they are present as git checkouts.

| Folder in `third_party/` | Used for | Source | Commit |
|---|---|---|---|
| `pec` | PEC mechanism: a shared frozen teacher with independent per-class students (`baselines/PEC/pec.py`) | https://github.com/michalzajac-ml/pec | `3c15633` |
| `spacenet` | SpaceNet reference (`baselines/SpaceNet/`) | https://github.com/GhadaSokar/SpaceNet *(URL not recorded by the campaign; verify)* | not recorded |
| `uniclun` | UniCLUN reference. The upstream repository omits the model module, so `baselines/UniCLUN/uniclun.py` is re-implemented from the paper. | *(not recorded by the campaign; fill in from the paper)* | not recorded |
| `gtep/PyCIL` | GTEP protocol reference code | https://github.com/G-U-N/PyCIL *(verify)* | not recorded |
| `gtep/LAMDA-PILOT` | GTEP protocol reference code | https://github.com/sun-hailong/LAMDA-PILOT *(verify)* | not recorded |

Adaptations to the GTEP schedule are recorded in the `upstream_adaptations`
field of `audited_gtep.queue`'s `protocol.json` and in docs/BASELINES.md.
