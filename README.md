# Motion Challenge · heft

[Author project](https://github.com/Axellwppr/motion_tracking) · [Checkpoint](https://github.com/Axellwppr/motion_tracking/tree/0d5ba31e33397f3543d350d98b637e26d92f470a/sim2real/config/g1/ckpts/G1_PMG) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/motion-challenge-heft.git
cd motion-challenge-heft
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. MuJoCo 3.17 CPU batch; `--jobs 8` sets rollout and
verification parallelism. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `motion-challenge`
submodule owns datasets, physics, metrics and verification. See `--help` for options.
