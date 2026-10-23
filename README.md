# Motion Challenge · heft

[heft](https://github.com/Axellwppr/motion_tracking) checkpoint adapter. G1 29DoF; LAFAN40 is the first suite.

## Install and run

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-heft.git
cd motion-challenge-heft
uv sync
uv run evaluate.py
```

Reuses local `checkpoints/policy.onnx` or the external cache; missing artifacts are
fetched and hash-checked. Use `--weights PATH --data PATH --offline` for local files.
Results and traces are written outside Git. `--help` lists options.

The root `motion-challenge` submodule owns data, physics, metrics and verification.
This repository only adapts model inference; no `sim2real` dependency.

[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.


## Checkpoint source

[Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking/blob/0d5ba31e33397f3543d350d98b637e26d92f470a/README.md) · [G1_PMG/policy.onnx + policy.onnx.data](https://github.com/Axellwppr/motion_tracking/tree/0d5ba31e33397f3543d350d98b637e26d92f470a/sim2real/config/g1/ckpts/G1_PMG). Author weights with verified input wrapper.

The distributed single-file ONNX adds one input Concat. All 33 initializer tensors and all 29 native graph nodes are preserved exactly; action outputs matched on 100 random input cases (maximum absolute error 0). This checks the checkpoint wrapper, not full controller/reset parity.

One `uv run evaluate.py` rolls out each complete motion once, then verifies the trace.
The same trajectory provides success, completion and all six MPBPE values in one
`result.json`, with one source revision. No branch switching is needed.
