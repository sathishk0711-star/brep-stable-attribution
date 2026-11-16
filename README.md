# Attribution baselines for B-rep face predictors (MFInstSeg)

Code and per-part outputs for the comparison of IG (16/64 steps), gradient×input, SmoothGrad, expected gradients, a GNNExplainer-style mask, occlusion and random ranking under a common deletion/insertion operator on the pretrained AAGNet predictor (64 held-out MFInstSeg parts). Accompanies the manuscript "Stable and Feature-Localized Attribution for Self-Supervised B-rep Graph Learning: Framework and Held-Out Pilot on MFInstSeg".

- `baselines/run_baselines.py` runs all methods; `baselines/aggregate.py` computes paired part-level bootstrap intervals (5,000 resamples, seed 20261103) and the figure.
- `baselines/rows.json`, `baselines/summary.json`: per-part and aggregated results.
- `dgl_shim/`: minimal single-graph stand-in for DGL (used where DGL could not be installed).

Not included (obtain from the upstream sources): the AAGNet code and checkpoint (https://github.com/whjdark/AAGNet) and the MFInstSeg graph data. Edit the absolute paths at the top of `run_baselines.py` before running. Add a licence of your choice before publishing.

## Small-scale full-pipeline training (Section 6.7)
`full_pipeline/` holds the scripts, logs, training histories and summaries. `train_stable.py` differs from the package's `train.py` only in learning rate (2e-4) and gradient clip (1.0); `evaluate.py` of the package needed `np.trapz`→`np.trapezoid` for NumPy 2. Default settings: 4 of 5 seeds diverged (see `run_full*.log`). Scores in `common_*.json` use one common zero-reference deletion/insertion operator.
