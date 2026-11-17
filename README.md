# Attribution baselines for B-rep face predictors (MFInstSeg)

Code and per-part outputs for the comparison of IG (16/64 steps), gradient×input, SmoothGrad, expected gradients, a GNNExplainer-style mask, occlusion and random ranking under a common deletion/insertion operator on the pretrained AAGNet predictor (64 held-out MFInstSeg parts). Accompanies the manuscript "Stable and Feature-Localized Attribution for Self-Supervised B-rep Graph Learning: Framework and Held-Out Pilot on MFInstSeg".

- `baselines/run_baselines.py` runs all methods; `baselines/aggregate.py` computes paired part-level bootstrap intervals (5,000 resamples, seed 20261103) and the figure.
- `baselines/rows.json`, `baselines/summary.json`: per-part and aggregated results.
- `dgl_shim/`: minimal single-graph stand-in for DGL (used where DGL could not be installed).

Not included (obtain from the upstream sources): the AAGNet code and checkpoint (https://github.com/whjdark/AAGNet) and the MFInstSeg graph data. Edit the absolute paths at the top of `run_baselines.py` before running. Add a licence of your choice before publishing.

## Small-scale full-pipeline training (Section 6.7)
`full_pipeline/` holds the scripts, logs, training histories and summaries. `train_stable.py` differs from the package's `train.py` only in learning rate (2e-4) and gradient clip (1.0); `evaluate.py` of the package needed `np.trapz`→`np.trapezoid` for NumPy 2. Default settings: 4 of 5 seeds diverged (see `run_full*.log`). All five seeds (11, 23, 37, 53, 71) were then rerun with the stable settings (`run_stable.sh` for 11/23/53, `run_stable2.sh` for 37/71); per-part outputs are `parts_stable_seed_*.json`, the five-seed summary is `common_summary.json`/`full_summary.json`. Scores in `common_*.json` use one common zero-reference deletion/insertion operator.


## Student pilot (Section 6.4)
`student_pilot/` holds the teacher-target script (`stage1_teacher.py`), the five-seed student script (`stage2_student5c.py`, features clipped to [-5,5], a disclosed post hoc change), aggregation (`agg5.py`), per-part rows, summary, logs and `make_fig5.py`. See `student_pilot/README_student_pilot.md`.

## Licence
MIT (see LICENSE).

## Data manifest and checkpoints (reproducing the experiments)
- `data_manifest/`: the exact 224-part subset used (128 train / 32 val / 64 test; `selection.json`, `selected_*.txt`), the normalization statistics (`attr_stat.json`) and SHA-256 checksums of every graph and label file (`subset_sha256.json`). Obtain MFInstSeg from the AAGNet authors (see the manuscript, ref. [29]), copy these parts into one folder, and verify the checksums.
- `checkpoints/`: the five fine-tuned full-pipeline checkpoints (stable settings, seeds 11, 23, 37, 53, 71) with checksums. They start from the AAGNet checkpoint, which is not redistributed here.
