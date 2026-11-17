# Student pilot (Section 6.4): per-part results and scripts

Teacher-student pilot on the frozen pretrained AAGNet predictor (MFInstSeg, 128 train / 32 val / 64 held-out parts).

- `stage1_teacher.py`: 64-step IG (zero reference) targets for the three most frequently predicted classes of each training/validation part.
- `stage2_student5c.py`: trains five students (seeds 11, 23, 37, 53, 71), evaluates raw and minimum-L2-projected outputs on the 64 held-out parts. Standardized features are clipped to [-5, 5] (post hoc fix; the first run failed on one test part).
- `agg5.py`: two-stage (seed x part) bootstrap, 5,000 resamples, seed 20261103.
- `student_rows5c.json`: per-part, per-seed results (640 rows). `student5c_summary.json`: aggregates and per-part seed-averaged differences used for Figure 5.
- `make_fig5.py`: draws Figure 5 from `student5c_summary.json`.

Needs the package's `validation/shared.py`, the AAGNet checkpoint and the MFInstSeg subset; edit the absolute paths at the top of the scripts.
