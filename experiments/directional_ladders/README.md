# Directional stability and bimodality experiments

The source and compact raw arrays for every currently completed continuation are included. Model weights and datasets remain external, with source paths and checkpoint SHA256 values in each meta.json. No claim is made that a missing external checkpoint is recoverable from plotting arrays.

From the repository root, run `OPENBLAS_NUM_THREADS=1 python experiments/directional_ladders/source/render.py` to regenerate PDF/PNG figures. For the full appendix and shape audit, run `python experiments/directional_ladders/source/analyze_publish.py`; this additionally needs pdflatex. These plotting commands resolve their input/output paths relative to the checkout and need no GPU or parent checkpoints.

The original GPU training/probing runner and shared helper sources are retained as provenance. They still require the eoss model/data implementation, PyTorch, CIFAR-10 subset, and specified checkpoint files; run.sbatch contains the Gandalf configuration. They are not represented as a self-contained training installation.

The original anchor starts at checkpoint10001. New cases use checkpoint20000 where available; two large-batch parents use their last saved checkpoint10001. Cases are indexed by architecture and batch, with one declared parent initialization and one IID continuation per case. H is reciprocal inverse moment, G is exp(mean log magnitude), A1 is mean absolute gain, and A2 is RMS gain. Fixed-axis and batch-gradient compressions are kept separate. Histograms are finite-window occupancy and do not prove stationarity or a universal harmonic-to-bimodal implication.

The experiment export is refreshed after final report generation; STATUS.md and live Slurm records are needed to distinguish running or interrupted jobs from completed data.
