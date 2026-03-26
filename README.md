This folder contains code for experiments in "CLIP-like Model as a Foundational Density Ratio Estimator".
- `clip_ig` Calculation of KL divergences, N-gram analysis
- `datacomp_filter` Data curation by KL divergences on DataComp codebase
- `iwl` Custom training code of OpenCLIP for Importance-Weighted Learning

In `clip_ig` and `datacomp_filter`, `MC`, `REVERSE_MC`, `CENTERIZED`, `WHITEN` are correspond to $D_\mathrm{KL}$, $D_\mathrm{KLR}$, $D_\mathrm{C}$ and $D_\mathrm{W}$, respectively.