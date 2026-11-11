# Oversampling rare failures: what SMOTE does to surrogate-based failure probabilities

What SMOTE oversampling does to surrogate-based failure probabilities in structural reliability, from a Singularity v3 forecast.

**Paper:** [paper.pdf](paper.pdf) · **Page:** [singularitysd.com/research/smote.html](https://singularitysd.com/research/smote.html) · Preprint, not yet peer reviewed.

## Abstract

Classifiers are increasingly used as surrogates of limit-state functions in structural reliability, and because failures are rare they are often trained on data rebalanced with the synthetic minority oversampling technique (SMOTE). Its effect on the estimated failure probability $P_f$ has not, to our knowledge, been measured. We compare, on three standard benchmarks (a four-branch series system, a non-linear oscillator and a ten-variable linear limit state), a classifier trained on plain Monte Carlo samples, class weighting, SMOTE, and SMOTE followed by a Bayes prior-shift correction, with gradient boosting and a neural network, 50 repetitions each and 2000 training points. The direction of the error depends on how many failures the training sample contains. With about nine failures (four-branch, $P_f=4.4\times10^{-3}$) SMOTE overestimates $P_f$ by a median factor of 11–13; the correction reduces this to 0.6–2.4. With about 57 failures (oscillator) all strategies are within a factor 0.7–1.2. With about three failures (linear, $P_f=1.35\times10^{-3}$) the plain classifier collapses to medians of 0.00–0.03 of the true value, SMOTE recovers 0.6–0.7 but with a 10–90% range of 0.12–1.96, and the correction collapses again. No strategy is safe across regimes: a thresholded classifier, rebalanced or not, should not be used as a failure-probability estimator without a check against the true limit state.

## How it was made

This study started from the Singularity forecast **T3v3-2026-0099**: a method predicted to enter a new field. It was then tested once, with the plan fixed before looking at the results.

## What is in this repository

| file | what it is |
|---|---|
| `experiment.py` | reproduces the study end to end |
| `fig_bias.png` | figure |
| `main.tex` | LaTeX source of the paper |
| `paper.pdf` | the paper |
| `results.json` | every number reported in the paper |

## Reproduce

```bash
pip install matplotlib numpy scikit-learn scipy
python experiment.py
```

It writes `results.json` and the figures next to the script.

## Cite

```bibtex
@misc{singularity_smote_rare_failure_trap_2026,
  title  = {Oversampling rare failures: what SMOTE does to surrogate-based failure probabilities},
  author = {Singularity},
  year   = {2026},
  url    = {https://github.com/singularity-sd/smote-rare-failure-trap}
}
```

## License

Code: MIT (`LICENSE`). Paper, text and figures: CC BY 4.0 (`LICENSE-paper.md`). Third-party data keep their own licence where stated.

---

Part of [Singularity](https://singularitysd.com): forecasting where science goes next, then testing it.
