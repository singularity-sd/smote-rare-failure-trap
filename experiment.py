"""Does SMOTE bias surrogate-based failure probabilities? A pre-specified benchmark.

From the Singularity v3 forecast in the view v3-NAI: synthetic minority oversampling technique -> Computational
Mechanics. In structural reliability, failures are rare, so classifiers used as surrogates of the limit-state
function are often trained on oversampled data. We measure what this does to the estimated failure probability, and
whether a one-line prior-shift correction repairs it.

Fixed before the single run: three limit-state functions, training size 2000, two classifiers, four strategies,
50 repetitions, 10^6 Monte Carlo points for the estimate, seed.
Run: python experiment.py -> results.json, fig_bias.png
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261105)
N_TRAIN, N_MC, REPS = 2000, 1_000_000, 50


def four_branch(x):           # series system with four branches (Waarts), k = 6
    a, b = x[:, 0], x[:, 1]
    return np.min(np.stack([3 + (a - b) ** 2 / 10 - (a + b) / np.sqrt(2), 3 + (a - b) ** 2 / 10 + (a + b) / np.sqrt(2),
                            (a - b) + 6 / np.sqrt(2), (b - a) + 6 / np.sqrt(2)]), axis=0)


def oscillator(x):            # non-linear undamped single-degree-of-freedom oscillator, standard benchmark
    m = 1 + 0.05 * x[:, 0]; c1 = 1 + 0.1 * x[:, 1]; c2 = 0.1 + 0.01 * x[:, 2]
    r = 0.5 + 0.05 * x[:, 3]; F1 = 1 + 0.2 * x[:, 4]; t1 = 1 + 0.2 * x[:, 5]
    w0 = np.sqrt((c1 + c2) / m)
    return 3 * r - np.abs(2 * F1 / (m * w0 ** 2) * np.sin(w0 * t1 / 2))


def linear10(x):              # g = beta * sqrt(n) - sum(x), beta = 3: Pf = Phi(-3) exactly
    return 3 * np.sqrt(x.shape[1]) - x.sum(axis=1)


PROBLEMS = {"four-branch (2 variables)": (four_branch, 2), "non-linear oscillator (6 variables)": (oscillator, 6),
            "linear, 10 variables": (linear10, 10)}


def smote(X, y, k=5):
    """Oversample the minority class to balance, by interpolation between minority neighbours."""
    Xm = X[y == 1]
    need = int((y == 0).sum() - (y == 1).sum())
    if need <= 0 or len(Xm) < 2:
        return X, y
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xm))).fit(Xm)
    _, idx = nn.kneighbors(Xm)
    i = RNG.integers(0, len(Xm), need)
    j = idx[i, RNG.integers(1, idx.shape[1], need)]
    lam = RNG.random((need, 1))
    Xs = Xm[i] + lam * (Xm[j] - Xm[i])
    return np.vstack([X, Xs]), np.concatenate([y, np.ones(need, dtype=int)])


def prior_correct(p, pi_train, pi_true):
    """Bayes prior-shift correction from a training prevalence to the true one."""
    a = p * pi_true / pi_train
    b = (1 - p) * (1 - pi_true) / (1 - pi_train)
    return a / (a + b)


def models():
    return {"gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=200, random_state=int(RNG.integers(1e9))),
            "neural network": lambda: MLPClassifier((32, 32), max_iter=600, random_state=int(RNG.integers(1e9)))}


def main():
    res = {"protocol": {"N_TRAIN": N_TRAIN, "N_MC": N_MC, "REPS": REPS}, "problems": {}}
    for pname, (g, d) in PROBLEMS.items():
        Xmc = RNG.standard_normal((N_MC, d))
        pf_true = float((g(Xmc) <= 0).mean()) if pname != "linear, 10 variables" else float(stats.norm.cdf(-3))
        out = {"pf_true": pf_true}
        for mname, mk in models().items():
            est = {s: [] for s in ["plain", "SMOTE", "SMOTE + prior correction", "class weights"]}
            rec = {s: [] for s in ["plain", "SMOTE"]}
            for _ in range(REPS):
                X = RNG.standard_normal((N_TRAIN, d))
                y = (g(X) <= 0).astype(int)
                if y.sum() < 2:
                    continue
                Xt = RNG.standard_normal((200_000, d))
                yt = g(Xt) <= 0
                m = mk().fit(X, y)
                p_plain = m.predict_proba(Xt)[:, 1]
                est["plain"].append((p_plain > 0.5).mean())
                Xs, ys = smote(X, y)
                ms = mk().fit(Xs, ys)
                p_s = ms.predict_proba(Xt)[:, 1]
                est["SMOTE"].append((p_s > 0.5).mean())
                pi_train = ys.mean()
                pi_hat = y.mean()            # the only prevalence the analyst knows: the one in the training sample
                est["SMOTE + prior correction"].append((prior_correct(p_s, pi_train, pi_hat) > 0.5).mean())
                if mname == "gradient boosting":
                    w = np.where(y == 1, (y == 0).sum() / max(1, y.sum()), 1.0)
                    mw = mk().fit(X, y, sample_weight=w)
                    est["class weights"].append((mw.predict_proba(Xt)[:, 1] > 0.5).mean())
                rec["plain"].append(float(((p_plain > 0.5) & yt).sum() / max(1, yt.sum())))
                rec["SMOTE"].append(float(((p_s > 0.5) & yt).sum() / max(1, yt.sum())))
            out[mname] = {s: {"ratio_median": float(np.median(np.array(v) / pf_true)), "ratio_q10": float(np.quantile(np.array(v) / pf_true, 0.1)),
                              "ratio_q90": float(np.quantile(np.array(v) / pf_true, 0.9)), "n": len(v)} for s, v in est.items() if v}
            out[mname]["recall"] = {s: float(np.mean(v)) for s, v in rec.items()}
            print(pname, mname, {s: round(x["ratio_median"], 2) for s, x in out[mname].items() if s != "recall"}, out[mname]["recall"], flush=True)
        res["problems"][pname] = out
    (HERE / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    figure(res)


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    strat = ["plain", "class weights", "SMOTE", "SMOTE + prior correction"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (pname, out) in zip(axs, res["problems"].items()):
        for k, (mname, c) in enumerate([("gradient boosting", "#1f77b4"), ("neural network", "#d62728")]):
            d = out[mname]
            xs = [i + (k - 0.5) * 0.25 for i, s in enumerate(strat) if s in d]
            med = [d[s]["ratio_median"] for s in strat if s in d]
            lo = [d[s]["ratio_median"] - d[s]["ratio_q10"] for s in strat if s in d]
            hi = [d[s]["ratio_q90"] - d[s]["ratio_median"] for s in strat if s in d]
            ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", color=c, label=mname, capsize=3)
        ax.axhline(1, color="0.4", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(strat)))
        ax.set_xticklabels(["plain", "class\nweights", "SMOTE", "SMOTE +\nprior corr."], fontsize=8)
        ax.set_title(f"{pname}\n$P_f$ = {out['pf_true']:.2e}", fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("estimated / true failure probability")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_bias.png", dpi=160)


if __name__ == "__main__":
    main()
