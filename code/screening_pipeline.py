"""False-discovery-controlled screening of bimetallic alloy surfaces for hydrogen evolution.

Implements the frozen protocol (PROTOCOL_frozen.md): reference and structure-type
transfer designs, classical and simulated quantum-kernel predictors, conformal
selection (Jin & Candes 2023), weighted conformal selection with deterministic
pruning (Jin & Candes 2026), and comparison rules.

Usage: python3 screening_pipeline.py <audit_surfaces.csv> <out_dir> <n_reps> [subset]
  subset = "all" (default) or "min3" (surfaces with >= 3 evaluated configurations)
"""
import re
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import Element
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DELTAS = (0.10, 0.15, 0.20, 0.30)
ALPHAS = (0.1, 0.2, 0.3)
M_CLIP = 10.0
N_QUBITS = 6

# ----------------------------------------------------------------------------- features

PROPS = ("group", "period", "X", "radius", "ie1", "dgroup")


def element_props(sym):
    e = Element(sym)
    g = e.group
    return {"group": float(g), "period": float(e.row), "X": float(e.X),
            "radius": float(e.atomic_radius), "ie1": float(e.ionization_energies[0]),
            "dgroup": float(g) if 3 <= g <= 12 else 0.0}


def parse_formula(s):
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", s)
    comp = {}
    for el, n in parts:
        comp[el] = comp.get(el, 0) + (int(n) if n else 1)
    return comp


def build_features(df):
    cache = {}
    rows = []
    for s, st in zip(df["surface"], df["structure"]):
        comp = parse_formula(s)
        tot = sum(comp.values())
        els = sorted(comp, key=lambda e: -comp[e])
        fr = [comp[e] / tot for e in els]
        if len(els) == 1:
            els, fr = els * 2, [1.0, 0.0]
        pa = cache.setdefault(els[0], element_props(els[0]))
        pb = cache.setdefault(els[1], element_props(els[1]))
        f = {}
        for p in PROPS:
            f[f"mean_{p}"] = fr[0] * pa[p] + fr[1] * pb[p]
            f[f"diff_{p}"] = abs(pa[p] - pb[p])
        f["major_frac"] = fr[0]
        for t in ("A1", "L10", "L12"):
            f[f"is_{t}"] = float(st == t)
        rows.append(f)
    return pd.DataFrame(rows).to_numpy(dtype=float)

# ----------------------------------------------------------------------------- quantum kernels


def _hadamard_n(n):
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    out = np.array([[1.0]])
    for _ in range(n):
        out = np.kron(out, h)
    return out


HN = _hadamard_n(N_QUBITS)
# spins s_q = 1 - 2 z_q for every basis state; qubit 0 is the most significant bit
BITS = ((np.arange(2 ** N_QUBITS)[:, None] >> np.arange(N_QUBITS)[::-1]) & 1)
SPIN = 1 - 2 * BITS                                      # (64, n)
RING = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
SPIN_PAIR = np.stack([SPIN[:, a] * SPIN[:, b] for a, b in RING], axis=1)   # (64, n)


def statevectors(x):
    """x: (N, n_qubits) rotation angles -> (N, 64) complex states of the two-layer feature map."""
    pair = np.stack([x[:, a] * x[:, b] for a, b in RING], axis=1)
    phase = -0.5 * (x @ SPIN.T + pair @ SPIN_PAIR.T)       # (N, 64)
    d = np.exp(1j * phase)
    psi0 = np.zeros(2 ** N_QUBITS); psi0[0] = 1.0
    psi = d * (HN @ psi0)[None, :]
    psi = d * (psi @ HN.T)
    return psi


def bloch_vectors(psi):
    """Single-qubit Bloch vectors (N, n_qubits, 3) from (N, 64) states."""
    n = psi.shape[0]
    out = np.zeros((n, N_QUBITS, 3))
    t = psi.reshape(n, *([2] * N_QUBITS))
    for q in range(N_QUBITS):
        a = np.moveaxis(t, q + 1, 1).reshape(n, 2, -1)
        rho = np.einsum("nik,njk->nij", a, a.conj())
        out[:, q, 0] = 2 * rho[:, 0, 1].real
        out[:, q, 1] = -2 * rho[:, 0, 1].imag
        out[:, q, 2] = (rho[:, 0, 0] - rho[:, 1, 1]).real
    return out


def fidelity_kernel(pa, pb):
    return np.abs(pa @ pb.conj().T) ** 2


def projected_kernel(ba, bb, gamma=1.0):
    fa = ba.reshape(len(ba), -1); fb = bb.reshape(len(bb), -1)
    d2 = (fa ** 2).sum(1)[:, None] + (fb ** 2).sum(1)[None, :] - 2 * fa @ fb.T
    return np.exp(-gamma * 0.5 * np.maximum(d2, 0))


def geometric_difference(kq, kc, reg=1e-3):
    w, v = np.linalg.eigh(kq)
    sq = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    kc_inv = np.linalg.inv(kc + reg * np.eye(len(kc)))
    mat = sq @ kc_inv @ sq
    return float(np.sqrt(np.linalg.norm(mat, 2)))

# ----------------------------------------------------------------------------- models


def group_cv_mae(make_fit_predict, n_train, groups, k=3):
    gkf = GroupKFold(n_splits=k)
    err = []
    for tr, va in gkf.split(np.zeros(n_train), groups=groups):
        pred, ytrue = make_fit_predict(tr, va)
        err.append(np.mean(np.abs(pred - ytrue)))
    return float(np.mean(err))


class Predictors:
    """Fits every predictor on one training partition and predicts on the other units."""

    def __init__(self, X, y, groups, train_idx):
        self.X, self.y, self.g, self.tr = X, y, groups, train_idx
        self.sc = StandardScaler().fit(X[train_idx])
        self.Z = self.sc.transform(X)
        self.pca = PCA(n_components=N_QUBITS).fit(self.Z[train_idx])
        P = self.pca.transform(self.Z)
        self.scale = np.abs(P[train_idx]).max(axis=0) + 1e-12
        self.P = np.clip(P / self.scale, -1, 1)          # PCA components scaled to [-1, 1]
        self.times, self.info = {}, {}

    # generic tuner for precomputed-kernel ridge
    def _krr_precomputed(self, kfun, params_grid):
        tr = self.tr; y = self.y[tr]; g = self.g[tr]
        best = None
        for kp, alpha in params_grid:
            K = kfun(kp)
            Ktt = K[np.ix_(tr, tr)]
            def fp(a, b, Ktt=Ktt, alpha=alpha):
                m = KernelRidge(alpha=alpha, kernel="precomputed").fit(Ktt[np.ix_(a, a)], y[a])
                return m.predict(Ktt[np.ix_(b, a)]), y[b]
            mae = group_cv_mae(fp, len(tr), g)
            if best is None or mae < best[0]:
                best = (mae, kp, alpha, K)
        _, kp, alpha, K = best
        m = KernelRidge(alpha=alpha, kernel="precomputed").fit(K[np.ix_(tr, tr)], y)
        return m.predict(K[:, tr]), {"param": kp, "alpha": alpha}

    def fit_all(self):
        tr, y, g = self.tr, self.y[self.tr], self.g[self.tr]
        preds = {}

        t0 = time.time()                                   # ridge
        best = min(((group_cv_mae(lambda a, b, al=al: (Ridge(alpha=al).fit(self.Z[tr][a], y[a]).predict(self.Z[tr][b]), y[b]),
                                  len(tr), g), al) for al in (0.01, 0.1, 1, 10, 100)))
        preds["Ridge"] = Ridge(alpha=best[1]).fit(self.Z[tr], y).predict(self.Z)
        self.times["Ridge"] = time.time() - t0

        for name, F in (("KRR-RBF", self.Z), ("KRR-RBF-PCA6", self.P)):
            t0 = time.time()
            grid = [(gm, al) for gm in (0.01, 0.03, 0.1, 0.3, 1) for al in (1e-3, 1e-2, 1e-1, 1)]
            preds[name], self.info[name] = self._krr_precomputed(lambda gm, F=F: rbf_kernel(F, F, gamma=gm), grid)
            self.times[name] = time.time() - t0

        for name, F in (("GP", self.Z), ("GP-PCA6", self.P)):
            t0 = time.time()
            k = ConstantKernel(1.0) * RBF(length_scale=np.ones(F.shape[1])) + WhiteKernel(0.01)
            gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0, random_state=0)
            gp.fit(F[tr], y)
            preds[name] = gp.predict(F)
            self.times[name] = time.time() - t0

        t0 = time.time()                                   # boosted trees
        cands = [(ne, md) for ne in (200, 500) for md in (2, 3, 4)]
        best = min((group_cv_mae(lambda a, b, ne=ne, md=md: (
            HistGradientBoostingRegressor(max_iter=ne, max_depth=md, learning_rate=0.05, early_stopping=False,
                                          random_state=0).fit(self.Z[tr][a], y[a]).predict(self.Z[tr][b]), y[b]),
            len(tr), g), ne, md) for ne, md in cands)
        preds["GBT"] = HistGradientBoostingRegressor(max_iter=best[1], max_depth=best[2], learning_rate=0.05,
                                                     early_stopping=False, random_state=0).fit(self.Z[tr], y).predict(self.Z)
        self.times["GBT"] = time.time() - t0

        # quantum kernels: states for every bandwidth, computed once
        bws = (0.25, 0.5, 1.0, 2.0, 4.0)
        states = {b: statevectors(self.P * b) for b in bws}
        t0 = time.time()
        grid = [(b, al) for b in bws for al in (1e-3, 1e-2, 1e-1, 1)]
        preds["Q-Fidelity"], self.info["Q-Fidelity"] = self._krr_precomputed(
            lambda b: fidelity_kernel(states[b], states[b]), grid)
        self.times["Q-Fidelity"] = time.time() - t0
        t0 = time.time()
        blochs = {b: bloch_vectors(states[b]) for b in bws}
        preds["Q-Projected"], self.info["Q-Projected"] = self._krr_precomputed(
            lambda b: projected_kernel(blochs[b], blochs[b]), grid)
        self.times["Q-Projected"] = time.time() - t0

        # geometric difference on the training partition, against the tuned RBF on the PCA features
        gm = self.info["KRR-RBF-PCA6"]["param"]
        kc = rbf_kernel(self.P[tr], self.P[tr], gamma=gm)
        bf = self.info["Q-Fidelity"]["param"]; bp = self.info["Q-Projected"]["param"]
        self.info["gd_fidelity"] = geometric_difference(fidelity_kernel(states[bf][tr], states[bf][tr]), kc)
        self.info["gd_projected"] = geometric_difference(projected_kernel(blochs[bp][tr], blochs[bp][tr]), kc)
        return preds

# ----------------------------------------------------------------------------- selection rules


def bh(p, alpha):
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= alpha * np.arange(1, m + 1) / m
    if not ok.any():
        return np.zeros(m, bool)
    k = np.max(np.nonzero(ok)[0]) + 1
    sel = np.zeros(m, bool)
    sel[order[:k]] = True
    return sel


def conformal_pvalues(y_cal, mu_cal, mu_cand, c, rng):
    v_cal = M_CLIP * (y_cal >= c) + c * (y_cal < c) - mu_cal
    v_hat = c - mu_cand
    vs = np.sort(v_cal)
    less = np.searchsorted(vs, v_hat, side="left")
    leq = np.searchsorted(vs, v_hat, side="right")
    u = rng.uniform(size=len(v_hat))
    return (less + u * (1 + (leq - less))) / (len(v_cal) + 1)


def weighted_selection(y_cal, mu_cal, w_cal, mu_cand, w_cand, c, alpha):
    """Weighted conformal selection with deterministic pruning (Jin & Candes, Biometrika 2026)."""
    v_cal = M_CLIP * (y_cal >= c) + c * (y_cal < c) - mu_cal
    v_hat = c - mu_cand
    m = len(v_hat)
    order = np.argsort(v_cal)
    vs, cw = v_cal[order], np.concatenate([[0.0], np.cumsum(w_cal[order])])
    A = cw[np.searchsorted(vs, v_hat, side="left")]      # sum_i w_i 1{V_i < Vhat_l}
    S = w_cal.sum()
    p = (A + w_cand) / (S + w_cand)
    # auxiliary p-values, row j: p_l^(j) = (A_l + w_j 1{Vhat_j < Vhat_l}) / (S + w_j), p_j^(j) = 0
    P = (A[None, :] + w_cand[:, None] * (v_hat[:, None] < v_hat[None, :])) / (S + w_cand[:, None])
    np.fill_diagonal(P, 0.0)
    Ps = np.sort(P, axis=1)
    thr = alpha * np.arange(1, m + 1) / m
    ok = Ps <= thr[None, :]
    kstar = np.where(ok.any(1), m - np.argmax(ok[:, ::-1], axis=1), 0)
    rsize = (P <= (alpha * kstar / m)[:, None]).sum(1)
    rsize = np.maximum(rsize, 1)
    first = p <= alpha * rsize / m
    if not first.any():
        return np.zeros(m, bool)
    cand_r = np.sort(rsize[first])
    rstar = 0
    for r in range(1, len(cand_r) + 1):
        if (cand_r <= r).sum() >= r:
            rstar = r
    return first & (rsize <= rstar)


def interval_rule(g_cal, gh_cal, gh_cand, delta, alpha):
    n = len(g_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.zeros(len(gh_cand), bool)
    q = np.sort(np.abs(g_cal - gh_cal))[k - 1]
    return (gh_cand - q >= -delta) & (gh_cand + q <= delta)

# ----------------------------------------------------------------------------- designs


def split_reference(df, rng):
    systems = df["system"].unique()
    perm = rng.permutation(systems)
    n = len(perm)
    tr_s, ca_s = set(perm[: n // 2]), set(perm[n // 2: (3 * n) // 4])
    lab = np.array(["cand"] * len(df), dtype=object)
    lab[df["system"].isin(tr_s).to_numpy()] = "train"
    lab[df["system"].isin(ca_s).to_numpy()] = "cal"
    return lab


def split_transfer(df, held, rng):
    lab = np.array(["cand"] * len(df), dtype=object)
    src = (df["structure"] != held).to_numpy()
    systems = df.loc[src, "system"].unique()
    perm = rng.permutation(systems)
    ca_s = set(perm[: int(round(0.3 * len(perm)))])
    is_cal = df["system"].isin(ca_s).to_numpy() & src
    lab[src] = "train"
    lab[is_cal] = "cal"
    return lab


def evaluate(df, X, design, rep):
    rng = np.random.default_rng(10_000 * {"reference": 0, "transfer_L10": 1, "transfer_L12": 2}[design] + rep)
    if design == "reference":
        lab = split_reference(df, rng)
    else:
        lab = split_transfer(df, design.split("_")[1], rng)
    tr, ca, cd = (np.nonzero(lab == s)[0] for s in ("train", "cal", "cand"))
    g = df["dGH_eV"].to_numpy()
    groups = df["system"].to_numpy()
    model = Predictors(X, g, groups, tr)
    preds = model.fit_all()

    weights = None
    if design != "reference":
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(
            np.vstack([model.Z[ca], model.Z[cd]]), np.r_[np.zeros(len(ca)), np.ones(len(cd))])
        def w(idx):
            pr = clf.predict_proba(model.Z[idx])[:, 1]
            return np.clip(pr / (1 - pr) * len(ca) / len(cd), 0.05, 20)
        weights = (w(ca), w(cd))

    rows = []
    for name, gh in preds.items():
        mu = -np.abs(gh)
        mae = float(np.mean(np.abs(gh[cd] - g[cd])))
        for delta in DELTAS:
            c = -delta
            truth = np.abs(g[cd]) <= delta
            y_cal = -np.abs(g[ca])
            for alpha in ALPHAS:
                prng = np.random.default_rng(rep * 7919 + int(delta * 1000) + int(alpha * 10))
                p = conformal_pvalues(y_cal, mu[ca], mu[cd], c, prng)
                sel = {"conformal": bh(p, alpha)}
                k = int(sel["conformal"].sum())
                topk = np.zeros(len(cd), bool)
                if k > 0:
                    topk[np.argsort(-mu[cd])[:k]] = True
                sel["topk_matched"] = topk
                sel["threshold"] = np.abs(gh[cd]) <= delta
                sel["interval"] = interval_rule(g[ca], gh[ca], gh[cd], delta, alpha)
                if weights is not None:
                    sel["weighted_conformal"] = weighted_selection(y_cal, mu[ca], weights[0], mu[cd], weights[1], c, alpha)
                for rule, s in sel.items():
                    ns = int(s.sum()); tp = int((s & truth).sum())
                    rows.append({"design": design, "rep": rep, "model": name, "delta": delta, "alpha": alpha,
                                 "rule": rule, "n_selected": ns, "true_hits": tp,
                                 "fdp": (ns - tp) / max(1, ns), "n_candidates": len(cd),
                                 "n_true_in_pool": int(truth.sum()), "n_cal": len(ca), "n_train": len(tr),
                                 "mae": mae, "fit_time_s": model.times[name]})
    meta = {"design": design, "rep": rep, "gd_fidelity": model.info["gd_fidelity"],
            "gd_projected": model.info["gd_projected"],
            "q_fidelity_bandwidth": model.info["Q-Fidelity"]["param"],
            "q_projected_bandwidth": model.info["Q-Projected"]["param"]}
    if weights is not None:
        meta.update({"w_cand_min": float(weights[1].min()), "w_cand_max": float(weights[1].max()),
                     "w_cal_min": float(weights[0].min()), "w_cal_max": float(weights[0].max()),
                     "frac_w_clipped": float(np.mean((weights[1] <= 0.05) | (weights[1] >= 20)))})
    return rows, meta


def main(audit_csv, out_dir, n_reps, subset="all"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(audit_csv)
    if subset == "min3":
        df = df[df["n_sites"] >= 3].reset_index(drop=True)
    X = build_features(df)
    print(f"surfaces {len(df)}, features {X.shape[1]}, subset {subset}", flush=True)
    for design in ("reference", "transfer_L10", "transfer_L12"):
        res_f, meta_f = out / f"results_{subset}_{design}.csv", out / f"meta_{subset}_{design}.jsonl"
        done = set(pd.read_csv(res_f)["rep"].unique()) if res_f.exists() else set()
        for rep in range(int(n_reps)):
            if rep in done:
                continue
            t0 = time.time()
            rows, meta = evaluate(df, X, design, rep)
            pd.DataFrame(rows).to_csv(res_f, mode="a", header=not res_f.exists(), index=False)
            with open(meta_f, "a") as fh:
                fh.write(json.dumps(meta) + "\n")
            r = pd.DataFrame(rows)
            prim = r[(r.model == "GP") & (r.rule == "conformal") & (r.delta == 0.2) & (r.alpha == 0.2)].iloc[0]
            print(f"[{subset}] {design} rep {rep+1}/{n_reps}: {time.time()-t0:.0f} s; "
                  f"GP conformal |S|={prim.n_selected} hits={prim.true_hits} FDP={prim.fdp:.2f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
