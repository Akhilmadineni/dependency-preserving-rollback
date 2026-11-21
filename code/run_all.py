#!/usr/bin/env python3
"""A5P2 - Tabu search for minimal dependency-preserving rollback sets (DERMS control/data plane).

Runs everything: preprocessing, fault-probability model (LightGBM, grouped 5-fold CV, 5 seeds),
dependency-aware must-rollback propagation, Tabu search, baselines (full rollback, naive component
rollback, greedy closure), exact reference, ablation, sensitivity, scalability, statistics and figures.
Usage: python3 run_all.py      (reads ../data, writes ../results and ../figures)
"""
import os, json, time, itertools, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
W_U, W_M, F_PLANE = 1.0, 2.0, 0.3          # unnecessary-rollback cost, missed-fault cost, plane fixed charge
TABU = dict(tenure=3, gamma0=2.0, max_iter=40, no_improve=15)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.3})
COL = {"Full rollback": "#9e9e9e", "Naive component": "#bdbdbd", "Greedy closure": "#616161", "Tabu (proposed)": "#1f5fa8"}

# ------------------------------------------------------------------ 1. load + preprocess
nodes = pd.read_csv(os.path.join(DATA, "A5P2_rollback_components.csv"))
edges = pd.read_csv(os.path.join(DATA, "A5P2_rollback_edges.csv"))
LABELS = ["root_cause", "must_rollback"]           # never used as features
summary = {"n_deployments": int(nodes.deployment_id.nunique()), "n_component_rows": len(nodes), "n_edge_rows": len(edges),
           "n_changed": int(nodes.changed_in_release.sum()), "n_root_cause": int(nodes.root_cause.sum()),
           "n_must_rollback": int(nodes.must_rollback.sum()),
           "missing_pct": {c: round(100 * nodes[c].isna().mean(), 2) for c in ["test_coverage", "p99_latency_delta_ms", "telemetry_gap_pct"]}}

# graph features from the edge table (observable post-rollout signals only; coupled_rollback_required excluded)
asc = edges.rename(columns={"consumer_component": "component"}).groupby(["deployment_id", "component"]).agg(
    own_sym_c=("consumer_symptom_score", "mean"), max_prov_sym=("provider_symptom_score", "max"),
    mean_prov_sym=("provider_symptom_score", "mean"),
    n_incompat_prov=("backward_compatible_declared", lambda s: int((1 - s).sum()))).reset_index()
asp = edges.rename(columns={"provider_component": "component"}).groupby(["deployment_id", "component"]).agg(
    own_sym_p=("provider_symptom_score", "mean"), max_cons_sym=("consumer_symptom_score", "max"),
    mean_cons_sym=("consumer_symptom_score", "mean")).reset_index()
df = nodes.merge(asc, on=["deployment_id", "component"], how="left").merge(asp, on=["deployment_id", "component"], how="left")
df["own_sym"] = df[["own_sym_c", "own_sym_p"]].mean(axis=1)
df["excess_sym"] = df["own_sym"] - df["max_prov_sym"].fillna(0)        # symptom not explained by providers
df["sym_vs_cons"] = df["own_sym"] - df["max_cons_sym"].fillna(0)
df["log_loc"] = np.log1p(df["lines_changed"])
df["is_data"] = (df["plane"] == "data").astype(int)
df["comp_code"] = df["component"].astype("category").cat.codes
df["strat_code"] = df["rollout_strategy"].astype("category").cat.codes
BASE_F = ["log_loc", "schema_migration", "api_contract_change", "test_coverage", "bake_time_min", "n_upstream_deps",
          "n_downstream_consumers", "n_changed_providers", "n_providers_with_contract_change", "error_rate_delta_pct",
          "p99_latency_delta_ms", "restart_count", "der_command_ack_rate_pct", "telemetry_gap_pct", "config_drift_flag",
          "is_data", "comp_code", "strat_code"]
GRAPH_F = ["own_sym", "max_prov_sym", "mean_prov_sym", "max_cons_sym", "mean_cons_sym", "n_incompat_prov", "excess_sym", "sym_vs_cons"]
ch = df[df.changed_in_release == 1].reset_index(drop=True)          # decision variables = changed components
deps_ids = ch.deployment_id.unique()

def preprocess(tr, te, feats):
    """Train-fold median imputation + 1/99 percentile winsorising of heavy-tailed metrics."""
    tr, te = tr[feats].copy(), te[feats].copy()
    for c in feats:
        med = tr[c].median()
        tr[c] = tr[c].fillna(med); te[c] = te[c].fillna(med)
        if c in ("p99_latency_delta_ms", "error_rate_delta_pct", "telemetry_gap_pct"):
            lo, hi = tr[c].quantile([0.01, 0.99]); tr[c] = tr[c].clip(lo, hi); te[c] = te[c].clip(lo, hi)
    return tr, te

def oof_probs(seed, feats):
    p = np.zeros(len(ch)); fold = np.zeros(len(ch), int); imp = np.zeros(len(feats))
    gkf = GroupKFold(n_splits=5)
    rng = np.random.default_rng(seed); perm = {d: i for i, d in enumerate(rng.permutation(deps_ids))}
    groups = ch.deployment_id.map(perm).values
    for k, (tri, tei) in enumerate(gkf.split(ch, groups=groups)):
        Xtr, Xte = preprocess(ch.iloc[tri], ch.iloc[tei], feats)
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               random_state=seed, num_threads=1, verbose=-1)
        m.fit(Xtr, ch.root_cause.values[tri]); p[tei] = m.predict_proba(Xte)[:, 1]; fold[tei] = k
        imp += m.booster_.feature_importance("gain")
    return p, fold, imp / imp.sum()

# ------------------------------------------------------------------ 2. per-deployment instances
ed_ch = edges[(edges.consumer_changed == 1) & (edges.provider_changed == 1)]
schema_of = dict(zip(zip(ch.deployment_id, ch.component), ch.schema_migration))
inst_edges = {}
for d, g in ed_ch.groupby("deployment_id"):
    con = [(c, p) for c, p, k in zip(g.consumer_component, g.provider_component, g.provider_contract_change) if k == 1]
    sch = [(c, p) for c, p, t in zip(g.consumer_component, g.provider_component, g.dependency_type)
           if t == "schema" and schema_of.get((d, c), 0) == 1]
    inst_edges[d] = (con, sch)

def implication(n, con, sch):
    """A[u, v] = 1 means x_u = 1 forces x_v = 1 (Eq. 4-5); returns transitive closure R."""
    A = np.eye(n, dtype=bool)
    for c, p in con: A[p, c] = True            # provider rolled back -> consumer on new contract rolls back
    for c, p in sch: A[c, p] = True            # consumer with schema migration rolled back -> data store rolls back
    R = A.copy()
    for k in range(n): R = R | (R[:, [k]] & R[[k], :])
    return R

def build_instances(prob):
    out = []
    for d, g in ch.assign(prob=prob).groupby("deployment_id", sort=False):
        names = list(g.component); ix = {c: i for i, c in enumerate(names)}
        con, sch = inst_edges.get(d, ([], []))
        con = [(ix[c], ix[p]) for c, p in con]; sch = [(ix[c], ix[p]) for c, p in sch]
        out.append(dict(d=d, n=len(names), p=g.prob.values, data=g.is_data.values.astype(bool), con=con, sch=sch,
                        R=implication(len(names), con, sch), y=g.must_rollback.values, rc=g.root_cause.values))
    return out

def estimate_rho(train_mask_deps):
    """P(changed consumer must roll back | contract provider must roll back), from training deployments only."""
    sub = ed_ch[ed_ch.deployment_id.isin(train_mask_deps) & (ed_ch.provider_contract_change == 1)]
    mr = dict(zip(zip(ch.deployment_id, ch.component), ch.must_rollback)); rc = dict(zip(zip(ch.deployment_id, ch.component), ch.root_cause))
    num = den = 0
    for d, c, p in zip(sub.deployment_id, sub.consumer_component, sub.provider_component):
        if mr[(d, p)] and not rc[(d, c)]: den += 1; num += mr[(d, c)]
    return num / max(den, 1)

def propagate(I, rho, iters=3):
    """Noisy-OR dependency propagation of must-rollback probability (Eq. 2)."""
    m = I["p"].copy()
    for _ in range(iters):
        q = 1 - I["p"]
        for c, pr in I["con"]: q = q * np.where(np.arange(I["n"]) == c, 1 - rho * m[pr], 1)
        for c, pr in I["sch"]: q = q * np.where(np.arange(I["n"]) == pr, 1 - m[c], 1)
        m = 1 - q
    return m

# ------------------------------------------------------------------ 3. objective and solvers
def J(X, I, m, gamma, F=F_PLANE, wm=W_M):
    """Expected rollback cost (Eq. 3) + plane fixed charge + gamma * dependency violations (Eq. 6). X: (M, n)."""
    X = np.atleast_2d(X).astype(float)
    base = X @ (W_U * (1 - m)) + (1 - X) @ (wm * m)
    fix = F * ((X[:, I["data"]].sum(1) > 0).astype(float) + (X[:, ~I["data"]].sum(1) > 0).astype(float))
    return base + fix + gamma * viol(X, I)

def viol(X, I):
    X = np.atleast_2d(X); v = np.zeros(len(X))
    for c, p in I["con"]: v += X[:, p] * (1 - X[:, c])
    for c, p in I["sch"]: v += X[:, c] * (1 - X[:, p])
    return v

def tabu(I, m, tenure=3, gamma0=2.0, max_iter=40, no_improve=15, closure_moves=True, hard=True,
         F=F_PLANE, wm=W_M, trace=False, init=None):
    n, R = I["n"], I["R"]
    x = (m > W_U / (W_U + wm)).astype(float) if init is None else init.astype(float).copy()
    gamma = gamma0 if hard else 0.0
    best_x, best_J = None, np.inf
    def consider(cand):
        nonlocal best_x, best_J
        if (not hard) or viol(cand, I)[0] == 0:
            v = J(cand, I, m, 0.0, F, wm)[0]
            if v < best_J - 1e-12: best_J, best_x = v, cand.copy(); return True
        return False
    consider(x); consider(np.zeros(n))
    tabu_until = np.zeros(n); stall = 0; inf_streak = feas_streak = 0; hist = [best_J]
    for it in range(1, max_iter + 1):
        moves, prim = [], []
        for c in range(n):
            y = x.copy(); y[c] = 1 - y[c]; moves.append(y); prim.append(c)
            if closure_moves:
                if x[c] == 0: y = np.maximum(x, R[c].astype(float))          # add c with its forward closure
                else: y = x * (1 - R[:, c].astype(float))                      # drop c with everything forcing it
                if not np.array_equal(y, moves[-1]): moves.append(y); prim.append(c)
        Mv = np.array(moves); vals = J(Mv, I, m, gamma, F, wm)
        feas = (viol(Mv, I) == 0) if hard else np.ones(len(Mv), bool)
        pureJ = J(Mv, I, m, 0.0, F, wm)
        allowed = np.array([tabu_until[c] < it for c in prim]) | (feas & (pureJ < best_J - 1e-12))   # aspiration
        if not allowed.any(): allowed[:] = True
        k = int(np.argmin(np.where(allowed, vals, np.inf)))
        changed = np.where(Mv[k] != x)[0]; x = Mv[k]
        tabu_until[changed] = it + tenure
        if consider(x): stall = 0
        else: stall += 1
        if hard:                                                                 # strategic oscillation of gamma
            if feas[k]: feas_streak += 1; inf_streak = 0
            else: inf_streak += 1; feas_streak = 0
            if inf_streak >= 2: gamma *= 2.0; inf_streak = 0
            if feas_streak >= 3: gamma = max(gamma0, gamma / 2.0); feas_streak = 0
        hist.append(best_J)
        if stall >= no_improve: break
    if trace:
        hist += [best_J] * (max_iter + 1 - len(hist))
        return best_x, best_J, hist
    return best_x, best_J

def exact_enum(I, m, F=F_PLANE, wm=W_M):
    n = I["n"]; X = np.array(list(itertools.product([0, 1], repeat=n)), float)
    v = J(X, I, m, 0.0, F, wm); v[viol(X, I) > 0] = np.inf; k = int(np.argmin(v)); return X[k], v[k]

def exact_milp(I, m, F=F_PLANE, wm=W_M):
    n = I["n"]; a = W_U * (1 - m) - wm * m                                  # x_c coefficient; constant = wm*sum(m)
    c = np.concatenate([a, [F, F]]); rows, lb = [], []
    for cc, p in I["con"]: r = np.zeros(n + 2); r[cc] = 1; r[p] = -1; rows.append(r)
    for cc, p in I["sch"]: r = np.zeros(n + 2); r[p] = 1; r[cc] = -1; rows.append(r)
    for i in range(n):
        r = np.zeros(n + 2); r[n + (0 if I["data"][i] else 1)] = 1; r[i] = -1; rows.append(r)
    res = milp(c, constraints=LinearConstraint(np.array(rows), 0, np.inf), integrality=np.ones(n + 2),
               bounds=Bounds(0, 1))
    return res.x[:n].round(), res.fun + wm * m.sum()

def naive(I, p, wm=W_M): return (p > W_U / (W_U + wm)).astype(float)
def greedy_closure(I, p, wm=W_M):
    s = naive(I, p, wm); return (I["R"][s.astype(bool)].any(0)).astype(float) if s.any() else s
def full(I, p, wm=W_M): return np.ones(I["n"])

# ------------------------------------------------------------------ 4. evaluation
def evaluate(sets, insts, runtime=None):
    tp = fp = fn = 0; exact = health = 0; vio = 0; size = 0; per_dep_cost = []; planes = 0
    for x, I in zip(sets, insts):
        y = I["y"]; tp += int((x * y).sum()); fp += int((x * (1 - y)).sum()); fn += int(((1 - x) * y).sum())
        exact += int(np.array_equal(x, y)); health += int(((1 - x) * I["rc"]).sum() == 0)
        v = viol(x, I)[0]; vio += v; size += x.sum()
        per_dep_cost.append(W_U * (x * (1 - y)).sum() + W_M * ((1 - x) * y).sum())
        planes += int((x * I["data"]).sum() > 0) + int((x * ~I["data"]).sum() > 0)
    N = len(insts); P = tp / max(tp + fp, 1); Rc = tp / max(tp + fn, 1)
    out = dict(precision=P, recall=Rc, f1=2 * P * Rc / max(P + Rc, 1e-9), exact_match=100 * exact / N,
               fp_per_dep=fp / N, fn_per_dep=fn / N, viol_per_dep=vio / N, health_restored=100 * health / N,
               size_per_dep=size / N, realised_cost=float(np.mean(per_dep_cost)), planes_per_dep=planes / N,
               ops_cost=float(np.mean(per_dep_cost)) + F_PLANE * planes / N)
    if runtime is not None: out["runtime_ms"] = 1000 * runtime / N
    return out, np.array(per_dep_cost)

def run_methods(insts, rho, which=("Full rollback", "Naive component", "Greedy closure", "Tabu (proposed)"), wm=W_M, **tk):
    res, costs, sets_all = {}, {}, {}
    for name in which:
        t0 = time.perf_counter(); sets = []
        for I in insts:
            if name == "Full rollback": sets.append(full(I, I["p"], wm))
            elif name == "Naive component": sets.append(naive(I, I["p"], wm))
            elif name == "Greedy closure": sets.append(greedy_closure(I, I["p"], wm))
            elif name == "Tabu (proposed)": sets.append(tabu(I, I["m"], wm=wm, **{**TABU, **tk})[0])
            elif name == "Exact reference": sets.append(exact_enum(I, I["m"], wm=wm)[0])
        res[name], costs[name] = evaluate(sets, insts, time.perf_counter() - t0); sets_all[name] = sets
    return res, costs, sets_all

t_start = time.time()
REGIMES = {"G": BASE_F + GRAPH_F, "L": BASE_F}      # G: graph-aware evidence, L: component-local evidence only
NAMES = ("Full rollback", "Naive component", "Greedy closure", "Tabu (proposed)", "Exact reference")
order = list(NAMES)
cache, all_rows, auc_rows, imps, rho_vals = {}, [], [], [], []
for reg, feats in REGIMES.items():
    for seed in SEEDS:
        p, fold, imp = oof_probs(seed, feats)
        if reg == "G": imps.append(imp)
        auc_rows.append(dict(regime=reg, seed=seed, auc=roc_auc_score(ch.root_cause, p), ap=average_precision_score(ch.root_cause, p)))
        insts = build_instances(p); fold_of = dict(zip(ch.deployment_id, fold))
        rho_f = {k: estimate_rho(set(ch.deployment_id[fold != k])) for k in range(5)}; rho_vals += list(rho_f.values())
        for I in insts: I["fold"] = fold_of[I["d"]]; I["m"] = propagate(I, rho_f[I["fold"]])
        cache[(reg, seed)] = insts
        res, costs, sets = run_methods(insts, None, NAMES)
        for k, v in res.items(): all_rows.append(dict(regime=reg, seed=seed, method=k, **v))
        if seed == 0:
            cache[(reg, "costs")] = costs; cache[(reg, "sets")] = sets
            cache[(reg, "ops")] = {k: np.array([costs[k][i] + F_PLANE * (int((x * I["data"]).sum() > 0) + int((x * ~I["data"]).sum() > 0))
                                                for i, (x, I) in enumerate(zip(sets[k], insts))]) for k in sets}
            gaps = np.array([J(xt, I, I["m"], 0.0)[0] - exact_enum(I, I["m"])[1] for I, xt in zip(insts, sets["Tabu (proposed)"])])
            cache[(reg, "gaps")] = gaps
        print(f"regime {reg} seed {seed} done  {time.time() - t_start:.0f}s", flush=True)

main = pd.DataFrame(all_rows)
main.round(5).to_csv(os.path.join(RES, "main_comparison_per_seed.csv"), index=False)
agg = main.drop(columns="seed").groupby(["regime", "method"], sort=False).agg(["mean", "std"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]; agg.round(4).to_csv(os.path.join(RES, "main_comparison.csv"))
auc = pd.DataFrame(auc_rows); auc.round(4).to_csv(os.path.join(RES, "fault_model_auc.csv"), index=False)

stat = {}
for reg in REGIMES:
    cs = cache[(reg, "costs")]; stat[reg] = {}
    for b in ["Full rollback", "Naive component", "Greedy closure"]:
        d = cs[b] - cs["Tabu (proposed)"]
        try: pv = float(wilcoxon(cs["Tabu (proposed)"], cs[b]).pvalue)
        except ValueError: pv = float("nan")
        stat[reg][b] = dict(mean_cost_diff=float(d.mean()), tabu_better=int((d > 0).sum()), tabu_worse=int((d < 0).sum()), wilcoxon_p=pv)
        do = cache[(reg, "ops")][b] - cache[(reg, "ops")]["Tabu (proposed)"]
        try: pvo = float(wilcoxon(cache[(reg, "ops")]["Tabu (proposed)"], cache[(reg, "ops")][b]).pvalue)
        except ValueError: pvo = float("nan")
        stat[reg][b].update(ops_mean_diff=float(do.mean()), ops_tabu_better=int((do > 1e-12).sum()), ops_tabu_worse=int((do < -1e-12).sum()), ops_wilcoxon_p=pvo)
    # seed-level paired comparison of realised cost (5 seeds)
    t = main[(main.regime == reg)].pivot(index="seed", columns="method", values="realised_cost")
    stat[reg]["seed_level_cost_diff_vs_greedy"] = [float(v) for v in (t["Greedy closure"] - t["Tabu (proposed)"]).values]
    t = main[(main.regime == reg)].pivot(index="seed", columns="method", values="ops_cost")
    stat[reg]["seed_level_ops_diff_vs_greedy"] = [float(v) for v in (t["Greedy closure"] - t["Tabu (proposed)"]).values]

plane_rows = []
for reg in REGIMES:
    for name in ["Naive component", "Greedy closure", "Tabu (proposed)"]:
        for pl, flag in (("control", False), ("data", True)):
            tp = fp = fn = 0
            for x, I in zip(cache[(reg, "sets")][name], cache[(reg, 0)]):
                s_ = I["data"] == flag; y = I["y"][s_]; xx = x[s_]
                tp += (xx * y).sum(); fp += (xx * (1 - y)).sum(); fn += ((1 - xx) * y).sum()
            P = tp / max(tp + fp, 1); Rr = tp / max(tp + fn, 1)
            plane_rows.append(dict(regime=reg, method=name, plane=pl, precision=P, recall=Rr, f1=2 * P * Rr / max(P + Rr, 1e-9)))
pd.DataFrame(plane_rows).round(4).to_csv(os.path.join(RES, "plane_breakdown.csv"), index=False)

# ------------------------------------------------------------------ 5. ablation (both regimes, all seeds)
abl_rows = []
variants = {
    "Full Tabu": lambda I: tabu(I, I["m"], **TABU)[0],
    "w/o probability propagation (p instead of m)": lambda I: tabu(I, I["p"], **TABU)[0],
    "w/o closure moves (single flips)": lambda I: tabu(I, I["m"], closure_moves=False, **TABU)[0],
    "w/o hard dependency constraints": lambda I: tabu(I, I["m"], hard=False, **TABU)[0],
    "w/o plane fixed charge (F=0)": lambda I: tabu(I, I["m"], F=0.0, **TABU)[0],
}
for reg in REGIMES:
    for seed in SEEDS:
        insts = cache[(reg, seed)]
        for k, f in variants.items():
            r, _ = evaluate([f(I) for I in insts], insts); abl_rows.append(dict(regime=reg, seed=seed, variant=k, **r))
    print(f"ablation {reg} done  {time.time() - t_start:.0f}s", flush=True)
abl = pd.DataFrame(abl_rows).drop(columns="seed").groupby(["regime", "variant"], sort=False).agg(["mean", "std"])
abl.columns = [f"{a}_{b}" for a, b in abl.columns]; abl.round(4).to_csv(os.path.join(RES, "ablation.csv"))

# ------------------------------------------------------------------ 6. sensitivity to miss cost w_M (seed 0, both regimes)
sens_rows = []
for reg in REGIMES:
    for wm in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        res, _, _ = run_methods(cache[(reg, 0)], None, ("Naive component", "Greedy closure", "Tabu (proposed)"), wm=wm)
        for k, v in res.items(): sens_rows.append(dict(regime=reg, w_M=wm, method=k, **v))
sens = pd.DataFrame(sens_rows); sens.round(4).to_csv(os.path.join(RES, "sensitivity_wM.csv"), index=False)

# ------------------------------------------------------------------ 7. scalability + Tabu parameter sensitivity
# instances use the (uncertain) local-evidence probabilities so that the optimisation is non-trivial
pool_p = np.concatenate([I["p"] for I in cache[("L", 0)]]); rho_bar = float(np.mean(rho_vals))
def rand_instance(n, rng):
    data = rng.random(n) < 0.5; con, sch = [], []
    for c in range(1, n):
        for p in rng.choice(c, size=min(c, rng.poisson(1.4)), replace=False):
            t = rng.choice(["api", "message", "schema", "config"], p=[0.45, 0.32, 0.09, 0.14])
            if rng.random() < 0.25: con.append((c, p))
            if t == "schema" and rng.random() < 0.3: sch.append((c, p))
    p = rng.choice(pool_p, n)
    I = dict(n=n, p=p, data=data, con=con, sch=sch, R=implication(n, con, sch)); I["m"] = propagate(I, rho_bar); return I

scal_rows, conv = [], {}
rng = np.random.default_rng(7)
for n in [16, 32, 64, 128, 256]:
    insts_s = [rand_instance(n, rng) for _ in range(10)]
    for i, I in enumerate(insts_s):
        t0 = time.perf_counter(); xm, jm = exact_milp(I, I["m"]); tm = time.perf_counter() - t0
        it = max(40, 2 * n)
        t0 = time.perf_counter(); xt, jt, h = tabu(I, I["m"], tenure=3, gamma0=2.0, max_iter=it, no_improve=it, trace=True); tt = time.perf_counter() - t0
        jg = J(greedy_closure(I, I["p"]), I, I["m"], 0.0)[0]; jn = J(full(I, I["p"]), I, I["m"], 0.0)[0]
        scal_rows.append(dict(n=n, inst=i, J_milp=jm, J_tabu=jt, J_greedy=jg, J_full=jn, gap_tabu_pct=100 * (jt - jm) / abs(jm),
                              gap_greedy_pct=100 * (jg - jm) / abs(jm), t_milp_ms=1000 * tm, t_tabu_ms=1000 * tt))
        gtrace = 100 * (np.array(h[:it + 1]) - jm) / abs(jm); conv.setdefault(n, []).append(gtrace)
        scal_rows[-1]["iters_to_opt"] = int(np.argmax(gtrace < 1e-7)) if (gtrace < 1e-7).any() else -1
    print(f"scal n={n} done  {time.time() - t_start:.0f}s", flush=True)
scal = pd.DataFrame(scal_rows); scal.round(4).to_csv(os.path.join(RES, "scalability.csv"), index=False)
scal_agg = scal.drop(columns="inst").groupby("n").agg(["mean", "std"]); scal_agg.columns = [f"{a}_{b}" for a, b in scal_agg.columns]
scal_agg.round(4).to_csv(os.path.join(RES, "scalability_summary.csv"))

rng = np.random.default_rng(11); hard_set = [rand_instance(128, rng) for _ in range(8)]
opt = [exact_milp(I, I["m"])[1] for I in hard_set]
grid_closure = float(np.mean([100 * (tabu(I, I["m"], tenure=3, gamma0=2.0, max_iter=30, no_improve=30, init=np.zeros(I["n"]))[1] - o) / abs(o)
                               for I, o in zip(hard_set, opt)]))
grid = []
for ten in [0, 1, 3, 5, 8, 12]:
    for g0 in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        gp = [100 * (tabu(I, I["m"], tenure=ten, gamma0=g0, max_iter=30, no_improve=30, closure_moves=False, init=np.zeros(I["n"]))[1] - o) / abs(o)
              for I, o in zip(hard_set, opt)]
        grid.append(dict(tenure=ten, gamma0=g0, gap_pct=float(np.mean(gp))))
grid = pd.DataFrame(grid); grid.round(4).to_csv(os.path.join(RES, "tabu_param_sensitivity.csv"), index=False)
print(f"grid done  {time.time() - t_start:.0f}s", flush=True)

# ------------------------------------------------------------------ 8. figures
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RES, name + ".jpg"), dpi=300, bbox_inches="tight"); plt.close(fig)

meths = ["Full rollback", "Naive component", "Greedy closure", "Tabu (proposed)"]
short = ["Full", "Naive", "Greedy", "Tabu"]
fig, ax = plt.subplots(2, 2, figsize=(3.45, 3.1))
for a, (met, lab) in zip(ax.ravel(), [("f1", "Component F1"), ("exact_match", "Exact set match (%)"),
                                      ("ops_cost", "Operational cost / release"), ("health_restored", "Health restored (%)")]):
    w = 0.38
    for j, reg in enumerate(["G", "L"]):
        mu = [agg.loc[(reg, m_), met + "_mean"] for m_ in meths]; sd = [agg.loc[(reg, m_), met + "_std"] for m_ in meths]
        a.bar(np.arange(4) + (j - 0.5) * w, mu, w, yerr=sd, color=[COL[m_] for m_ in meths], capsize=1.5, edgecolor="k",
              linewidth=0.4, hatch=None if reg == "G" else "////", alpha=1.0 if reg == "G" else 0.75)
    a.set_xticks(range(4)); a.set_xticklabels(short, fontsize=7, rotation=25); a.set_ylabel(lab, fontsize=7)
    if met == "health_restored": a.set_ylim(75, 101)
    elif met == "ops_cost": a.set_yscale("log"); a.set_ylim(0.08, 12)
    else: a.set_ylim(0, a.get_ylim()[1] * 1.15)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor="#bbbbbb", edgecolor="k", label="G: graph-aware evidence"),
                    Patch(facecolor="#bbbbbb", edgecolor="k", hatch="////", label="L: local evidence only")],
           loc="upper center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(pad=0.3); save(fig, "fig3_comparison")

fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=False)
for a, reg in zip(axs, ["G", "L"]):
    for m_, mk in zip(["Naive component", "Greedy closure", "Tabu (proposed)"], ["s", "^", "o"]):
        s_ = sens[(sens.method == m_) & (sens.regime == reg)]
        a.plot(s_.fp_per_dep, s_.fn_per_dep, marker=mk, color=COL[m_] if m_ != "Naive component" else "#8a8a8a", ms=3, lw=1.0, label=m_.replace(" (proposed)", ""))
    a.set_title(f"Regime {reg}", fontsize=7, pad=2); a.set_xlabel("Unnecessary / release", fontsize=7)
axs[0].set_ylabel("Missed must-rollbacks / release", fontsize=7); axs[1].legend(frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.3, w_pad=0.5); save(fig, "fig4_tradeoff")

fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.9))
ns = sorted(scal.n.unique()); ga = scal.groupby("n")
axs[0].errorbar(ns, ga.gap_greedy_pct.mean(), yerr=ga.gap_greedy_pct.std(), color=COL["Greedy closure"], marker="^", ms=3.5, lw=1.1, capsize=2, label="Greedy closure")
axs[0].plot(ns, ga.gap_tabu_pct.mean(), color=COL["Tabu (proposed)"], marker="o", ms=3.5, lw=1.3, label="Tabu")
axs[0].set_xscale("log", base=2); axs[0].set_xlabel("Changed components n"); axs[0].set_ylabel("Gap to MILP optimum (%)"); axs[0].legend(frameon=False, fontsize=6.5)
axs[1].plot(ns, ga.t_tabu_ms.mean(), color=COL["Tabu (proposed)"], marker="o", ms=3.5, lw=1.3, label="Tabu (Python)")
axs[1].plot(ns, ga.t_milp_ms.mean(), color="k", marker="s", ms=3.5, lw=1.1, ls="--", label="MILP (HiGHS)")
axs[1].set_xscale("log", base=2); axs[1].set_yscale("log"); axs[1].set_xlabel("Changed components n"); axs[1].set_ylabel("Solve time (ms)"); axs[1].legend(frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.3); save(fig, "fig5_scalability")

fig, a = plt.subplots(figsize=(3.3, 2.2))
H = grid.pivot(index="tenure", columns="gamma0", values="gap_pct")
im = a.imshow(H.values, cmap="Blues", aspect="auto", origin="lower")
a.set_xticks(range(H.shape[1])); a.set_xticklabels([f"{v:g}" for v in H.columns]); a.set_yticks(range(H.shape[0])); a.set_yticklabels(H.index)
a.set_xlabel(r"Initial penalty weight $\gamma_0$ (single-flip moves)"); a.set_ylabel(r"Tabu tenure $\tau$")
vmax = max(H.values.max(), 1e-9)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.values[i, j]; a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w" if v > vmax * 0.6 else "k")
fig.colorbar(im, ax=a, label="Mean gap to optimum (%)", fraction=0.05); a.grid(False)
fig.tight_layout(pad=0.3); save(fig, "fig6_param_heatmap")

fi = pd.Series(np.mean(imps, 0), index=BASE_F + GRAPH_F).sort_values(ascending=False)
fi.round(4).to_csv(os.path.join(RES, "feature_importance.csv"))
fig, a = plt.subplots(figsize=(3.3, 1.9)); f10 = fi.head(10)[::-1]
a.barh(range(len(f10)), f10.values, color=["#1f5fa8" if k in GRAPH_F else "#9e9e9e" for k in f10.index])
a.set_yticks(range(len(f10))); a.set_yticklabels(f10.index, fontsize=6.5); a.set_xlabel("Normalised gain importance (regime G)")
fig.tight_layout(pad=0.3); save(fig, "fig2_importance")

# ------------------------------------------------------------------ 9. metrics.json
metrics = dict(dataset=summary, objective=dict(w_U=W_U, w_M=W_M, F_plane=F_PLANE), tabu=TABU,
               rho_mean=float(np.mean(rho_vals)), rho_std=float(np.std(rho_vals)),
               fault_model={reg: dict(auc_mean=float(auc[auc.regime == reg].auc.mean()), auc_std=float(auc[auc.regime == reg].auc.std()),
                                      ap_mean=float(auc[auc.regime == reg].ap.mean()), ap_std=float(auc[auc.regime == reg].ap.std())) for reg in REGIMES},
               main={f"{r}|{m_}": {k: float(v) for k, v in agg.loc[(r, m_)].items()} for r in REGIMES for m_ in order},
               tabu_vs_exact_seed0={reg: dict(optimal_pct=float(100 * np.mean(cache[(reg, "gaps")] < 1e-9)),
                                              max_gap=float(cache[(reg, "gaps")].max())) for reg in REGIMES},
               paired_stats_seed0=stat, scalability={int(k): {c: float(v) for c, v in r.items()} for k, r in scal_agg.iterrows()},
               param_grid_closure_moves_gap=grid_closure, param_grid_best=grid.loc[grid.gap_pct.idxmin()].to_dict(), param_grid_worst=grid.loc[grid.gap_pct.idxmax()].to_dict(),
               runtime_s=time.time() - t_start)
json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=1)
pd.set_option("display.width", 250)
print(json.dumps({k: metrics[k] for k in ["fault_model", "tabu_vs_exact_seed0", "paired_stats_seed0"]}, indent=1))
print(agg[[c for c in agg.columns if c.endswith("mean")]].round(3).to_string())
print(f"total {time.time() - t_start:.0f}s")
