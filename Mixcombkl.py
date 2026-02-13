import numpy as np
import itertools
import time
import math
import matplotlib.pyplot as plt
plt.rcParams.update({
    # figure size
    "figure.figsize": (8, 5),
    "figure.dpi": 120,

    # BIGGER TEXT (important)
    "font.size": 16,          # base
    "axes.titlesize": 18,     # title
    "axes.labelsize": 16,     # x/y label
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,

    # cleaner style
    "lines.linewidth": 2.5
})

# optional: exact LP solver for decomposition
try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# -----------------------------
# Helpers: action set
# -----------------------------
def all_m_subsets(d, m):
    idxs = list(itertools.combinations(range(d), m))
    N = len(idxs)
    S = np.zeros((N, d), dtype=np.int8)
    for k, comb in enumerate(idxs):
        S[k, list(comb)] = 1
    return S


# -----------------------------
# Bernoulli environment (NEW)
# -----------------------------
class BernoulliMsetsEnv:
    def __init__(self, mu, S, rng=None):
        """
        mu : array-like, length d, base-arm means in [0,1]
        S  : (N, d) super-arm indicator matrix (used to compute best_reward)
        rng: seed or np.random.Generator or None
        """
        self.mu = np.asarray(mu, dtype=float)
        self.d = len(self.mu)
        self.S = S
        # Random generator for stochastic rewards
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)
        # best reward in expectation (used for regret)
        self.best_reward = float(np.max(S @ self.mu))

    def pull(self, M_vec):
        """
        Sample Bernoulli rewards for all base arms and return the sum on M_vec.
        Returns a float.
        """
        # independent Bernoulli draws per base arm
        w = (self.rng.random(self.d) < self.mu).astype(float)
        return float(M_vec @ w)


# -----------------------------
# Numeric utilities (unchanged)
# -----------------------------
def kl_project_onto_capped_simplex(q_tilde, ub, tol=1e-12, max_iter=100):
    q = np.asarray(q_tilde, dtype=float)
    q = np.clip(q, 1e-20, None)
    d = len(q)

    if ub * d < 1.0 - 1e-15:
        p = np.full(d, ub)
        p /= p.sum()
        return p

    low = 0.0
    high = 1.0
    s_high = np.minimum(ub, high * q).sum() - 1.0
    cnt = 0
    while s_high < 0 and cnt < 200:
        high *= 2.0
        s_high = np.minimum(ub, high * q).sum() - 1.0
        cnt += 1

    if s_high < 0:
        p = np.minimum(ub, q / q.sum())
        p /= p.sum()
        return p

    mid = None
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p_mid = np.minimum(ub, mid * q)
        s_mid = p_mid.sum() - 1.0
        if abs(s_mid) <= tol:
            p_mid /= p_mid.sum()
            return p_mid
        if s_mid < 0:
            low = mid
        else:
            high = mid

    p = np.minimum(ub, mid * q)
    p /= p.sum()
    return p


def lp_decomposition(S, target_vec):
    N, d = S.shape
    target = np.asarray(target_vec, dtype=float)

    if SCIPY_AVAILABLE:
        try:
            A_eq = np.vstack([S.T, np.ones((1, N))])
            b_eq = np.concatenate([target, np.array([1.0])])
            c = np.zeros(N)
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * N, method='highs')
            if res.success:
                p = res.x
                p[p < 1e-14] = 0.0
                if abs(p.sum() - 1.0) > 1e-9:
                    p = p / p.sum()
                return p
        except Exception:
            pass

    # greedy fallback (same as original)
    p = np.zeros(N, dtype=float)
    residual = target.copy()
    S_f = S.astype(float)
    active = []
    max_iter = min(N, d + 10)
    for _ in range(max_iter):
        scores = S_f @ residual
        idx = int(np.argmax(scores))
        if idx in active:
            break
        active.append(idx)
        A = S_f[active].T
        try:
            w, *_ = np.linalg.lstsq(A, target, rcond=None)
            w = np.maximum(w, 0.0)
            if w.sum() == 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
            p_tmp = np.zeros(N)
            for i, a in enumerate(active):
                p_tmp[a] = w[i]
            approx = S_f.T @ p_tmp
            residual = target - approx
            p = p_tmp
            if np.linalg.norm(residual) < 1e-8:
                break
        except Exception:
            break

    p = np.maximum(p, 0.0)
    if p.sum() == 0:
        p = np.ones(N) / N
    else:
        p /= p.sum()
    return p


# -----------------------------
# MixCombKL (algorithm logic unchanged; env.pull is now stochastic)
# -----------------------------
def mixcombkl(S, env, T=10000, alpha=0.2, gamma=None, eta=None, seed=0, report_every=0):
    """
    S: (N,d) action matrix
    env: environment object with .pull(M_vec) returning a scalar reward, and env.best_reward
    T, alpha, gamma, eta, seed: same semantics as original
    returns: regret (T,), R_super (N,), R_base (d,), logs dict
    """
    rng = np.random.default_rng(seed)
    N, d = S.shape
    m = int(S[0].sum())
    B = m

    # rho0 and q0
    rho0 = (S.sum(axis=0) / float(m * N))
    rho0 = np.clip(rho0, 1e-20, None)
    rho0 /= rho0.sum()
    q = rho0.copy()
    ub = 1.0 / m

    # lambda_min (for heuristics)
    M_prior = S.T @ ((1.0 / N) * S)
    eigs = np.linalg.eigvalsh(M_prior)
    pos = eigs[eigs > 1e-12]
    lambda_min = float(pos.min()) if pos.size > 0 else 0.0

    # default gamma & eta (same formulas)
    if gamma is None:
        rho_min = max(rho0.min(), 1e-12)
        C = (lambda_min / (m ** 1.5)) if lambda_min > 0 else 1.0 / (m ** 1.5)
        numer = math.sqrt(max(1e-12, m * math.log(1.0 / rho_min)))
        denom = numer + math.sqrt(max(1e-12, C * (C * m * m * d + m) * T))
        gamma = float(max(1e-12, numer / denom))
    if eta is None:
        C = (lambda_min / (m ** 1.5)) if lambda_min > 0 else 1.0 / (m ** 1.5)
        eta = gamma * C

    # bookkeeping
    regret = np.zeros(T, dtype=float)
    cum_reg = 0.0
    R_super = np.zeros(N, dtype=float)
    R_base = np.zeros(d, dtype=float)

    # logs for offline MSE
    Klist = []
    plist = []
    Utlist = []
    Ylist = []

    start = time.time()
    for t in range(1, T + 1):
        qprime = (1.0 - gamma) * q + gamma * rho0
        target_vec = m * qprime
        p = lp_decomposition(S, target_vec)

        prob_ut1 = min(1.0, 1.0 / (2.0 * (t ** alpha)))
        Ut = rng.random() < prob_ut1

        if Ut:
            pi = np.full(N, 1.0 / N)
        else:
            pi = p.copy()

        K = int(rng.choice(N, p=pi))
        Mvec = S[K]

        # interaction with stochastic environment
        Y = env.pull(Mvec)

        # record logs BEFORE any possible mutation
        Klist.append(K)
        plist.append(p.copy())   # store the p vector (not pi) so we can reconstruct marginals
        Utlist.append(Ut)
        Ylist.append(float(Y))

        Sigma = S.T @ (p[:, None] * S)
        Sigma_plus = np.linalg.pinv(Sigma, rcond=1e-8)
        wtilde = float(Y) * (Sigma_plus @ Mvec)
        wtilde = np.clip(wtilde, -1e6, 1e6)

        log_q = np.log(np.clip(q, 1e-20, None))
        log_q_tilde = log_q + eta * wtilde
        q_tilde_unnorm = np.exp(log_q_tilde - np.max(log_q_tilde))
        q_tilde = q_tilde_unnorm / q_tilde_unnorm.sum()

        if Ut:
            q_new = q.copy()
        else:
            q_new = kl_project_onto_capped_simplex(q_tilde, ub=ub)
        q = q_new

        if Ut:
            scalar = 2.0 * (t ** alpha)
            contribs = scalar * (S @ wtilde)   # length N
            R_super += contribs
            R_base += scalar * wtilde

        cum_reg += (env.best_reward - Y)
        regret[t - 1] = cum_reg

        if report_every and ((t == 1) or (t % report_every == 0)):
            elapsed = time.time() - start
            print(f"t={t:7d} cum_reg={cum_reg:8.3f} avg_reg={cum_reg/t:8.6f} elapsed={elapsed:5.2f}s")

    logs = {
        "Klist": np.array(Klist, dtype=np.int32),   # length T
        "plist": np.array(plist, dtype=float),      # shape (T, N)
        "Utlist": np.array(Utlist, dtype=bool),    # length T
        "Ylist": np.array(Ylist, dtype=float)      # length T
    }

    return regret, R_super, R_base, logs


# -----------------------------
# Experiment runner: alpha sweep + average over random mu per run (PAIRED)
# -----------------------------
def run_alpha_sweep(
    d=8, m=3, T=5000, alphas=None, n_runs=20, seeds_start=1000, report_every=0
):
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 1.0]

    S = all_m_subsets(d, m)
    N, _ = S.shape

    # --- SAMPLE ALL mu ONCE (paired across alphas) ---
    rng_mu = np.random.default_rng(seeds_start + 9999)
    mu_all = [rng_mu.uniform(0.1, 0.9, size=d) for _ in range(n_runs)]

    results = {}
    results_logs = {}   # store per-alpha per-run logs

    for alpha in alphas:
        print(f"\n=== Running alpha = {alpha} (n_runs = {n_runs}) ===")
        regrets_all = np.zeros((n_runs, T), dtype=float)
        Rsuper_all = np.zeros((n_runs, N), dtype=float)
        Rbase_all = np.zeros((n_runs, d), dtype=float)
        per_run_logs = []
        R_checkpoints_all = []

        times = []

        for run in range(n_runs):
            seed = seeds_start + run
            t0 = time.time()

            # use pre-sampled mu (paired)
            mu = mu_all[run]

            # create Bernoulli environment for this mu (seeded RNG per run)
            env = BernoulliMsetsEnv(mu, S, rng=seed + 12345)

            # run algorithm unchanged
            regret, R_super, R_base, logs = mixcombkl(S, env, T=T, alpha=alpha, seed=seed, report_every=report_every)
            times.append(time.time() - t0)

            regrets_all[run] = regret
            Rsuper_all[run] = R_super
            Rbase_all[run] = R_base

            # compute R checkpoints at fractions T/5,...,T from the regret vector
            fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
            R_cp = {}
            for f in fractions:
                t_frac = int(T * f)
                R_cp[f] = float(regret[t_frac - 1])  # cumulative regret at this round
            R_checkpoints_all.append(R_cp)

            per_run_logs.append(logs)

            print(f" run {run+1}/{n_runs} done, final cum_reg={regret[-1]:.3f}, time={times[-1]:.2f}s")

        avg_time = np.mean(times)
        print(f" alpha={alpha} average run time: {avg_time:.2f}s")

        # Save raw outputs (still useful)
        results[alpha] = {
            "regrets_all": regrets_all,
            "Rsuper_all": Rsuper_all,
            "Rbase_all": Rbase_all,
            "R_checkpoints": R_checkpoints_all,
            "mu_all": mu_all  # same mu_all for every alpha (paired)
        }
        results_logs[alpha] = per_run_logs

        # ------------------------------------------------------------------
        # Offline IW-based MSE computation (stable, uses only logged history)
        # ------------------------------------------------------------------
        eps = 1e-12
        mse_super_runs_iw = np.zeros(n_runs, dtype=float)
        mse_base_runs_iw = np.zeros(n_runs, dtype=float)

        for r in range(n_runs):
            logs = per_run_logs[r]
            Klist = logs["Klist"]       # length T
            plist = logs["plist"]       # shape (T, N)
            Utlist = logs["Utlist"]     # length T (bool)
            Ylist = logs["Ylist"]       # length T (float)

            Tcur = len(Klist)
            # compute time-by-time marginal probs p_t(e)
            p_marg = np.zeros((Tcur, d), dtype=float)
            for t in range(Tcur):
                if Utlist[t]:
                    pi = np.ones(N) / N
                else:
                    pi = plist[t]
                # marginal: p_t(e) = sum_M pi(M) * 1[e in M] = (pi @ S)
                p_marg[t] = pi @ S   # shape (d,)

            # importance-weight estimator per base arm (self-normalized)
            mu_hat = np.zeros(d, dtype=float)
            for e in range(d):
                num = 0.0
                den = 0.0
                for t in range(Tcur):
                    k = int(Klist[t])
                    if S[k, e] == 1:
                        wt = 1.0 / max(p_marg[t, e], eps)
                        num += wt * (Ylist[t]*m)
                        den += wt
                mu_hat[e] = num / max(den, eps)

            # no shrink (faithful to paper)
            true_base_rewards = results[alpha]["mu_all"][r]
            mu_hat = 1 * mu_hat + 0 * true_base_rewards.mean()

            # super-arm estimates
            super_hat = S @ mu_hat  # length N

            # compute MSE over super-arm-pair gaps
            pairs_super = [(i, j) for i in range(N) for j in range(i + 1, N)]
            sqsum = 0.0
            for (i, j) in pairs_super:
                est_gap = float(super_hat[i] - super_hat[j])
                true_gap = float((S @ true_base_rewards)[i] - (S @ true_base_rewards)[j])
                sqsum += (est_gap - true_gap) ** 2
            mse_super_runs_iw[r] = sqsum / len(pairs_super)

            # compute MSE over base-arm-pair gaps
            pairs_base = [(i, j) for i in range(d) for j in range(i + 1, d)]
            sqsum = 0.0
            for (i, j) in pairs_base:
                est_gap = float(mu_hat[i] - mu_hat[j])
                true_gap = float(true_base_rewards[i] - true_base_rewards[j])
                sqsum += (est_gap - true_gap) ** 2
            mse_base_runs_iw[r] = sqsum / len(pairs_base)

        # Also compute the original MSEs based on R_super / R_base (kept for comparison)
        idx_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        n_pairs_super = len(idx_pairs)
        mse_super_runs_orig = np.zeros(n_runs, dtype=float)
        for r in range(n_runs):
            est = (Rsuper_all[r][:, None] - Rsuper_all[r][None, :]) / float(T)
            ssum = 0.0
            for (i, j) in idx_pairs:
                est_gap = est[i, j]
                true_gap = float((S @ results[alpha]["mu_all"][r])[i] - (S @ results[alpha]["mu_all"][r])[j])
                err = est_gap - true_gap
                ssum += err * err
            mse_super_runs_orig[r] = ssum / n_pairs_super

        idx_pairs_base = [(i, j) for i in range(d) for j in range(i + 1, d)]
        n_pairs_base = len(idx_pairs_base)
        mse_base_runs_orig = np.zeros(n_runs, dtype=float)
        for r in range(n_runs):
            estb = (Rbase_all[r][:, None] - Rbase_all[r][None, :]) / float(T)
            ssum = 0.0
            for (i, j) in idx_pairs_base:
                est_gap = estb[i, j]
                true_gap = float(results[alpha]["mu_all"][r][i] - results[alpha]["mu_all"][r][j])
                err = est_gap - true_gap
                ssum += err * err
            mse_base_runs_orig[r] = ssum / n_pairs_base

        # store both IW-based MSE and original MSE so you can compare
        results[alpha]["mse_super_runs_iw"] = mse_super_runs_iw
        results[alpha]["mse_base_runs_iw"] = mse_base_runs_iw
        results[alpha]["mse_super_runs_orig"] = mse_super_runs_orig
        results[alpha]["mse_base_runs_orig"] = mse_base_runs_orig
        # also store logs
        results_logs[alpha] = per_run_logs

    # ============================================================
    # 1. Regret curves (averaged across runs)
    # ============================================================
    plt.figure(figsize=(10, 6))
    t_axis = np.arange(1, T + 1)
    for alpha in alphas:
        arr = results[alpha]["regrets_all"]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        plt.plot(t_axis, mean, label=f"alpha={alpha}")
    plt.xlabel("Round t")
    plt.ylabel("Cumulative regret")
    plt.title(f"MixCombKL regret curves (d={d}, m={m}, T={T})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 2. MSE bar plots (IW-based)
    # ============================================================
    mean_mse_super = [results[a]["mse_super_runs_iw"].mean() for a in alphas]
    std_mse_super = [results[a]["mse_super_runs_iw"].std() for a in alphas]
    mean_mse_base = [results[a]["mse_base_runs_iw"].mean() for a in alphas]
    std_mse_base = [results[a]["mse_base_runs_iw"].std() for a in alphas]

    x = np.arange(len(alphas))
    width = 0.38

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, mean_mse_super, width, yerr=std_mse_super, capsize=6,
            label=r"MSE($\Delta_M$)")
    plt.bar(x + width / 2, mean_mse_base, width, yerr=std_mse_base, capsize=6,
            label=r"MSE($\Delta_\mu$)")
    plt.xticks(x, [str(a) for a in alphas])
    plt.xlabel("alpha")
    plt.ylabel("MSE")
    plt.title("MixCombKL gap estimator MSE")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 3. Summary tables (prints both original and IW MSE)
    # ============================================================
    print("\nSummary (averaged over runs):")
    print("alpha\t final_reg_mean\t final_reg_std\t MSE_super_iw_mean\t MSE_super_iw_std\t MSE_base_iw_mean\t MSE_base_iw_std")
    for alpha in alphas:
        regs = results[alpha]["regrets_all"][:, -1]
        print(f"{alpha}\t {regs.mean():10.3f}\t {regs.std():8.3f}\t "
              f"{results[alpha]['mse_super_runs_iw'].mean():10.6e}\t {results[alpha]['mse_super_runs_iw'].std():8.6e}\t "
              f"{results[alpha]['mse_base_runs_iw'].mean():10.6e}\t {results[alpha]['mse_base_runs_iw'].std():8.6e}")

    print("\n(OLD estimator based on R_super/R_base) — kept for comparison:")
    print("alpha\t MSE_super_orig_mean\t MSE_super_orig_std\t MSE_base_orig_mean\t MSE_base_orig_std")
    for alpha in alphas:
        print(f"{alpha}\t {results[alpha]['mse_super_runs_orig'].mean():10.6e}\t {results[alpha]['mse_super_runs_orig'].std():8.6e}\t "
              f"{results[alpha]['mse_base_runs_orig'].mean():10.6e}\t {results[alpha]['mse_base_runs_orig'].std():8.6e}")

    # ============================================================
    # 4. Print R(T/5), R(2T/5), ..., R(T) averages
    # ============================================================
    print("\nR(T/5), R(2T/5), ..., R(T) averages:")
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    print("alpha\t" + "\t".join([f"R({int(T*f)})" for f in fractions]))
    for alpha in alphas:
        avg_regs = []
        for f in fractions:
            vals = [r[f] for r in results[alpha]["R_checkpoints"]]
            avg_regs.append(np.mean(vals))
        print(f"{alpha}\t" + "\t".join(f"{v:.1f}" for v in avg_regs))

    return results, results_logs


# -----------------------------
# Main: run experiment
# -----------------------------
if __name__ == "__main__":
    # configuration (moderate size)
    d = 8
    m = 3
    T = 5000
    n_runs = 20
    alphas = [0.0, 0.25, 0.5, 1.0]
    seed_start = 3600
    report_every = 0

    t_all = time.time()
    results, results_logs = run_alpha_sweep(
        d=d, m=m, T=T, alphas=alphas, n_runs=n_runs, seeds_start=seed_start, report_every=report_every
    )
    print("All experiments done in {:.2f}s".format(time.time() - t_all))

