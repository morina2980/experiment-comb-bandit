import numpy as np
import itertools
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


# ================================================================
# InitUCB (unchanged)
# ================================================================
def init_ucb_oracle(true_means, d, m, M_list, rng, max_iters=2000):
    u = np.ones(d, dtype=int)
    counts = np.zeros(d, dtype=int)
    sums = np.zeros(d, dtype=float)
    Me_seq = []
    init_rounds = 0

    while np.any(u == 1) and init_rounds < max_iters:
        best_k = max(range(len(M_list)),
                      key=lambda kk: sum(u[e] for e in M_list[kk]))
        M = list(M_list[best_k])
        Me_seq.append(M)

        w = (rng.random(d) < true_means).astype(float)

        for e in M:
            if u[e] == 1:
                u[e] = 0
            counts[e] += 1
            sums[e] += w[e]

        init_rounds += 1

    return counts, sums, Me_seq, len(Me_seq), init_rounds


# ================================================================
# MixCombUCB (unchanged except small safety tweaks)
# ================================================================
class MixCombUCB_Fixed:
    def __init__(self, d, M_list, m, T=2000, rng=None, alpha=0.5):
        self.d = int(d)
        self.M_list = [list(M) for M in M_list]
        self.K = len(M_list)
        self.m = m
        self.T = T
        self.rng = np.random.default_rng(rng)
        self.alpha = alpha

        self.counts = np.zeros(d, int)
        self.sums = np.zeros(d, float)
        self.R = np.zeros(d, float)

        self.Me_seq = []
        self.m0 = 0

    def initialize(self, c0, s0, Me_seq, m0):
        self.counts = c0.copy()
        self.sums = s0.copy()
        self.Me_seq = [list(M) for M in Me_seq]
        self.m0 = m0
        self.R = np.zeros_like(self.R)

    def _compute_ucb(self, t):
        denom = np.maximum(1, self.counts)
        mu_hat = self.sums / denom
        bonus = np.sqrt(2 * np.log(max(2, t)) / denom)
        return np.minimum(1, mu_hat + bonus), mu_hat

    def _argmax_superarm(self, ucb):
        return max(self.M_list, key=lambda M: sum(ucb[e] for e in M))

    def _sample_mixture(self, Mtilda, alpha_t):
        if self.m0 == 0:
            return Mtilda
        prob = max(0.0, 1 - self.m0 * alpha_t)
        if self.rng.random() < prob:
            return Mtilda
        j = self.rng.integers(self.m0)
        return self.Me_seq[j]

    def run_single(self, true_means, opt_reward, init_rounds):
        regrets = np.zeros(self.T)
        cum = 0.0
        offset = init_rounds

        for tt in range(1, self.T + 1):
            t = tt + offset
            ucb, mu_hat = self._compute_ucb(t)

            Mtilda = self._argmax_superarm(ucb)
            alpha_t = 1.0 / (self.m0 * (t**self.alpha)) if self.m0 > 0 else 0
            M_t = self._sample_mixture(Mtilda, alpha_t)

            w = (self.rng.random(self.d) < true_means).astype(float)

            included = np.zeros(self.d)
            if self.m0 > 0:
                for Me in self.Me_seq:
                    for e in Me:
                        included[e] += 1

            P_e = np.zeros(self.d)
            for e in range(self.d):
                P_e[e] = (1 - self.m0 * alpha_t) * (1 if e in Mtilda else 0)
                if self.m0 > 0:
                    P_e[e] += alpha_t * included[e]
                P_e[e] = max(P_e[e], 1e-12)

            for e in M_t:
                self.counts[e] += 1
                self.sums[e] += w[e]
                self.R[e] += w[e] / P_e[e]

            expected = sum(true_means[e] for e in M_t)
            cum += opt_reward - expected
            regrets[tt - 1] = cum

        mu_hat_final = self.sums / np.maximum(1, self.counts)
        return regrets, mu_hat_final, self.R.copy()


# ================================================================
# MSE utilities
# ================================================================
def mse_pairwise_gaps(mu_hat, mu_true, M_list):
    d = len(mu_true)
    K = len(M_list)

    # base-arm
    gaps_hat = mu_hat[:, None] - mu_hat[None, :]
    gaps_true = mu_true[:, None] - mu_true[None, :]
    mse_base = np.mean((gaps_hat - gaps_true)**2)

    # super-arm
    f_hat = np.array([mu_hat[M].sum() for M in M_list])
    f_true = np.array([mu_true[M].sum() for M in M_list])

    ghat = f_hat[:, None] - f_hat[None, :]
    gtrue = f_true[:, None] - f_true[None, :]
    mse_super = np.mean((ghat - gtrue)**2)

    return float(mse_base), float(mse_super)


# ================================================================
# Experiment: μ sampling + α sweep
# ================================================================
def run_alpha_sweep_ucb(
    d=9, m=4, T=2000,
    alphas=[0, 0.25, 0.5, 1.0],
    n_mu=20,
    seed0=2025
):
    rng = np.random.default_rng(seed0)
    M_list = [list(c) for c in itertools.combinations(range(d), m)]
    checkpoints = [400, 800, 1200, 1600, 2000]

    results = {}
    curves = {}

    print("\n=== MixCombUCB α-sweep with μ~U[0.1,0.9]^d ===\n")

    for alpha in alphas:
        print(f"\n----- α = {alpha} -----\n")

        final_regs, mse_supers, mse_bases = [], [], []
        ck_all = []
        reg_mat = []

        for r in range(n_mu):
            mu = rng.uniform(0.1, 0.9, size=d)

            opt_reward = max(sum(mu[e] for e in M) for M in M_list)

            rng2 = np.random.default_rng(seed0 + r)
            counts0, sums0, Me_seq, m0, init_rounds = init_ucb_oracle(
                mu, d, m, M_list, rng2
            )

            alg = MixCombUCB_Fixed(d, M_list, m, T=T,
                                   rng=seed0 + 77*r + int(100*alpha),
                                   alpha=alpha)
            alg.initialize(counts0, sums0, Me_seq, m0)

            regrets, mu_hat, R_final = alg.run_single(
                mu, opt_reward, init_rounds
            )

            reg_mat.append(regrets)

            ck = [regrets[t - 1] for t in checkpoints]
            ck_all.append(ck)
            final_regs.append(regrets[-1])

            mse_b, mse_s = mse_pairwise_gaps(mu_hat, mu, M_list)
            mse_bases.append(mse_b)
            mse_supers.append(mse_s)

        results[alpha] = {
            "final_reg": np.mean(final_regs),

            "mse_base_mean": np.mean(mse_bases),
            "mse_base_std": np.std(mse_bases),

            "mse_super_mean": np.mean(mse_supers),
            "mse_super_std": np.std(mse_supers),

            "checkpoints": np.mean(ck_all, axis=0)
        }

        curves[alpha] = np.stack(reg_mat, axis=0)

        print(f"Avg final regret : {results[alpha]['final_reg']:.3f}")
        print(f"Avg MSE_super    : {results[alpha]['mse_super_mean']:.3e} ± {results[alpha]['mse_super_std']:.3e}")
        print(f"Avg MSE_base     : {results[alpha]['mse_base_mean']:.3e} ± {results[alpha]['mse_base_std']:.3e}")
        print(f"Avg checkpoints  : {np.round(results[alpha]['checkpoints'],1)}")

    # plot regret curves
    plt.figure(figsize=(10, 6))
    t_axis = np.arange(1, T + 1)
    for alpha in alphas:
        arr = curves[alpha]
        mean = arr.mean(0)
        std = arr.std(0)
        plt.plot(t_axis, mean, label=f"α={alpha}", lw=2)
    plt.xlabel("t")
    plt.ylabel("Cumulative regret")
    plt.title(f"MixCombUCB regret curves (d={d}, m={m}, T={T})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ============================================================
    # MSE bar plot
    # ============================================================
    xs = np.arange(len(alphas))
    width = 0.35

    mse_base_means = [results[a]["mse_base_mean"] for a in alphas]
    mse_base_stds = [results[a]["mse_base_std"] for a in alphas]
    mse_super_means = [results[a]["mse_super_mean"] for a in alphas]
    mse_super_stds = [results[a]["mse_super_std"] for a in alphas]

    plt.figure(figsize=(10, 6))
    plt.bar(xs - width / 2, mse_base_means, width,
            yerr=mse_base_stds, capsize=5,
            label="MSE($\Delta_M$)")
    plt.bar(xs + width / 2, mse_super_means, width,
            yerr=mse_super_stds, capsize=5,
            label="MSE($\Delta_\mu$)")
    plt.xticks(xs, [str(a) for a in alphas])
    plt.xlabel("alpha")
    plt.ylabel("MSE")
    plt.title("MixCombUCB gap estimator MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return results


# ================================================================
# Run
# ================================================================
if __name__ == "__main__":
    results = run_alpha_sweep_ucb(
        d=9, m=4, T=2000,
        alphas=[0, 0.25, 0.5, 1.0],
        n_mu=20,
        seed0=1234
    )

    print("\n=== FINAL SUMMARY ===")
    for alpha in results:
        r = results[alpha]
        print(f"\nα={alpha}")
        print(f"Final regret:  {r['final_reg']:.3f}")
        print(f"MSE_super  :   {r['mse_super_mean']:.3e} ± {r['mse_super_std']:.3e}")
        print(f"MSE_base   :   {r['mse_base_mean']:.3e} ± {r['mse_base_std']:.3e}")
        print(f"Checkpoints:   {np.round(r['checkpoints'],1)}")

