
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from math import pi
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plot)

# --- Configuration ---
# Set style for academic quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.5,
    'figure.titlesize': 16,
    'figure.figsize': (8, 6)
})

# --------------------------------------------------------------------
# Learner's Tau Model: Core Functions
# --------------------------------------------------------------------

def tau_func(n, h, K):
    """
    Discrete hazard (Tau function): probability of success at trial n
    given failure on all previous trials.
    """
    n = np.array(n, dtype=float)
    return (n ** h) / (n ** h + K ** h)


def pmf_func(n_max, h, K):
    """
    Compute the PMF of the first-success trial N up to n_max
    using the product of survival and hazard.
    """
    n_values = np.arange(1, n_max + 1)
    taus = tau_func(n_values, h, K)

    pmf = np.zeros_like(taus)
    survival = 1.0

    for i, t in enumerate(taus):
        pmf[i] = survival * t
        survival *= (1 - t)

    return n_values, pmf


def calculate_metrics(h, K):
    """
    Compute the five learner metrics for given (h, K):

    1. Hazard at n = 10
    2. Difficulty Ratio (K / h)
    3. Early Success Probability (sum P(N <= floor(K)))
    4. Mean Time to Mastery (E[N])
    5. Peak Effort (max_n n * P(N = n))
    """
    n_max = 200  # Approximate infinity for sums
    n_vals, pmf = pmf_func(n_max, h, K)

    # 1. Discrete Hazard at n=10
    hazard_10 = tau_func(10, h, K)

    # 2. Difficulty Ratio
    diff_ratio = K / h

    # 3. Early Success Probability
    k_floor = int(np.floor(K))
    if k_floor < 1:
        early_success = 0.0
    else:
        early_success = np.sum(pmf[:k_floor])

    # 4. Mean Time to Mastery
    mtm = np.sum(n_vals * pmf)

    # 5. Peak Effort
    impact_curve = n_vals * pmf
    peak_effort = np.max(impact_curve)

    return [hazard_10, diff_ratio, early_success, mtm, peak_effort]

# --------------------------------------------------------------------
# Simulation + MLE + 500-dataset Parameter Recovery
# --------------------------------------------------------------------

def simulate_learner(h, K, N_max=500):
    """
    Simulate a single learner's first-success trial N from the
    Learner's Tau model using the discrete hazard.

    Returns:
        n (int): trial of first success (or N_max if censored).
    """
    if h <= 0 or K <= 0:
        raise ValueError("h and K must be positive")

    for n in range(1, N_max + 1):
        tau_n = (n ** h) / (n ** h + K ** h)
        if np.random.rand() < tau_n:
            return n
    # If no success by N_max, treat as censored at N_max
    return N_max


def log_likelihood_tau(params, data):
    """
    Log-likelihood of the Learner's Tau model for observed first-success data.

    params:
        [h, K] (both > 0)
    data:
        iterable of trial numbers n_j (first-success trials)

    Returns:
        total log-likelihood (float)
    """
    h, K = params

    # Enforce positivity
    if h <= 0 or K <= 0:
        return -np.inf

    total_ll = 0.0

    for n_j in data:
        # Failures on trials 1,...,n_j-1
        log_p_nj = 0.0
        for i in range(1, n_j):
            tau_i = (i ** h) / (i ** h + K ** h)
            if tau_i >= 1.0:
                log_p_nj = -np.inf
                break
            log_p_nj += np.log(1.0 - tau_i)

        if not np.isfinite(log_p_nj):
            total_ll += log_p_nj
            continue

        # Success on trial n_j
        tau_nj = (n_j ** h) / (n_j ** h + K ** h)
        if tau_nj <= 0.0:
            log_p_nj = -np.inf
        else:
            log_p_nj += np.log(tau_nj)

        total_ll += log_p_nj

    return total_ll


def neg_log_likelihood_tau(params, data):
    """Negative log-likelihood wrapper for optimization."""
    ll = log_likelihood_tau(params, data)
    return -ll if np.isfinite(ll) else np.inf


def mle_optimize_tau(data, h0=1.0, K0=10.0):
    """
    Maximum likelihood estimation for (h, K) using L-BFGS-B with positivity bounds.

    Args:
        data: list/array of first-success trials
        h0, K0: initial guesses

    Returns:
        h_hat, K_hat, max_loglik, success_flag
    """
    result = minimize(
        fun=neg_log_likelihood_tau,
        x0=np.array([h0, K0]),
        args=(np.array(data),),
        bounds=((1e-3, None), (1e-3, None)),
        method="L-BFGS-B"
    )
    h_hat, K_hat = result.x
    return h_hat, K_hat, -result.fun, result.success


def run_param_recovery(num_datasets=500, m=100,
                       h_true=2.0, K_true=10.0,
                       seed=42):
    """
    Parameter-recovery / bootstrap experiment:

    - Simulate `num_datasets` datasets of size m from (h_true, K_true)
    - Fit MLE (h_hat, K_hat) for each
    - Return arrays boot_h, boot_k for plotting histograms and ellipse.
    """
    np.random.seed(seed)
    boot_h = np.zeros(num_datasets)
    boot_k = np.zeros(num_datasets)

    for b in range(num_datasets):
        data_b = [simulate_learner(h_true, K_true) for _ in range(m)]
        h_hat, K_hat, _, success = mle_optimize_tau(data_b)
        if not success:
            print(f"Warning: optimization did not converge for dataset {b}")
        boot_h[b] = h_hat
        boot_k[b] = K_hat

    return boot_h, boot_k

# --------------------------------------------------------------------
# Profiles and Common Grid
# --------------------------------------------------------------------

profiles = [
    {'label': 'Fast (h=4, K=5)', 'h': 4, 'K': 5, 'color': '#f1c40f'},   # Yellow/Gold
    {'label': 'Balanced (h=2, K=10)', 'h': 2, 'K': 10, 'color': '#e67e22'},  # Orange
    {'label': 'Slow (h=1, K=15)', 'h': 1, 'K': 15, 'color': '#e74c3c'}  # Red
]

# Common N range for plots
N_MAX_PLOT = 50
n_plot = np.arange(1, N_MAX_PLOT + 1)

# --------------------------------------------------------------------
# 1. Figure: PMF Curves (fig_pmf_curves.png)
# --------------------------------------------------------------------
plt.figure()
for p in profiles:
    _, pmf = pmf_func(N_MAX_PLOT, p['h'], p['K'])
    plt.plot(n_plot, pmf, label=p['label'], color=p['color'])

plt.title("PMF Curves for Different Learners")
plt.xlabel("Trial Number (n)")
plt.ylabel("Probability P(N=n)")
plt.legend()
plt.tight_layout()
plt.savefig('fig_pmf_curves.png', dpi=300)
print("Generated fig_pmf_curves.png")

# --------------------------------------------------------------------
# 2. Figure: Single Histogram (fig_hist_single.png)
# --------------------------------------------------------------------
# Simulation for Balanced (h=2, K=10)
h_sim, k_sim = 2, 10
num_samples = 10000

# Generate synthetic data based on PMF weights
n_vals_sim, pmf_sim = pmf_func(100, h_sim, k_sim)
pmf_sim /= pmf_sim.sum()
samples = np.random.choice(n_vals_sim, size=num_samples, p=pmf_sim)

plt.figure()
plt.hist(samples, bins=range(1, 40), density=True, alpha=0.6,
         color='#3498db', edgecolor='black')
plt.plot(n_vals_sim[:40], pmf_sim[:40], 'r--', linewidth=2,
         label='Theoretical PMF')
plt.title(f"First Success Histogram (h={h_sim}, K={k_sim})")
plt.xlabel("Trial Number of First Success")
plt.ylabel("Frequency (Density)")
plt.legend()
plt.tight_layout()
plt.savefig('fig_hist_single.png', dpi=300)
print("Generated fig_hist_single.png")

# --------------------------------------------------------------------
# 3. Figure: Vary h (fig_tau_vary_h.png)
# --------------------------------------------------------------------
plt.figure()
K_fixed = 10
h_values = [0.5, 1, 2, 4]
colors_h = plt.cm.viridis(np.linspace(0, 1, len(h_values)))

for h, c in zip(h_values, colors_h):
    t_vals = tau_func(n_plot, h, K_fixed)
    plt.plot(n_plot, t_vals, label=f'h={h}', color=c)

plt.title(f"Discrete Hazard varying h (K={K_fixed})")
plt.xlabel("Trial Number")
plt.ylabel(r"Hazard $\lambda_n$")
plt.legend()
plt.tight_layout()
plt.savefig('fig_tau_vary_h.png', dpi=300)
print("Generated fig_tau_vary_h.png")

# --------------------------------------------------------------------
# 4. Figure: Vary K (fig_tau_vary_k.png)
# --------------------------------------------------------------------
plt.figure()
h_fixed = 2
K_values = [5, 10, 15, 20]
colors_k = plt.cm.magma(np.linspace(0.2, 0.8, len(K_values)))

for K, c in zip(K_values, colors_k):
    t_vals = tau_func(n_plot, h_fixed, K)
    plt.plot(n_plot, t_vals, label=f'K={K}', color=c)

plt.title(f"Discrete Hazard varying K (h={h_fixed})")
plt.xlabel("Trial Number")
plt.ylabel(r"Hazard $\lambda_n$")
plt.legend()
plt.tight_layout()
plt.savefig('fig_tau_vary_k.png', dpi=300)
print("Generated fig_tau_vary_k.png")

# --------------------------------------------------------------------
# 5. Figure: Discrete Hazard Combined (fig_tps_combined.png)
# --------------------------------------------------------------------
plt.figure()
for p in profiles:
    t_vals = tau_func(n_plot, p['h'], p['K'])
    plt.plot(n_plot, t_vals, label=p['label'], color=p['color'])

plt.title("Discrete Hazard Curves for Learner Profiles")
plt.xlabel("Trial Number")
plt.ylabel(r"Hazard $\lambda_n$")
plt.legend()
plt.tight_layout()
plt.savefig('fig_tps_combined.png', dpi=300)
print("Generated fig_tps_combined.png")

# --------------------------------------------------------------------
# 6, 7, 8. Figures: Bar Charts (Metrics)
# --------------------------------------------------------------------
metrics_data = []
labels = []
colors = []

for p in profiles:
    m = calculate_metrics(p['h'], p['K'])
    metrics_data.append(m)
    labels.append(p['label'].split(' ')[0])  # Just "Fast", "Balanced", etc.
    colors.append(p['color'])

metrics_data = np.array(metrics_data)  # Shape (3, 5)

# Metric Indices: 2=Early Success, 3=Mean Time, 4=Peak Effort
metric_configs = [
    (2, 'Early Success Probability', 'fig_sfi_bar.png'),
    (3, 'Mean Time to Mastery', 'fig_pw_bar.png'),
    (4, 'Peak Effort (Weighted Success)', 'fig_pis_bar.png')
]

for idx, title, fname in metric_configs:
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, metrics_data[:, idx], color=colors,
                   alpha=0.8, edgecolor='black')
    plt.title(title)
    plt.grid(axis='x')

    # Add value labels on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"Generated {fname}")

# --------------------------------------------------------------------
# 9. Figure: Log-Likelihood Surface (fig_loglik_surface.png)
# --------------------------------------------------------------------
# Generate synthetic data for Balanced profile
data_lik = np.random.choice(n_vals_sim, size=100, p=pmf_sim)

h_range = np.linspace(0.5, 3.5, 30)
k_range = np.linspace(5, 15, 30)
H, K_grid = np.meshgrid(h_range, k_range)
Z = np.zeros_like(H)

for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        hh = H[i, j]
        kk = K_grid[i, j]
        n_arr = data_lik
        ll_total = 0
        for val in n_arr:
            ii = np.arange(1, val)
            if len(ii) > 0:
                t_ii = (ii ** hh) / (ii ** hh + kk ** hh)
                ll_total += np.sum(np.log(1 - t_ii + 1e-9))
            t_n = (val ** hh) / (val ** hh + kk ** hh)
            ll_total += np.log(t_n + 1e-9)
        Z[i, j] = ll_total

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(H, K_grid, Z, cmap='viridis',
                       edgecolor='none', alpha=0.9)
ax.set_xlabel('h')
ax.set_ylabel('K')
ax.set_zlabel('Log-Likelihood')
ax.set_title('Log-Likelihood Surface')
plt.tight_layout()
plt.savefig('fig_loglik_surface.png', dpi=300)
print("Generated fig_loglik_surface.png")

# --------------------------------------------------------------------
# 10 & 15. Bootstrap via Real 500-dataset Recovery
# --------------------------------------------------------------------
boot_h, boot_k = run_param_recovery(
    num_datasets=500,
    m=100,
    h_true=2.0,
    K_true=10.0,
    seed=42
)

plt.figure(figsize=(6, 5))
plt.hist(boot_h, bins=20, color='#3498db',
         edgecolor='black', alpha=0.7)
plt.axvline(2.0, color='r', linestyle='--', label='True h=2')
plt.title(r"Bootstrap Distribution of $\hat{h}$")
plt.xlabel("Estimated h")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig('fig_bootstrap_h.png', dpi=300)
print("Generated fig_bootstrap_h.png")

plt.figure(figsize=(6, 5))
plt.hist(boot_k, bins=20, color='#2ecc71',
         edgecolor='black', alpha=0.7)
plt.axvline(10.0, color='r', linestyle='--', label='True K=10')
plt.title(r"Bootstrap Distribution of $\hat{K}$")
plt.xlabel("Estimated K")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig('fig_bootstrap_k.png', dpi=300)
print("Generated fig_bootstrap_k.png")

# --------------------------------------------------------------------
# 11. Figure: Confidence Ellipse (fig_conf_ellipse.png)
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(boot_h, boot_k, s=10, alpha=0.5, color='#f39c12')

cov = np.cov(boot_h, boot_k)
lambda_, v = np.linalg.eig(cov)
lambda_ = np.sqrt(lambda_)

ell = Ellipse(
    xy=(np.mean(boot_h), np.mean(boot_k)),
    width=lambda_[0] * 2 * 2,  # ~2 SDs in each direction
    height=lambda_[1] * 2 * 2,
    angle=np.rad2deg(np.arccos(v[0, 0]))
)
ell.set_facecolor('none')
ell.set_edgecolor('red')
ell.set_linewidth(2)
ax.add_artist(ell)

ax.set_title(r"95% Confidence Ellipse for $(\hat{h}, \hat{K})$")
ax.set_xlabel("h")
ax.set_ylabel("K")
plt.grid(True)
plt.tight_layout()
plt.savefig('fig_conf_ellipse.png', dpi=300)
print("Generated fig_conf_ellipse.png")

# --------------------------------------------------------------------
# 13. Figure: Histogram Overlay (fig_hist_overlay.png)
# --------------------------------------------------------------------
plt.figure()
for p in profiles:
    _, pmf_p = pmf_func(100, p['h'], p['K'])
    pmf_p /= pmf_p.sum()
    samps = np.random.choice(np.arange(1, 101), size=2000, p=pmf_p)
    plt.hist(
        samps, bins=range(1, 40), density=True,
        histtype='step', linewidth=2.5,
        label=p['label'], color=p['color']
    )

plt.title("First Success Histograms by Learner Type")
plt.xlabel("Trial Number")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig('fig_hist_overlay.png', dpi=300)
print("Generated fig_hist_overlay.png")

# --------------------------------------------------------------------
# 14. Figure: Radar Chart (fig_radar_chart.png)
# --------------------------------------------------------------------
raw_data = metrics_data.T
max_vals = raw_data.max(axis=1, keepdims=True)
normalized_data = raw_data / max_vals

categories = ['Hazard(10)', 'Diff Ratio', 'Early Success',
              'Mean Time', 'Peak Effort']
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75, 1.0],
           ["", "", "", ""], color="grey", size=7)
plt.ylim(0, 1)

for i, p in enumerate(profiles):
    values = normalized_data[:, i].tolist()
    values += values[:1]
    ax.plot(
        angles, values, linewidth=2, linestyle='solid',
        label=p['label'].split(' ')[0], color=p['color']
    )
    ax.fill(angles, values, color=p['color'], alpha=0.1)

plt.title("Radar Chart of Learner Metrics (Normalized)", y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()
plt.savefig('fig_radar_chart.png', dpi=300)
print("Generated fig_radar_chart.png")

print("\nAll figures and simulations completed successfully.")

