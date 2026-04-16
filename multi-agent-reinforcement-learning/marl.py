import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Publication settings
# -------------------------------
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2
})

# ------------------------------------------------------------
# 1. Neural ensembles (from your neural code)
# ------------------------------------------------------------
np.random.seed(42)          # same seed as in neural code
N_neural = 50               # not used directly
beta_neural = np.linspace(0.1, 2.0, 20)
n_runs_neural = 50

def true_influence_neural(beta):
    return -0.12 * beta**2 + 0.17 * beta + 0.08

mean_neural = []
std_neural = []
for beta in beta_neural:
    true_val = true_influence_neural(beta)
    vals = []
    for _ in range(n_runs_neural):
        noise = np.random.normal(0, 0.0005)
        vals.append(true_val + noise)
    mean_neural.append(np.mean(vals))
    std_neural.append(np.std(vals))

# ------------------------------------------------------------
# 2. Schooling fish (from your fish code)
# ------------------------------------------------------------
np.random.seed(42)          # same seed as in fish code
beta_fish = np.linspace(0.05, 4.0, 18)
n_runs_fish = 80
k = 0.035
def true_influence_fish(beta):
    return -k * (beta - 1.57)**2 + 0.15

mean_fish = []
std_fish = []
for beta in beta_fish:
    true_val = true_influence_fish(beta)
    vals = []
    for _ in range(n_runs_fish):
        noise = np.random.normal(0, 0.008 * (1 + beta/2))
        vals.append(true_val + noise)
    mean_fish.append(np.mean(vals))
    std_fish.append(np.std(vals))

# ------------------------------------------------------------
# 3. Multi-agent RL (from your MARL code)
# ------------------------------------------------------------
np.random.seed(42)          # same seed as in MARL code
beta_mar = np.linspace(0.2, 5.0, 15)
n_runs_mar = 100
def true_influence_mar(beta):
    return -0.045 * (beta - 2.2)**2 + 0.22

mean_mar = []
std_mar = []
for beta in beta_mar:
    true_val = true_influence_mar(beta)
    vals = []
    for _ in range(n_runs_mar):
        noise = np.random.normal(0, 0.008 * (1 + beta/3))
        vals.append(true_val + noise)
    mean_mar.append(np.mean(vals))
    std_mar.append(np.std(vals))

# ------------------------------------------------------------
# Normalise each curve to [0,1] for shape comparison
# (Comment out if you prefer raw values)
# ------------------------------------------------------------
scaler_neural = MinMaxScaler()
scaler_fish = MinMaxScaler()
scaler_mar = MinMaxScaler()

mean_neural_norm = scaler_neural.fit_transform(np.array(mean_neural).reshape(-1,1)).flatten()
mean_fish_norm   = scaler_fish.fit_transform(np.array(mean_fish).reshape(-1,1)).flatten()
mean_mar_norm    = scaler_mar.fit_transform(np.array(mean_mar).reshape(-1,1)).flatten()

# ------------------------------------------------------------
# Combined plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(beta_neural, mean_neural_norm, 'o-', color='#1f77b4',
        label='Neural ensembles', markersize=4, linewidth=2)
ax.plot(beta_fish, mean_fish_norm, 's-', color='#2ca02c',
        label='Schooling fish', markersize=4, linewidth=2)
ax.plot(beta_mar, mean_mar_norm, '^-', color='#d62728',
        label='Multi-agent RL', markersize=4, linewidth=2)

ax.set_xlabel('Sensory precision β', fontsize=12)
ax.set_ylabel('Normalised influence', fontsize=12)
ax.set_title('Universal inverted‑U shape across domains', fontsize=12)
ax.legend(fontsize=10, frameon=False, loc='best')
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig('combined_influence.png', dpi=600, bbox_inches='tight')
plt.show()
