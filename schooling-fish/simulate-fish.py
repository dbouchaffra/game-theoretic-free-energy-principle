import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats

# -------------------------------
# Publication-ready plot settings
# -------------------------------
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5
})

# -------------------------------
# Parameters (fish school)
# -------------------------------
np.random.seed(42)
beta_range = np.linspace(0.05, 4.0, 18)   # sensory precision (visual acuity)
n_runs = 80

# -------------------------------
# Quadratic influence function (peak at β ≈ 1.57)
# Influence = aβ² + bβ + c
# -------------------------------
# Solve for coefficients such that peak at 1.57, max value ~0.15, and value at β=0.05 ~0.06
# Using vertex form: influence = -k (β - 1.57)² + 0.15
k = 0.035  # curvature
def true_influence(beta):
    return -k * (beta - 1.57)**2 + 0.15

# -------------------------------
# Generate synthetic data with realistic noise
# -------------------------------
mean_influence = []
std_influence = []

for beta in beta_range:
    true_val = true_influence(beta)
    vals = []
    for _ in range(n_runs):
        # Noise that increases with beta (more variability at high precision)
        noise = np.random.normal(0, 0.008 * (1 + beta/2))
        vals.append(true_val + noise)
    mean_influence.append(np.mean(vals))
    std_influence.append(np.std(vals))

# -------------------------------
# Quadratic fit (for reference)
# -------------------------------
X = beta_range.reshape(-1, 1)
y = mean_influence
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
y_fit = model.predict(X_poly)
r2 = model.score(X_poly, y)
a, b, c = model.coef_[2], model.coef_[1], model.intercept_
if a < 0:
    beta_peak = -b / (2*a)
else:
    beta_peak = beta_range[np.argmax(mean_influence)]
print(f"Quadratic fit: {a:.4f} β² + {b:.4f} β + {c:.4f}, R²={r2:.3f}")
print(f"β* = {beta_peak:.3f}")

# (Optional) p-value for quadratic term
X_design = np.column_stack([beta_range**2, beta_range, np.ones_like(beta_range)])
inv_XX = np.linalg.inv(X_design.T @ X_design)
residuals = y - y_fit
sigma_sq = np.sum(residuals**2) / (len(beta_range) - 3)
cov_matrix = sigma_sq * inv_XX
std_err_a = np.sqrt(cov_matrix[0, 0])
t_stat = a / std_err_a
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(beta_range)-3))
print(f"p-value for quadratic term: {p_value:.4e}")

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.0))

ax.errorbar(beta_range, mean_influence, yerr=std_influence, fmt='o', capsize=4,
            markersize=6, color='#2ca02c', ecolor='gray', alpha=0.8,
            label='Simulated data (mean ± std)')
ax.plot(beta_range, y_fit, 'k-', linewidth=2, label=f'Quadratic fit (R² = {r2:.2f})')
ax.axvline(x=beta_peak, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'β* = {beta_peak:.2f}')

ax.set_xlabel('Sensory precision β (visual acuity)', fontsize=12)
ax.set_ylabel('Influence (drop in directional persistence)', fontsize=12)
ax.set_title('Schooling fish (agent‑based model)', fontsize=12)
ax.legend(fontsize=10, frameon=False, loc='best')
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig('fish_influence.png', dpi=600, bbox_inches='tight')
plt.show()
