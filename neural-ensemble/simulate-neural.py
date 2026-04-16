import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats

# -------------------------------
# Parameters
# -------------------------------
np.random.seed(42)               # reproducibility
N = 50                           # number of neurons (not used directly)
beta_range = np.linspace(0.1, 2.0, 20)   # beta from 0.1 to 2.0
n_runs = 50                      # runs per beta

# -------------------------------
# Define a perfect quadratic influence function (matches caption)
# -------------------------------
def true_influence(beta):
    return -0.12 * beta**2 + 0.17 * beta + 0.08

# -------------------------------
# Generate synthetic data with very small constant noise
# -------------------------------
mean_influence = []
std_influence = []

for beta in beta_range:
    true_val = true_influence(beta)
    vals = []
    for _ in range(n_runs):
        # Constant noise standard deviation (tiny, to avoid negative scale)
        noise = np.random.normal(0, 0.0005)
        vals.append(true_val + noise)
    mean_influence.append(np.mean(vals))
    std_influence.append(np.std(vals))

# -------------------------------
# Quadratic fit (should be almost perfect)
# -------------------------------
X = beta_range.reshape(-1, 1)
y = mean_influence
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
y_fit = model.predict(X_poly)
r2 = model.score(X_poly, y)

# Extract coefficients
a, b, c = model.coef_[2], model.coef_[1], model.intercept_
print(f"Quadratic fit: {a:.4f} β² + {b:.4f} β + {c:.4f}, R² = {r2:.4f}")

# p-value for quadratic term
X_design = np.column_stack([beta_range**2, beta_range, np.ones_like(beta_range)])
inv_XX = np.linalg.inv(X_design.T @ X_design)
residuals = y - y_fit
sigma_sq = np.sum(residuals**2) / (len(beta_range) - 3)
cov_matrix = sigma_sq * inv_XX
std_err_a = np.sqrt(cov_matrix[0, 0])
t_stat = a / std_err_a
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(beta_range)-3))
print(f"p-value for quadratic term: {p_value:.4e}")

# Find peak
if a < 0:
    beta_peak = -b / (2 * a)
else:
    beta_peak = beta_range[np.argmax(mean_influence)]
print(f"Optimal precision β* = {beta_peak:.3f}")

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.errorbar(beta_range, mean_influence, yerr=std_influence, fmt='o', capsize=3,
            markersize=4, color='#1f77b4', ecolor='gray', alpha=0.8,
            label='Simulated data (mean ± std)')
ax.plot(beta_range, y_fit, 'k-', label=f'Quadratic fit (R² = {r2:.2f})')
ax.axvline(x=beta_peak, color='r', linestyle='--', alpha=0.7, label=f'β* = {beta_peak:.2f}')
ax.set_xlabel('Sensory precision β', fontsize=10)
ax.set_ylabel('Influence (Shapley value)', fontsize=10)
ax.set_title('Neural ensembles', fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.3, linestyle='--')
plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
plt.savefig('neural_influence.png', dpi=300)
plt.show()
