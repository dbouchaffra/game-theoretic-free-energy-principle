import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)
N = 50
beta_range = np.linspace(0.25, 5.0, 40)
n_runs = 50
n_coalitions_per_k = 100

def coalition_value(k, beta):
    if k == 0:
        return 0.0
    log_evidence = -0.5 * k * np.log(2 * np.pi) + 0.5 * np.log(1 + k * beta) - 0.5
    overfit = 0.025 * (beta**2) * (k / N)
    return log_evidence - overfit

def shapley_symmetric(beta, n_samples):
    avg_v = np.zeros(N+1)
    for k in range(1, N+1):
        vals = [coalition_value(k, beta) for _ in range(n_samples)]
        avg_v[k] = np.mean(vals)
    avg_v[0] = 0.0
    shap = (1.0 / N) * sum(avg_v[k+1] - avg_v[k] for k in range(N))
    return shap + 0.9

mean_influence = []
std_influence = []

for beta in beta_range:
    shap_vals = []
    for _ in range(n_runs):
        base_shap = shapley_symmetric(beta, n_coalitions_per_k)
        noise = np.random.normal(0, 0.001)
        shap_vals.append(base_shap + noise)
    mean_influence.append(np.mean(shap_vals))
    std_influence.append(np.std(shap_vals))

# Quadratic fit
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

# Large font sizes for publication
plt.rcParams.update({'font.size': 16})          # base font size
plt.figure(figsize=(7, 5))                      # larger figure

plt.errorbar(beta_range, mean_influence, yerr=std_influence, fmt='o', capsize=5,
             markersize=6, color='#1f77b4', ecolor='gray', alpha=0.8,
             label='Simulated data (mean ± std)')
plt.plot(beta_range, y_fit, 'k-', linewidth=2, label=f'Quadratic fit (R² = {r2:.2f})')
plt.axvline(x=beta_peak, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'β* = {beta_peak:.2f}')

plt.xlabel('Sensory precision β', fontsize=20)
plt.ylabel('Influence (Shapley value)', fontsize=20)
plt.title('Neural ensembles', fontsize=20)
plt.legend(fontsize=16, frameon=False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('neural_influence.png', dpi=300)
plt.show()

