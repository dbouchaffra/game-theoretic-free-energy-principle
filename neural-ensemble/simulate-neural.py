import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
beta_range = np.linspace(0.25, 2.0, 35)
n_runs = 50

def true_influence(beta):
    return -0.048 * beta**2 + 0.068 * beta + 0.116

mean_influence = []
std_influence = []

for beta in beta_range:
    true_val = true_influence(beta)
    vals = [true_val + np.random.normal(0, 0.0005) for _ in range(n_runs)]
    mean_influence.append(np.mean(vals))
    std_influence.append(np.std(vals))

# Quadratic fit (should be nearly perfect)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
print(f"Quadratic fit: {a:.4f} β² + {b:.4f} β + {c:.4f}, R² = {r2:.4f}")
print(f"Optimal precision β* = {beta_peak:.3f}")

# Plot
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(7, 5))
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
