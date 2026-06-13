"""
Grid Search vs Random Search — Where They Differ
=================================================
Dataset : California Housing (regression, 20,000+ samples, noisy)
Model   : Random Forest
Goal    : Show that Random Search finds better hyperparameters
          than Grid Search when the search space is wide/continuous.

Key insight: Grid Search is stuck to fixed points on a grid.
             Random Search can land anywhere — including between grid points.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score
from scipy.stats import randint, uniform

# ── Data ──────────────────────────────────────────────────────────────────────
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 1. Grid Search — coarse fixed grid ────────────────────────────────────────
# Only 3×3×3 = 27 combinations. Values are hand-picked and spread far apart.
grid_params = {
    "n_estimators":   [50, 100, 200],
    "max_depth":      [5, 10, 20],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=grid_params,
    cv=3,
    scoring="r2",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
grid_r2 = r2_score(y_test, grid_search.predict(X_test))

# ── 2. Random Search — wide continuous ranges, same budget ────────────────────
# Also 27 iterations, but C can land on 87, 143, 176 ... not just 50/100/200.
random_params = {
    "n_estimators":      randint(50, 300),      # any integer 50–300
    "max_depth":         randint(3, 25),         # any integer 3–25
    "min_samples_split": randint(2, 20),         # any integer 2–20
    "max_features":      uniform(0.3, 0.7),      # any float 0.3–1.0
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=random_params,
    n_iter=27,          # same number of fits as Grid Search for fair comparison
    cv=3,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
)
random_search.fit(X_train, y_train)
random_r2 = r2_score(y_test, random_search.predict(X_test))

# ── Results ───────────────────────────────────────────────────────────────────
print("=" * 55)
print(f"{'':30} {'Grid':>10} {'Random':>10}")
print("=" * 55)
print(f"{'Combinations tried':<30} {len(grid_search.cv_results_['params']):>10} {len(random_search.cv_results_['params']):>10}")
print(f"{'Best CV R² score':<30} {grid_search.best_score_:>10.4f} {random_search.best_score_:>10.4f}")
print(f"{'Test R² score':<30} {grid_r2:>10.4f} {random_r2:>10.4f}")
print("=" * 55)

print("\nGrid best params :", grid_search.best_params_)
print("Random best params:", random_search.best_params_)

winner = "Random Search" if random_r2 > grid_r2 else "Grid Search"
diff   = abs(random_r2 - grid_r2)
print(f"\n→ {winner} wins by {diff:.4f} R² points")
print(  "  (Random Search explored fractional values Grid never considered)")
