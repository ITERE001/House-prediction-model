# Ames House Price Prediction

Predicting residential sale prices on the Ames Housing dataset with scikit-learn, comparing **Ridge**, **Lasso** and **Ordinary Least Squares** regression. The project covers missing-value handling, categorical encoding, a log-transformed target, scaling inside pipelines, and hyperparameter tuning with cross-validated grid search.

## Results

Evaluated on a held-out 20% test set (292 houses). Errors are in **log-price units**, because the models predict `log1p(SalePrice)`.

| Model | Tuned alpha | RMSE | MAE | R² |
|---|---|---|---|---|
| **Lasso** | 0.01 | **0.151** | 0.102 | **0.878** |
| Ridge | 483 | 0.155 | 0.099 | 0.872 |
| Linear Regression | n/a | 0.171 | 0.096 | 0.844 |

An RMSE of about 0.15 in log units corresponds to a typical error of roughly 15% of the house price.

**Takeaways**

- Lasso and Ridge perform comparably. The gap between them is small and within what a different train/test split could change, so neither is a decisive winner.
- Lasso keeps **57 of 282 features** and sets the rest to exactly zero, giving a much simpler model with accuracy at least on par with Ridge.
- Plain Linear Regression is the weakest on RMSE and R², which is the case for regularization on
