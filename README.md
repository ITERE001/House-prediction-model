# House Price Prediction with scikit-learn

Predicts residential house sale prices using regularized linear regression on the Ames Housing dataset (Kaggle "House Prices: Advanced Regression Techniques", 1,460 homes, 80 features).

## Approach
- **Cleaning:** treated missing values that mean "feature absent" (e.g. no pool, no garage) as their own category, and imputed the rest (neighborhood median for `LotFrontage`, mode for `Electrical`).
- **Encoding:** one-hot encoded categorical features, including numeric codes that are really categories (`MSSubClass`, `MoSold`).
- **Target transform:** applied `log1p` to `SalePrice` to handle its right skew, which turned Linear Regression from R² -0.07 into R² 0.84.
- **Models:** Linear Regression, Ridge and Lasso in `Pipeline`s with `StandardScaler`, so scaling is fit on training data only (no leakage).
- **Tuning:** `GridSearchCV` (5-fold CV) over a log-spaced grid of `alpha`.

## Results (log-price scale, 80/20 train/test split)

| Model | RMSE | R² |
|---|---|---|
| Ridge | 0.155 | 0.872 |
| Linear Regression | 0.171 | 0.844 |
| Lasso | _fill in after re-tuning_ | _fill in_ |

An RMSE of 0.155 in log units means predictions are typically within about 15% of the true price.

## Lessons learned
- Regularization strength has to be re-tuned when the target scale changes: an alpha tuned on raw dollars zeroes out every Lasso coefficient on a log target.
- Predictions are converted back to dollars with `np.expm1`.

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter

## Run it
1. Download `train.csv` from the Kaggle competition page.
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn jupyter`
3. Open `House_predictions_sciit_learn.ipynb` and run all cells.
