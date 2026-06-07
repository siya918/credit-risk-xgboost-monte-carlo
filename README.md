# credit-risk-xgboost-monte-carlo
XGBoost credit risk classifier with Monte Carlo uncertainty quantification.  Achieves AUC 0.947 across 5-fold CV [0.942, 0.953]. Features manual feature  engineering, Youden's J threshold optimisation, and a three-tier lending  decision framework built in R.


# Credit Risk Modelling — XGBoost with Monte Carlo Uncertainty Quantification

## Overview
End-to-end credit risk modelling pipeline built in R on 32,581 loan records. 
The goal is to predict loan default probability and translate model outputs 
into a defensible three-tier lending decision system — Auto Approve, Manual 
Review, and Auto Reject — using Monte Carlo uncertainty quantification to 
identify unreliable predictions.

---

## Dataset
- **Source:** Credit Risk Dataset (Kaggle)
- **Records:** 32,581 loan applications
- **Features:** 12 variables including borrower demographics, loan 
  characteristics, and credit history
- **Target:** `loan_status` (0 = No Default, 1 = Default)
- **Class distribution:** 77.9% No Default / 22.1% Default

---

## Pipeline Summary

### 1. Exploratory Data Analysis
- Univariate distributions via histograms and density plots
- Bivariate analysis — boxplots and scatterplots split by default status
- Correlation matrix and variable ranking by association with default
- Cross-tabulation of categorical variables against default status
- Missing value analysis: `loan_int_rate` (9.56%), `person_emp_length` (2.75%)

### 2. Data Engineering
- `OTHER` home ownership (107 obs, 0.3%) merged into `RENT` based on 
  economic similarity — both groups lack ownership stake
- Loan grades `E`, `F`, `G` binned into single subprime category `E` — 
  combined `F` and `G` represented only 0.9% of observations
- Skewness analysis confirmed all variables below 1.2 — log transformation 
  unnecessary as XGBoost is invariant to monotonic transformations

### 3. Feature Encoding
| Variable | Type | Encoding |
|---|---|---|
| `loan_grade` | Ordinal | A=1, B=2, C=3, D=4, E=5 |
| `cb_person_default_on_file` | Binary | N=0, Y=1 |
| `person_home_ownership` | Nominal | One-hot, RENT as reference |
| `loan_intent` | Nominal | One-hot, EDUCATION as reference |

### 4. Missing Value Imputation
Mean imputation applied to `person_emp_length` and `loan_int_rate`. 
Both variables exhibit mild skewness below 1.0 making mean and median 
practically equivalent.

### 5. Modelling — XGBoost
```r
params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.1,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  gamma            = 1
)
```
- Early stopping triggered at round 290
- Train AUC: 0.9837 | Eval AUC: 0.9492

### 6. Cross Validation
5-fold stratified cross validation for robust performance estimation:

| Metric | Value |
|---|---|
| CV AUC Mean | 0.9474 |
| CV AUC Std Dev | 0.0029 |
| 95% Confidence Interval | [0.9417, 0.9531] |
| Test AUC | 0.9494 |

Test AUC falls within the CV confidence interval confirming generalisation.

### 7. Threshold Optimisation
Youden's J statistic applied to identify the optimal decision threshold:

| Threshold | Sensitivity | Specificity | False Negatives |
|---|---|---|---|
| 0.500 (default) | 74.7% | 98.9% | 365 |
| 0.300 | 79.4% | 96.3% | 297 |
| 0.192 (Youden) | 84.6% | 91.8% | 222 |

Threshold of **0.192** selected as operational boundary — minimises missed 
defaults while maintaining acceptable specificity for credit decisions.

### 8. Feature Importance
Top predictors by XGBoost Gain:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `loan_percent_income` | 21.9% |
| 2 | `loan_grade_encoded` | 20.2% |
| 3 | `person_income` | 16.6% |
| 4 | `home_MORTGAGE` | 8.95% |
| 5 | `loan_amnt` | 5.5% |

Top 3 features drive 58.7% of the model. Loan burden relative to income 
is the single strongest default signal.

### 9. Monte Carlo Uncertainty Quantification
1,000 simulations with Gaussian perturbations (sd=0.01) applied to test 
features to quantify prediction stability per observation:

| Metric | Value |
|---|---|
| Average prediction std dev | 0.082 |
| Max prediction std dev | 0.495 |
| High uncertainty observations | 652 (10%) |
| Default rate in uncertain group | 49.5% |
| Threshold uncertainty zone (0.15-0.25) | 499 obs (7.66%) |

High uncertainty observations show a 49.5% default rate — effectively a 
coin flip — confirming the model cannot reliably classify these cases 
automatically.

### 10. Three-Tier Decision System

| Decision | Count | % | Criteria |
|---|---|---|---|
| Auto Approve | 4,353 | 66.8% | Predicted probability < 0.15 |
| Manual Review | 499 | 7.66% | Predicted probability 0.15 — 0.25 |
| Auto Reject | 1,665 | 25.6% | Predicted probability > 0.25 |

10.1% of applications additionally flagged for underwriter review based 
on high prediction variance across Monte Carlo simulations.

### 11. Calibration Analysis
Mean absolute calibration error of 0.0397 across probability deciles. 
Model well calibrated in low (0.0-0.3) and high (0.6-1.0) probability 
ranges. Mild overconfidence observed in 0.3-0.5 range (max bin error 
0.155) — this region falls above the operational threshold of 0.192 
limiting practical impact.

---

## Results Summary

| Metric | Value |
|---|---|
| CV AUC (5-fold) | 0.947 [0.942, 0.953] |
| Test AUC | 0.949 |
| Sensitivity at optimal threshold | 84.6% |
| Specificity at optimal threshold | 91.8% |
| Kappa | 0.725 |
| Auto-decidable applications | 92.3% |
| Applications requiring manual review | 7.7% |

---

## Repository Structure
