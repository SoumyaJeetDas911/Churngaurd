# ChurnGuard Model Card

## Model details

- **Model:** Selected from Logistic Regression, Random Forest, and XGBoost by
  validation ROC-AUC; the selected model and measured metrics are recorded by
  each training run in `models/metadata.json` and `artifacts/training/metrics.json`.
- **Version:** Generated at training time and stored with the model artifact.
- **Input:** Customer attributes in the exact schema recorded in the artifact.
- **Output:** Estimated probability of the observed churn label, a class at the
  stored decision threshold, risk band, optional feature attribution, and
  rule-based suggestions.

## Intended use

This is an educational decision-support prototype for aggregate analysis and
human-reviewed prioritization. It is not intended for automated account
decisions, denial of service, pricing, or claims that an individual will churn.
Retention suggestions are hypotheses for business testing, not proven
interventions.

## Data provenance and processing

The checked-in raw file is `Data/Raw/Copy of Telco_customer_churn.csv`, a
7,043-row Telco customer churn dataset. With explicit approval, this project
uses `Churn Label` (`Yes`/`No`) as its target and converts `Total Charges` to
numeric. The raw file is not changed. `CustomerID`, `Churn Value` (duplicate
label), `Churn Score` (outcome-related score), `Churn Reason` (post-outcome),
`CLTV` (potential outcome leakage), and geographic fields are excluded.
Geographic fields are excluded to reduce location-based privacy and
generalization risks; `Count` is excluded as a record-count field. Any constant
feature is also identified and excluded from training.

## Evaluation

Model selection uses a stratified validation split and ROC-AUC. The held-out
test split is evaluated once after selecting the model. Each run stores actual
validation comparisons and held-out test ROC-AUC, PR-AUC, precision, recall,
F1, accuracy, confusion matrix, and classification report. No score is
pre-stated; inspect the generated report for the measured values from the
current data and software environment.

The default decision threshold is 0.5 and is not a universal business choice.
Risk bands are display groupings, not calibrated policy categories. Probability
calibration and threshold utility should be assessed with representative,
current data and business costs before operational use.

## Limitations and risks

- Historical data may be stale, non-representative, or reflect past business
  practices and structural inequities.
- False positives can cause unnecessary outreach; false negatives can miss
  customers who later leave. Neither error should automatically trigger a
  customer-impacting action.
- Feature attributions describe model contribution, not causation.
- Unknown categories are accepted by preprocessing but may have limited
  training support.
- No fairness, subgroup calibration, prospective, or causal intervention
  analysis is claimed.
- The dataset's licensing and redistribution terms should be verified by the
  user before sharing the CSV or derived records.

## Privacy and security

The prototype stores prediction input attributes in local SQLite history so
results can be reviewed. Use synthetic or authorized data only. Before real
customer data is used, add access control, encryption, retention/deletion
policies, audit logging, input minimization, secure deployment, and a privacy
review. The application is not production-secure.
