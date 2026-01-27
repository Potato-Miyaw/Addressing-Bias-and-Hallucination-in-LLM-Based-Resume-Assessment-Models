# Notes: task_xgboost_implementation.ipynb

## High-level purpose
End-to-end XGBoost + fairness workflow for resume screening bias analysis with JD-resume feature engineering and statistical testing.

## Key datasets
- Resume dataset: mdtalhask/ai-powered-resume-screening-dataset-2025 (KaggleHub).
- Job description dataset: kshitizregmi/jobs-and-job-description (KaggleHub).

## Main steps in the notebook
1) Install dependencies (kagglehub, xgboost, fairlearn, aif360, scikit-learn, pandas, numpy, matplotlib, statsmodels).
2) Load datasets and preprocess text and numeric fields.
3) Feature engineering, including TF-IDF similarity between resumes and job descriptions and skills matching features.
4) Train baseline XGBoost classifier with hyperparameter tuning (GridSearchCV).
5) Apply fairness interventions:
   - Pre-processing: AIF360 Reweighing.
   - In-processing: Fairlearn ExponentiatedGradient (Demographic Parity).
   - Post-processing: Fairlearn ThresholdOptimizer.
6) Evaluate accuracy and fairness metrics.
7) Statistical comparison using McNemar's test.

## Outputs / artifacts
- Results saved to task1_results_with_jd.csv (per notebook).

## Notes
- This notebook is the preferred XGBoost workflow for the project.

