# Module 2 — Analytics Pipeline

## Overview

This module implements a complete Titanic analytics and predictive-modeling
pipeline.

The workflow is intentionally cohesive:

1. Load Titanic once through Seaborn.
2. Immediately save the loaded dataset to `titanic.csv`.
3. Profile and clean the dataset.
4. Perform exploratory data analysis.
5. Save the cleaned hand-off dataset to `titanic.csv`.
6. Read the same CSV in the modeling stage.
7. Perform a stratified train/test split.
8. Fit preprocessing only on training data.
9. Train Logistic Regression, Decision Tree and Random Forest classifiers.
10. Evaluate all classifiers.
11. Compare imbalance strategies.
12. Tune Random Forest with GridSearchCV.
13. Perform multivariate fare regression.
14. Save the complete best classification pipeline with joblib.
15. Reload the saved pipeline and verify raw-input prediction.

The raw Titanic dataset is loaded with `sns.load_dataset("titanic")` exactly
once, in `01_eda.py`.

`02_modeling.py` does not call `sns.load_dataset()`.

---

# Dataset

The dataset is the classic Seaborn Titanic dataset.

The first execution of `01_eda.py` may require internet access because Seaborn's
dataset loader retrieves the example dataset from the Seaborn data repository.
The resulting CSV is committed so the modeling stage and grading can proceed
without another network request.

---

# Running

From the project root:

```bash
cd analytics
```
