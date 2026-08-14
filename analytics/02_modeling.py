# pathlib is used to work with file and folder paths
# in a way that works across different operating systems.
from pathlib import Path

# warnings allows us to control warning messages displayed
# while running the program.
import warnings

# joblib is used to save and reload the trained machine-learning
# pipeline as a file.
import joblib

# NumPy provides numerical operations used later for calculations.
import numpy as np

# pandas is used for loading, cleaning, transforming and analysing
# tabular data.
import pandas as pd

# seaborn and matplotlib are used to create charts and visualisations.
import seaborn as sns
import matplotlib.pyplot as plt


# SMOTE is used to create synthetic examples of the minority class
# when the target classes are imbalanced.
from imblearn.over_sampling import SMOTE

# ImbPipeline allows SMOTE to be included safely inside a machine-learning
# pipeline so that it is applied only to training data.
from imblearn.pipeline import Pipeline as ImbPipeline


# ColumnTransformer allows us to apply different preprocessing steps
# to different groups of columns.
from sklearn.compose import ColumnTransformer

# SimpleImputer is used to fill missing values.
from sklearn.impute import SimpleImputer

# OneHotEncoder converts categorical text values into numeric columns.
# StandardScaler standardizes numeric values.
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

# Pipeline combines preprocessing and the machine-learning model
# into one complete workflow.
from sklearn.pipeline import Pipeline


# train_test_split divides the data into training and testing sets.
# GridSearchCV searches for the best model hyperparameters.
# StratifiedKFold creates balanced cross-validation folds.
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)


# Classification models
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
)

from sklearn.ensemble import RandomForestClassifier


# Evaluation metrics for classification and regression.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# Configuration
# ============================================================

# Find the folder where this Python script is located.
# This allows the program to create files relative to the script.
BASE_DIR = Path(__file__).resolve().parent

# Input CSV file containing the cleaned Titanic dataset.
CSV_PATH = BASE_DIR / "titanic.csv"

# Folder where charts will be saved.
CHART_DIR = BASE_DIR / "charts"

# Folder where text files and model results will be saved.
OUTPUT_DIR = BASE_DIR / "outputs"

# Folder where trained model files will be saved.
ARTIFACT_DIR = BASE_DIR / "artifacts"


# Create the required folders if they do not already exist.
# exist_ok=True means the program will not fail if the folder
# already exists.
CHART_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


# Location where the final trained machine-learning pipeline
# will be stored.
MODEL_PATH = (
    ARTIFACT_DIR
    / "titanic_pipeline.joblib"
)


# A fixed random state makes the results reproducible.
# Using the same value allows us to get the same train/test split
# and similar model results when the program is run again.
RANDOM_STATE = 42


# Set a consistent visual style for all seaborn charts.
sns.set_theme(style="whitegrid")

# Ignore warning messages so that the output is easier to read.
# This does not change the actual model calculations.
warnings.filterwarnings("ignore")


# ============================================================
# Load cleaned CSV
#
# NO sns.load_dataset() HERE.
# ============================================================

print("=" * 80)
print("MODULE 2 — MODELING PIPELINE")
print("=" * 80)

print("\n[1] Loading cleaned Titanic CSV...")


# Read the cleaned Titanic CSV file into a pandas DataFrame.
#
# The modeling stage uses the CSV created during the EDA stage.
# This is important because the modeling pipeline should work
# from the cleaned hand-off dataset.
df = pd.read_csv(CSV_PATH)


print(
    f"Loaded {len(df)} rows and {len(df.columns)} columns."
)


# ============================================================
# Classification features
# ============================================================

# "survived" is the value that we want the classification models
# to predict.
#
# 0 = passenger did not survive
# 1 = passenger survived
target = "survived"


# These columns will be used as input features for classification.
#
# They contain information that may help predict survival.
classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]


# X contains the input features.
X = df[classification_features].copy()

# y contains the target/output that the model needs to predict.
y = df[target].copy()


# ============================================================
# 7. STRATIFIED TRAIN/TEST SPLIT
# ============================================================

print("\n[2] Stratified train/test split...")


# Display how many passengers belong to each target class.
print("\nOverall class balance:")

class_balance = (
    y.value_counts(normalize=False)
    .sort_index()
)

print(class_balance)


# Calculate the percentage of each target class.
class_balance_pct = (
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

print("\nClass percentages:")
print(class_balance_pct)


# ------------------------------------------------------------
# Split the dataset
# ------------------------------------------------------------

# Split the data into:
#
# 80% training data
# 20% testing data
#
# random_state makes the split reproducible.
#
# stratify=y ensures that the proportion of survivors and
# non-survivors is approximately the same in both datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)


print(
    f"\nTraining rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)


# Stratification is useful because the target classes are not
# perfectly balanced.
#
# Without stratification, the random split could accidentally
# create noticeably different class proportions in the train
# and test datasets.
print(
    "\nStratification is used because survival classes are not "
    "perfectly balanced. It preserves approximately the same "
    "survivor/non-survivor proportions in both train and test."
)


# ============================================================
# 8. TRAIN-ONLY PREPROCESSING
# ============================================================

# Separate numeric columns from categorical columns.
#
# Numeric columns can be imputed and scaled using numerical
# preprocessing methods.
numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]


# These columns contain categorical/text values.
categorical_features = [
    "sex",
    "embarked",
]


# ------------------------------------------------------------
# Numeric preprocessing pipeline
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            # Replace missing numeric values with the median.
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            # Standardize numeric columns so that they have
            # approximately mean 0 and standard deviation 1.
            "scaler",
            StandardScaler(),
        ),
    ]
)


# ------------------------------------------------------------
# Categorical preprocessing pipeline
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            # Replace missing categorical values with the
            # most frequently occurring category.
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            # Convert categories such as male/female into
            # numeric indicator columns.
            #
            # handle_unknown="ignore" prevents an error if the
            # test data contains a category not seen during training.
            #
            # sparse_output=False produces a normal dense array.
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


# ------------------------------------------------------------
# Combine numeric and categorical preprocessing
# ------------------------------------------------------------

# ColumnTransformer allows us to send numeric columns through
# numeric_pipeline and categorical columns through
# categorical_pipeline.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ============================================================
# Helper: classifier evaluation
# ============================================================

def evaluate_classifier(
    name,
    model,
    X_test,
    y_test,
):
    """
    Evaluate a trained classification model.

    The function calculates several common classification metrics:
    accuracy, precision, recall, F1 score and ROC-AUC.

    It also returns the confusion matrix and predicted probabilities.
    """

    # Generate the final class prediction.
    #
    # Example:
    # 0 = predicted not survived
    # 1 = predicted survived
    y_pred = model.predict(X_test)


    # predict_proba() gives probabilities for each class.
    #
    # [:, 1] selects the probability of class 1,
    # which represents survival.
    y_probability = model.predict_proba(
        X_test
    )[:, 1]


    # Accuracy = percentage of all predictions that were correct.
    accuracy = accuracy_score(
        y_test,
        y_pred,
    )


    # Precision answers:
    # "Of the passengers predicted to survive,
    # how many actually survived?"
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )


    # Recall answers:
    # "Of all passengers who actually survived,
    # how many did the model identify?"
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )


    # F1 combines precision and recall into one score.
    #
    # F1 is particularly useful when the target classes
    # are not perfectly balanced.
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )


    # ROC-AUC measures how well the model separates the
    # two classes across different probability thresholds.
    auc = roc_auc_score(
        y_test,
        y_probability,
    )


    # Confusion matrix shows:
    #
    # True Negatives
    # False Positives
    # False Negatives
    # True Positives
    cm = confusion_matrix(
        y_test,
        y_pred,
    )


    # Return all important evaluation information in a dictionary.
    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "y_probability": y_probability,
    }


# ============================================================
# 9. THREE CLASSIFIERS
# ============================================================

# Create three different classification models.
#
# Each model includes the same preprocessing pipeline followed
# by a different machine-learning algorithm.
models = {

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    "Logistic Regression": Pipeline(
        steps=[
            (
                # First preprocess the input data.
                "preprocessor",
                preprocessor,
            ),
            (
                # Then train Logistic Regression.
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),


    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------

    "Decision Tree": Pipeline(
        steps=[
            (
                # Preprocess the data first.
                "preprocessor",
                preprocessor,
            ),
            (
                # Train a Decision Tree.
                #
                # max_depth limits how deep the tree can grow.
                # This helps reduce overfitting.
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    "Random Forest": Pipeline(
        steps=[
            (
                # Preprocess the data first.
                "preprocessor",
                preprocessor,
            ),
            (
                # Random Forest combines many decision trees
                # to improve predictive performance.
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# Store the evaluation results for all models.
results = []

# Store confusion matrices separately so they can later
# be displayed together in one chart.
confusion_matrices = {}

# Store ROC information for later plotting.
roc_data = {}


# Train and evaluate every classifier.
for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    # Fit means:
    # - learn preprocessing parameters from training data
    # - train the machine-learning model
    model.fit(
        X_train,
        y_train,
    )


    # Evaluate the trained model using the untouched test data.
    result = evaluate_classifier(
        name,
        model,
        X_test,
        y_test,
    )


    # Save the complete evaluation result.
    results.append(result)


    # Save the confusion matrix for plotting.
    confusion_matrices[name] = (
        result["confusion_matrix"]
    )


    # Save true labels, probabilities and AUC
    # for creating ROC curves later.
    roc_data[name] = (
        y_test,
        result["y_probability"],
        result["auc"],
    )


# ============================================================
# Classification metrics table
# ============================================================

# Convert the model results into a DataFrame.
#
# This makes it easier to compare the models side by side.
metrics_df = pd.DataFrame(
    [
        {
            "model": r["model"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "auc": r["auc"],
        }
        for r in results
    ]
)


print("\nClassification model comparison:")
print(metrics_df)


# Save the model comparison table as a CSV file.
metrics_df.to_csv(
    OUTPUT_DIR / "model_metrics.csv",
    index=False,
)


# ============================================================
# Confusion matrices
# ============================================================

# Create three plots side by side.
#
# There are three models, so we need three axes.
fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4),
)


# Add one confusion matrix to each subplot.
for ax, (name, cm) in zip(
    axes,
    confusion_matrices.items(),
):

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


# Adjust spacing between the plots.
plt.tight_layout()


# Save the confusion matrix figure.
plt.savefig(
    CHART_DIR / "confusion_matrices.png",
    dpi=150,
)

# Close the figure to free memory.
plt.close()


# ============================================================
# ROC curves
# ============================================================

# Create a new figure for the ROC curves.
plt.figure(figsize=(8, 6))


# Create one ROC curve for each classifier.
for name, (
    y_true,
    probabilities,
    auc_value,
) in roc_data.items():

    # Calculate false-positive rate and true-positive rate
    # at different classification thresholds.
    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities,
    )


    # Plot the ROC curve.
    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc_value:.3f})",
    )


# A random classifier is represented by this diagonal line.
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
)


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()

plt.tight_layout()


# Save the ROC chart.
plt.savefig(
    CHART_DIR / "roc_curves.png",
    dpi=150,
)

plt.close()


# ============================================================
# Decision Tree visualization
# ============================================================

# Select the trained Decision Tree pipeline.
tree_model = models["Decision Tree"]


# Get the preprocessing part of the pipeline.
tree_preprocessor = (
    tree_model
    .named_steps["preprocessor"]
)


# Get the actual Decision Tree classifier.
tree_classifier = (
    tree_model
    .named_steps["classifier"]
)


# After preprocessing, categorical columns are converted
# into additional one-hot encoded columns.
#
# Therefore, we need the transformed feature names to
# correctly label the tree visualization.
feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)


# Create a large figure because decision trees can contain
# many nodes.
plt.figure(
    figsize=(24, 12)
)


# Draw the trained Decision Tree.
plot_tree(
    tree_classifier,
    feature_names=feature_names,
    class_names=[
        "Did Not Survive",
        "Survived",
    ],
    filled=True,
    rounded=True,
    fontsize=8,
)


plt.title(
    "Decision Tree Classifier"
)

plt.tight_layout()


# Save the decision-tree visualization.
plt.savefig(
    CHART_DIR / "decision_tree.png",
    dpi=150,
)

plt.close()


# ============================================================
# 11. IMBALANCE COMPARISON
# ============================================================

print("\n[3] Imbalance comparison...")


# We will compare three different approaches to handling
# the class imbalance.
imbalance_models = {}


# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

# Baseline Logistic Regression does not apply any special
# class-balancing technique.
imbalance_models["Baseline"] = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ------------------------------------------------------------
# class_weight balanced
# ------------------------------------------------------------

# class_weight="balanced" automatically gives more importance
# to the minority class during model training.
#
# It does NOT create new rows.
imbalance_models["class_weight=balanced"] = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ------------------------------------------------------------
# SMOTE
# ------------------------------------------------------------

# SMOTE creates synthetic examples of the minority class.
#
# Important:
# SMOTE is placed inside the pipeline so that it is performed
# only on training data.
#
# The test set must remain untouched because it should represent
# real unseen data.
imbalance_models["SMOTE"] = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# List used to store results for the three imbalance strategies.
imbalance_results = []


# Train and evaluate every imbalance strategy.
for name, model in imbalance_models.items():

    # Train using only the training data.
    model.fit(
        X_train,
        y_train,
    )


    # Make predictions on the untouched test set.
    prediction = model.predict(
        X_test,
    )


    # Calculate precision, recall and F1.
    imbalance_results.append(
        {
            "strategy": name,
            "precision": precision_score(
                y_test,
                prediction,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                prediction,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                prediction,
                zero_division=0,
            ),
        }
    )


# Convert the results into a DataFrame.
imbalance_df = pd.DataFrame(
    imbalance_results
)


print(
    "\nImbalance comparison:"
)

print(imbalance_df)


# Save the imbalance comparison.
imbalance_df.to_csv(
    OUTPUT_DIR / "imbalance_comparison.csv",
    index=False,
)


# ------------------------------------------------------------
# Determine best imbalance strategy
# ------------------------------------------------------------

# Sort strategies by F1 score from highest to lowest.
#
# The first row will therefore contain the best F1 score.
best_imbalance = (
    imbalance_df
    .sort_values(
        "f1",
        ascending=False,
    )
    .iloc[0]
)


# Create a written conclusion explaining which strategy performed best.
imbalance_conclusion = f"""
Imbalance comparison conclusion
================================

The best strategy by F1 score was:
{best_imbalance['strategy']}

Precision: {best_imbalance['precision']:.4f}
Recall:    {best_imbalance['recall']:.4f}
F1:        {best_imbalance['f1']:.4f}

The class-weighted and SMOTE variants are useful because the Titanic target
contains more non-survivors than survivors. Class weighting changes the
training loss without creating synthetic observations, whereas SMOTE
creates synthetic minority examples. SMOTE was applied only inside the
training pipeline, so the test set remains untouched and provides an
unbiased evaluation.
"""


print(imbalance_conclusion)


# ============================================================
# 12. RANDOM FOREST GRID SEARCH
# ============================================================

print("\n[4] Random Forest GridSearchCV...")


# Create a Random Forest pipeline.
#
# The preprocessing is performed before the Random Forest model.
rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            RandomForestClassifier(
                # OOB = Out Of Bag.
                #
                # It allows us to calculate an additional
                # performance estimate using observations that
                # were not selected for particular trees.
                oob_score=True,

                # Bootstrap sampling means each tree receives
                # a random sample of the training data.
                bootstrap=True,

                random_state=RANDOM_STATE,

                # Use all available CPU cores where possible.
                n_jobs=-1,
            ),
        ),
    ]
)


# ------------------------------------------------------------
# Hyperparameter grid
# ------------------------------------------------------------

# GridSearchCV will test different combinations of these values.
#
# n_estimators = number of trees in the forest.
#
# max_depth = maximum depth of each tree.
#
# max_features = number of features considered when splitting
# each node.
param_grid = {
    "classifier__n_estimators": [
        100,
        200,
        300,
    ],

    "classifier__max_depth": [
        None,
        5,
        10,
        15,
    ],

    "classifier__max_features": [
        "sqrt",
        "log2",
    ],
}


# ------------------------------------------------------------
# Stratified cross-validation
# ------------------------------------------------------------

# Divide the training data into five folds.
#
# Stratification helps preserve the same target-class proportion
# in every fold.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ------------------------------------------------------------
# GridSearchCV
# ------------------------------------------------------------

# GridSearchCV tries every hyperparameter combination and
# evaluates each one using cross-validation.
#
# F1 is selected because it balances precision and recall.
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
)


# Run the complete hyperparameter search using training data.
grid_search.fit(
    X_train,
    y_train,
)


# Best combination of hyperparameters found by GridSearchCV.
best_params = grid_search.best_params_


# Best average F1 score across the cross-validation folds.
best_cv_score = grid_search.best_score_


# GridSearchCV automatically refits the best model when
# refit=True.
best_rf_pipeline = (
    grid_search.best_estimator_
)


# Extract the Random Forest classifier from the pipeline.
best_rf_classifier = (
    best_rf_pipeline
    .named_steps["classifier"]
)


# Retrieve the Out-Of-Bag score from the fitted Random Forest.
oob_score = best_rf_classifier.oob_score_


print(
    "\nBest Random Forest parameters:"
)

print(best_params)

print(
    f"\nBest cross-validation F1: "
    f"{best_cv_score:.4f}"
)

print(
    f"OOB score: {oob_score:.4f}"
)


# Save the hyperparameter-tuning results to a text file.
hyperparameter_text = f"""
RANDOM FOREST HYPERPARAMETER TUNING
===================================

Scoring metric:
F1

Best parameters:
{best_params}

Best cross-validation F1:
{best_cv_score:.4f}

OOB score:
{oob_score:.4f}

The RandomForestClassifier was explicitly constructed with
oob_score=True and bootstrap=True, allowing the fitted estimator's
oob_score_ attribute to be reported.
"""


with (
    OUTPUT_DIR / "hyperparameter_tuning.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(hyperparameter_text)


# ============================================================
# 13. REGRESSION SIDE TASK
# ============================================================

print("\n[5] Regression — predicting fare...")


# In this section, the problem changes from classification
# to regression.
#
# Instead of predicting survived (0/1), we predict the
# continuous numerical value "fare".
regression_target = "fare"


# Use all available columns initially except the target column.
regression_features = [
    column
    for column in df.columns
    if column != regression_target
]


# Remove columns that are redundant, text-heavy or derived
# representations that are not suitable as raw predictors
# for this regression task.
regression_features = [
    column
    for column in regression_features
    if column not in [
        "alive",
        "class",
        "who",
        "deck",
        "embark_town",
    ]
]


# X_reg contains the regression input features.
X_reg = df[
    regression_features
].copy()


# y_reg contains the fare values we want to predict.
y_reg = df[
    regression_target
].copy()


# Split regression data into 80% training and 20% testing.
#
# Stratification is not required here because fare is a continuous
# numerical target.
X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
)


# Automatically identify numeric regression features.
reg_numeric_features = (
    X_reg_train
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)


# Automatically identify categorical regression features.
reg_categorical_features = (
    X_reg_train
    .select_dtypes(
        include=["object", "category", "bool"]
    )
    .columns
    .tolist()
)


# ------------------------------------------------------------
# Regression numeric preprocessing
# ------------------------------------------------------------

# Missing numeric values are replaced with their median.
#
# StandardScaler then puts the numeric features on a comparable
# scale.
reg_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# ------------------------------------------------------------
# Regression categorical preprocessing
# ------------------------------------------------------------

# Missing categorical values are replaced with the most common
# category.
#
# OneHotEncoder converts categories into numeric columns.
reg_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


# Combine the numeric and categorical preprocessing pipelines.
reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_pipeline,
            reg_numeric_features,
        ),
        (
            "categorical",
            reg_categorical_pipeline,
            reg_categorical_features,
        ),
    ]
)


# Create the complete regression pipeline.
#
# First:
#   preprocess the data
#
# Then:
#   train Linear Regression
regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor,
        ),
        (
            "regressor",
            LinearRegression(),
        ),
    ]
)


# Train the regression model using only the training data.
regression_pipeline.fit(
    X_reg_train,
    y_reg_train,
)


# Predict fare values for the unseen test data.
y_reg_pred = regression_pipeline.predict(
    X_reg_test,
)


# ============================================================
# Regression evaluation metrics
# ============================================================

# MAE = Mean Absolute Error.
#
# It tells us the average absolute difference between
# actual and predicted fare.
mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred,
)


# RMSE = Root Mean Squared Error.
#
# RMSE gives more weight to larger prediction errors.
rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred,
    )
)


# R² measures the proportion of variation in fare
# explained by the regression model.
r2 = r2_score(
    y_reg_test,
    y_reg_pred,
)


# Number of observations in the regression test set.
n = len(y_reg_test)


# ------------------------------------------------------------
# Number of regression coefficients
# ------------------------------------------------------------

# Extract the fitted LinearRegression model.
regressor = (
    regression_pipeline
    .named_steps["regressor"]
)


# len(coef_) tells us how many model coefficients were created
# after preprocessing and one-hot encoding.
p = len(
    regressor.coef_
)


# ------------------------------------------------------------
# Adjusted R²
# ------------------------------------------------------------

# Adjusted R² modifies R² based on the number of predictors.
#
# This is useful when a model contains many features because
# normal R² can increase when additional variables are added.
adjusted_r2 = (
    1
    - (
        (1 - r2)
        * (n - 1)
        / (n - p - 1)
    )
)


# ============================================================
# Residual plot
# ============================================================

# A residual is:
#
# actual value - predicted value
#
# Ideally, residuals should be randomly distributed around zero.
residuals = (
    y_reg_test - y_reg_pred
)


# Create a scatter plot of predicted values versus residuals.
plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=y_reg_pred,
    y=residuals,
    alpha=0.65,
)


# Add a horizontal line at zero.
#
# This makes it easier to see whether residuals are
# systematically above or below zero.
plt.axhline(
    0,
    color="red",
    linestyle="--",
)


plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")

plt.tight_layout()


# Save the residual plot.
plt.savefig(
    CHART_DIR / "regression_residuals.png",
    dpi=150,
)

plt.close()


# ============================================================
# Basic heteroscedasticity diagnostic
# ============================================================

# Convert residuals to absolute values.
#
# We are interested in whether the size of errors changes
# as the predicted fare increases.
absolute_residuals = np.abs(residuals)


# Calculate the correlation between predicted fare and
# absolute residual size.
#
# A stronger relationship can indicate that prediction error
# changes depending on the predicted value.
residual_correlation = np.corrcoef(
    y_reg_pred,
    absolute_residuals,
)[0, 1]


# Use 0.30 as a simple exploratory threshold.
#
# This is a basic diagnostic, not a formal statistical test.
if abs(residual_correlation) >= 0.30:

    heteroscedasticity_conclusion = (
        "The residual plot shows evidence consistent with "
        "heteroscedasticity because the magnitude of residuals "
        "changes systematically with predicted fare."
    )

else:

    heteroscedasticity_conclusion = (
        "The residual plot does not show strong evidence of "
        "heteroscedasticity; residual magnitude appears reasonably "
        "stable across predicted fare values."
    )


# Create a readable summary of the regression results.
regression_text = f"""
FARE REGRESSION
===============

Features used:
{regression_features}

MAE:
{mae:.4f}

RMSE:
{rmse:.4f}

R²:
{r2:.4f}

Adjusted R²:
{adjusted_r2:.4f}

Number of regression coefficients:
{p}

Residual-vs-prediction absolute residual correlation:
{residual_correlation:.4f}

Heteroscedasticity conclusion:
{heteroscedasticity_conclusion}
"""


print(regression_text)


# Save regression results to a text file.
with (
    OUTPUT_DIR / "regression_metrics.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(regression_text)


# ============================================================
# 14. FINAL MODEL COMPARISON
# ============================================================

print("\n[6] Final model comparison...")


# ------------------------------------------------------------
# Select best classifier by F1
# ------------------------------------------------------------

# Sort the three classifiers by F1 score.
#
# The highest F1 score is placed first.
best_classifier_row = (
    metrics_df
    .sort_values(
        "f1",
        ascending=False,
    )
    .iloc[0]
)


# Store the name of the best-performing classifier.
best_classifier_name = (
    best_classifier_row["model"]
)


# ------------------------------------------------------------
# Classification comparison table
# ------------------------------------------------------------

# Make a copy of the classification results.
classification_comparison = metrics_df.copy()


# Rename the columns to make the final output more
# presentation-friendly.
classification_comparison.columns = [
    "Classifier",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC",
]


# ------------------------------------------------------------
# Regression comparison table
# ------------------------------------------------------------

# Create a separate table for regression results.
#
# Classification and regression metrics are kept separate because
# they measure different types of prediction problems.
regression_comparison = pd.DataFrame(
    [
        {
            "Regression Model": (
                "Multivariate Linear Regression"
            ),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted R2": adjusted_r2,
        }
    ]
)


print("\nCLASSIFICATION METRICS")
print(
    classification_comparison.to_string(
        index=False
    )
)


print("\nREGRESSION METRICS")
print(
    regression_comparison.to_string(
        index=False
    )
)


# ============================================================
# Final recommendation
# ============================================================

# Prepare a written explanation of why the selected classifier
# is recommended.
#
# F1 is used as the primary selection metric because it balances
# precision and recall.
recommendation = f"""
FINAL MODEL RECOMMENDATION
==========================

The recommended classifier is {best_classifier_name}, selected because it
achieved the strongest F1 score among the three classifiers on the held-out
test set. Its accuracy was {best_classifier_row['accuracy']:.3f}, precision
was {best_classifier_row['precision']:.3f}, recall was
{best_classifier_row['recall']:.3f}, F1 was {best_classifier_row['f1']:.3f},
and AUC was {best_classifier_row['auc']:.3f}. F1 is particularly useful here
because the survival classes are not perfectly balanced and it considers both
precision and recall. The regression model is reported separately because
MAE, RMSE, R² and Adjusted R² measure a different prediction problem and must
not be interpreted as directly comparable to classification metrics.
"""


print(recommendation)


# Save the final model comparison and recommendation.
with (
    OUTPUT_DIR / "model_comparison.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "CLASSIFICATION METRICS\n"
    )

    f.write(
        classification_comparison.to_string(
            index=False
        )
    )

    f.write(
        "\n\nREGRESSION METRICS\n"
    )

    f.write(
        regression_comparison.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
        + recommendation
    )


# ============================================================
# 15. SAVE COMPLETE BEST PIPELINE
# ============================================================

print("\n[7] Saving complete fitted pipeline...")


# ------------------------------------------------------------
# Select the best classifier
# ------------------------------------------------------------

# The variable final_pipeline will point to the classifier
# that achieved the highest F1 score.
#
# Importantly, these are complete pipelines containing both
# preprocessing and the model.
if best_classifier_name == "Logistic Regression":

    final_pipeline = models[
        "Logistic Regression"
    ]

elif best_classifier_name == "Decision Tree":

    final_pipeline = models[
        "Decision Tree"
    ]

else:

    final_pipeline = models[
        "Random Forest"
    ]


# ------------------------------------------------------------
# Refit the selected pipeline
# ------------------------------------------------------------

# Train the selected final pipeline using the complete
# classification training dataset.
#
# The test set is still kept separate and is not used here.
final_pipeline.fit(
    X_train,
    y_train,
)


# ------------------------------------------------------------
# Save the trained pipeline
# ------------------------------------------------------------

# joblib stores the complete fitted preprocessing + model pipeline.
#
# This is useful because later we can provide raw input data
# directly to the saved pipeline without manually repeating
# preprocessing steps.
joblib.dump(
    final_pipeline,
    MODEL_PATH,
)


print(
    f"Saved complete pipeline to: "
    f"{MODEL_PATH}"
)


# ============================================================
# Reload and verify raw-input prediction
# ============================================================

print(
    "\nReloading saved pipeline..."
)


# Load the saved model back into memory.
#
# This checks that the saved model can actually be reused.
loaded_pipeline = joblib.load(
    MODEL_PATH
)


# Select the first five rows from the test dataset.
#
# These rows contain raw feature values before the pipeline's
# internal preprocessing.
raw_sample = X_test.iloc[
    :5
].copy()


# Pass the raw sample directly to the reloaded pipeline.
#
# The pipeline automatically performs:
#
# 1. Missing-value handling
# 2. Scaling
# 3. One-hot encoding
# 4. Classification prediction
raw_predictions = (
    loaded_pipeline
    .predict(raw_sample)
)


print(
    "\nRaw input sample:"
)

print(
    raw_sample
)


print(
    "\nPredictions from reloaded pipeline:"
)

print(
    raw_predictions
)


# Make sure the model returned one prediction for every
# input row.
#
# If this assertion fails, the saved pipeline is not behaving
# as expected.
assert len(
    raw_predictions
) == len(
    raw_sample
)


print(
    "\nReload verification successful."
)


# ============================================================
# Final message
# ============================================================

print("\n" + "=" * 80)
print("MODELING COMPLETE")
print("=" * 80)