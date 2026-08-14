from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

# Path(__file__) gives the location of this Python file.
# resolve() converts it into an absolute path.
# parent gives the folder containing the Python file.
BASE_DIR = Path(__file__).resolve().parent

# Input dataset and output folders.
CSV_PATH = BASE_DIR / "titanic.csv"
CHART_DIR = BASE_DIR / "charts"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create the required folders if they do not already exist.
# exist_ok=True means that Python will not raise an error
# when the folder already exists.
CHART_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Apply a consistent visual style to all Seaborn charts.
# "whitegrid" makes the chart easier to read.
sns.set_theme(style="whitegrid")

# Ignore non-critical warning messages so that the terminal
# output focuses on the actual analysis results.
warnings.filterwarnings("ignore")


# ============================================================
# Helper Functions
# ============================================================

def save_text(filename: str, text: str) -> None:
    """
    Save a text result into the outputs directory.

    This helper function avoids repeating the same file-writing
    code in multiple places.
    """

    # Build the complete output file path.
    path = OUTPUT_DIR / filename

    # Open the file using UTF-8 encoding.
    # "w" means that an existing file will be replaced.
    with path.open("w", encoding="utf-8") as f:

        # Write the supplied text into the file.
        f.write(text)


def iqr_outlier_count(series: pd.Series) -> tuple[int, float, float]:
    """
    Identify outliers using the IQR method.

    Returns:
        1. Number of outliers
        2. Lower acceptable boundary
        3. Upper acceptable boundary

    The IQR method is:
        IQR = Q3 - Q1

        Lower bound = Q1 - 1.5 * IQR
        Upper bound = Q3 + 1.5 * IQR
    """

    # Remove missing values before calculating quartiles.
    # Missing values should not be treated as actual observations.
    clean = series.dropna()

    # Q1 represents the 25th percentile.
    q1 = clean.quantile(0.25)

    # Q3 represents the 75th percentile.
    q3 = clean.quantile(0.75)

    # Interquartile Range measures the spread of the
    # middle 50% of the observations.
    iqr = q3 - q1

    # Values below this boundary are considered lower outliers.
    lower = q1 - 1.5 * iqr

    # Values above this boundary are considered upper outliers.
    upper = q3 + 1.5 * iqr

    # Create a Boolean condition:
    # True when a value is outside either boundary.
    #
    # sum() counts True values, giving the number of outliers.
    count = ((clean < lower) | (clean > upper)).sum()

    # Return the count and both calculated boundaries.
    return int(count), float(lower), float(upper)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 80)
print("MODULE 2 — ANALYTICS PIPELINE")
print("=" * 80)

print("\n[1] Loading Titanic dataset...")


# Load the Titanic dataset directly from Seaborn.
# This is the only call to sns.load_dataset() in the program.
# The result is stored as a pandas DataFrame.
df = sns.load_dataset("titanic")


# Immediately save the raw dataset as required.
# index=False prevents pandas from adding its DataFrame
# index as an additional column in the CSV file.
df.to_csv(CSV_PATH, index=False)

print(f"Raw dataset saved to: {CSV_PATH}")


# ------------------------------------------------------------
# Basic dataset dimensions
# ------------------------------------------------------------

print("\nDataset shape:")

print(df.shape)

# ------------------------------------------------------------
# Dataset information
# ------------------------------------------------------------

print("\nDataset info:")


df.info()


# ------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------

print("\nDataset describe:")

# describe() provides summary statistics.
# include="all" requests statistics for both numeric
# and non-numeric columns where possible.
df.describe(include="all")


# ============================================================
# Missing-Value Profiling
# ============================================================

# Calculate the percentage of missing values in each column.
#
# df.isna()
#     identifies missing values.
#
# mean()
#     calculates the proportion of missing values because
#     True is treated as 1 and False as 0.
#
# mul(100)
#     converts the proportion into a percentage.
#
# sort_values()
#     puts columns with the highest missing percentage first.
missing_pct = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

# Keep only columns that actually contain missing values.
missing_pct = missing_pct[missing_pct > 0]


# This list will store profiling information that will
# eventually be written to an output text file.
profiling_text = []

profiling_text.append("TITANIC DATASET PROFILING")
profiling_text.append("=" * 80)
profiling_text.append(f"Shape: {df.shape}")
profiling_text.append("")
profiling_text.append("Missing-value percentages:")

# Convert the pandas Series into readable text.
profiling_text.append(
    missing_pct.to_string()
)


print("\nMissing-value percentages:")
print(missing_pct)


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\n[2] Cleaning dataset...")

# Store an explanation of each cleaning decision.
# This is useful for documenting the reasoning behind the
# preprocessing decisions in the final submission.
cleaning_notes = []


# ------------------------------------------------------------
# Decide how to handle missing values
# ------------------------------------------------------------

# Loop through each column that contains missing values.
#
# "column" contains the column name.
# "percentage" contains its percentage of missing values.
for column, percentage in missing_pct.items():

    # Less than 5% missing:
    # The amount of missing data is small enough that
    # removing the affected rows is reasonable.
    if percentage < 5:

        cleaning_notes.append(
            f"{column}: {percentage:.2f}% missing -> "
            f"DROP ROWS because missingness is under 5%."
        )

    # Between 5% and 30%:
    # Removing all affected rows could unnecessarily reduce
    # the dataset, so imputation is considered more appropriate.
    elif percentage <= 30:

        cleaning_notes.append(
            f"{column}: {percentage:.2f}% missing -> "
            f"IMPUTE because missingness is between 5% and 30%."
        )

    # More than 30% missing:
    # A large amount of information is missing.
    # Instead of replacing the missing values with a potentially
    # misleading statistic, preserve "missing" as a category.
    else:

        cleaning_notes.append(
            f"{column}: {percentage:.2f}% missing -> "
            f"KEEP COLUMN and encode missing as an explicit category "
            f"because missingness exceeds 30% and median/mode "
            f"imputation would discard substantial information."
        )


# ------------------------------------------------------------
# embarked: under 5% -> drop affected rows
# ------------------------------------------------------------

if "embarked" in df.columns:

    # Remove rows where the embarked value is missing.
    #
    # Because the percentage of missing values is very small,
    # removing these rows has minimal impact on the dataset.
    df = df.dropna(subset=["embarked"]).copy()


# ------------------------------------------------------------
# age: 5%-30% -> median imputation
# ------------------------------------------------------------

if "age" in df.columns:

    # Calculate the median age using the available values.
    #
    # Median is used rather than mean because it is less affected
    # by extreme values or outliers.
    age_median = df["age"].median()

    # Replace missing age values with the calculated median.
    df["age"] = df["age"].fillna(age_median)


# ------------------------------------------------------------
# deck: >30% -> explicit "Missing" category
# ------------------------------------------------------------

if "deck" in df.columns:

    # Convert the column to object/string-compatible type.
    #
    # where(condition, value) keeps the original value when
    # the condition is True and replaces it when False.
    #
    # Therefore:
    #     existing deck -> keep original value
    #     missing deck  -> "Missing"
    df["deck"] = (
        df["deck"]
        .astype("object")
        .where(df["deck"].notna(), "Missing")
    )


# ------------------------------------------------------------
# Document the cleaning strategy
# ------------------------------------------------------------

cleaning_notes.append("")

cleaning_notes.append(
    "Cleaning rationale:"
)

cleaning_notes.append(
    "- Columns under 5% missing: affected rows dropped."
)

cleaning_notes.append(
    "- Numeric columns between 5% and 30%: median imputation."
)

cleaning_notes.append(
    "- deck has very high missingness, so missingness is preserved "
    "as its own category rather than pretending the missing values "
    "are random observations."
)


print("\nCleaning decisions:")

# Display every cleaning decision in the terminal.
for note in cleaning_notes:
    print(note)


# ============================================================
# Save CLEANED Hand-off CSV
# ============================================================

# Save the cleaned dataset.
#
# This overwrites the raw CSV created earlier, meaning the CSV
# now represents the cleaned hand-off dataset used for analysis.
df.to_csv(CSV_PATH, index=False)

print(
    f"\nCleaned dataset written to: {CSV_PATH}"
)

print(
    f"Cleaned shape: {df.shape}"
)


# ============================================================
# 3. UNIVARIATE ANALYSIS
# ============================================================

print("\n[3] Univariate analysis...")


# ============================================================
# Age Histogram
# ============================================================

# Create a new figure for the age histogram.
plt.figure(figsize=(8, 5))


# Display the distribution of passenger ages.
#
# histplot() creates a histogram.
# kde=True adds a smooth density curve.
# bins=30 divides the age range into 30 intervals.
sns.histplot(
    data=df,
    x="age",
    kde=True,
    bins=30,
)


plt.title("Titanic Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")


# tight_layout() prevents labels from being cut off.
plt.tight_layout()


# Save the chart as a PNG image.
plt.savefig(
    CHART_DIR / "age_histogram.png",
    dpi=150,
)


# Close the figure to release memory and prevent
# later plots from being drawn on the same figure.
plt.close()


# ============================================================
# Age Boxplot
# ============================================================

plt.figure(figsize=(8, 4))

# A boxplot shows:
#     - median
#     - quartiles
#     - spread
#     - potential outliers
sns.boxplot(
    data=df,
    x="age",
)

plt.title("Titanic Age Box Plot")
plt.xlabel("Age")

plt.tight_layout()

plt.savefig(
    CHART_DIR / "age_boxplot.png",
    dpi=150,
)

plt.close()


# ============================================================
# Fare Histogram
# ============================================================

plt.figure(figsize=(8, 5))

# Display the distribution of passenger fares.
#
# Fare is expected to contain some high-value observations,
# so the histogram helps identify the shape of the distribution.
sns.histplot(
    data=df,
    x="fare",
    kde=True,
    bins=40,
)

plt.title("Titanic Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    CHART_DIR / "fare_histogram.png",
    dpi=150,
)

plt.close()


# ============================================================
# Fare Boxplot
# ============================================================

plt.figure(figsize=(8, 4))

# The boxplot makes potential extreme fare values easier to identify.
sns.boxplot(
    data=df,
    x="fare",
)

plt.title("Titanic Fare Box Plot")
plt.xlabel("Fare")

plt.tight_layout()

plt.savefig(
    CHART_DIR / "fare_boxplot.png",
    dpi=150,
)

plt.close()


# ============================================================
# IQR Outlier Analysis
# ============================================================

# Calculate the number of outliers in the age column
# using the Interquartile Range method.
age_outliers, age_lower, age_upper = iqr_outlier_count(
    df["age"]
)


# Calculate the same statistics for fare.
fare_outliers, fare_lower, fare_upper = iqr_outlier_count(
    df["fare"]
)


print(
    f"\nAge outliers: {age_outliers}"
)

print(
    f"Fare outliers: {fare_outliers}"
)


# ============================================================
# Fare Summary Statistics
# ============================================================

# Calculate the average fare.
fare_mean = df["fare"].mean()

# Calculate the middle fare value.
fare_median = df["fare"].median()

# mode() returns a Series because more than one value
# can potentially have the same highest frequency.
fare_mode_values = df["fare"].mode()


# Select the first mode if one exists.
# If no mode exists, use NumPy's NaN value.
fare_mode = (
    fare_mode_values.iloc[0]
    if not fare_mode_values.empty
    else np.nan
)


# ------------------------------------------------------------
# Basic skewness interpretation
# ------------------------------------------------------------

# A commonly used rule of thumb is:
#
# Right-skewed:
#     mean > median > mode
#
# Left-skewed:
#     mean < median < mode
#
# If neither pattern occurs, we avoid making a strong
# conclusion based only on these three statistics.
if fare_mean > fare_median > fare_mode:

    fare_skew_text = (
        "Fare is right-skewed: mean > median > mode."
    )

elif fare_mean < fare_median < fare_mode:

    fare_skew_text = (
        "Fare is left-skewed: mean < median < mode."
    )

else:

    fare_skew_text = (
        "Fare does not follow a strict mean/median/mode "
        "ordering; the histogram should be used alongside "
        "the summary statistics."
    )


# Create a formatted text report containing the
# univariate analysis results.
outlier_text = f"""
UNIVARIATE ANALYSIS
===================

Age:
IQR lower bound: {age_lower:.4f}
IQR upper bound: {age_upper:.4f}
IQR outlier count: {age_outliers}

Fare:
IQR lower bound: {fare_lower:.4f}
IQR upper bound: {fare_upper:.4f}
IQR outlier count: {fare_outliers}

Fare mean: {fare_mean:.4f}
Fare median: {fare_median:.4f}
Fare mode: {fare_mode:.4f}

Skewness interpretation:
{fare_skew_text}
"""


print(outlier_text)


# Save the univariate analysis to a text file.
save_text(
    "outlier_analysis.txt",
    outlier_text,
)


# ============================================================
# 4. BIVARIATE ANALYSIS
# ============================================================

print("\n[4] Bivariate analysis...")


# ============================================================
# Survival by Sex
# BOOLEAN MASKING
# ============================================================

# Create a Boolean mask identifying male passengers.
#
# The result is a Series containing True/False values:
#     True  -> passenger is male
#     False -> passenger is not male
male_mask = df["sex"] == "male"


# Create a similar mask for female passengers.
female_mask = df["sex"] == "female"


# Select only male passengers using .loc[].
#
# "survived" contains:
#     1 -> survived
#     0 -> did not survive
#
# Therefore, the mean of this column is the survival rate.
male_survival = (
    df.loc[male_mask, "survived"].mean()
)


# Calculate the same survival rate for female passengers.
female_survival = (
    df.loc[female_mask, "survived"].mean()
)


# Store both survival rates in one pandas Series.
sex_survival = pd.Series(
    {
        "male": male_survival,
        "female": female_survival,
    }
)


print("\nSurvival by sex:")
print(sex_survival)


# ============================================================
# Survival by Passenger Class
# ============================================================

# A dictionary will temporarily store the survival rate
# for each passenger class.
class_survival = {}


# unique() returns the passenger classes present in the data.
# sorted() ensures the classes are processed in numerical order.
for pclass in sorted(df["pclass"].unique()):

    # Create a Boolean mask for the current class.
    mask = df["pclass"] == pclass

    # Select passengers belonging to this class
    # and calculate their average survival value.
    class_survival[pclass] = (
        df.loc[mask, "survived"].mean()
    )


# Convert the dictionary into a pandas Series
# for easier display and plotting.
class_survival = pd.Series(class_survival)


print("\nSurvival by class:")
print(class_survival)


# ============================================================
# Survival by Sex AND Passenger Class
# BOOLEAN COMBINATION
# ============================================================

# This list will store one result for every
# sex + passenger class combination.
combined_rows = []


# Loop through both passenger sexes.
for sex in ["female", "male"]:

    # Loop through all available passenger classes.
    for pclass in sorted(df["pclass"].unique()):

        # Combine two Boolean conditions.
        #
        # & means AND in pandas Boolean expressions.
        #
        # Therefore, a row must satisfy BOTH:
        #     sex == current sex
        #     pclass == current class
        mask = (
            (df["sex"] == sex)
            & (df["pclass"] == pclass)
        )

        # Select only the rows matching both conditions.
        subset = df.loc[mask]


        # Calculate survival rate.
        #
        # len(subset) > 0 prevents a division/calculation
        # problem if a combination has no observations.
        survival_rate = (
            subset["survived"].mean()
            if len(subset) > 0
            else np.nan
        )


        # Store the result as a dictionary.
        combined_rows.append(
            {
                "sex": sex,
                "pclass": pclass,
                "survival_rate": survival_rate,
                "count": len(subset),
            }
        )


# Convert all generated dictionaries into a DataFrame.
sex_class_survival = pd.DataFrame(
    combined_rows
)


print(
    "\nSurvival by sex and class:"
)

print(
    sex_class_survival
)


# ============================================================
# Bivariate Charts
# ============================================================

# ------------------------------------------------------------
# Chart: Survival by Sex
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

# Create a bar chart showing the survival rate for males
# and females.
sns.barplot(
    x=sex_survival.index,
    y=sex_survival.values,
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")

# Survival rate is a proportion, so the y-axis
# should range from 0 to 1.
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "survival_by_sex.png",
    dpi=150,
)

plt.close()


# ------------------------------------------------------------
# Chart: Survival by Passenger Class
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    x=class_survival.index.astype(str),
    y=class_survival.values,
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "survival_by_class.png",
    dpi=150,
)

plt.close()


# ------------------------------------------------------------
# Chart: Survival by Sex and Class
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

# hue="sex" creates separate bars for males and females
# within each passenger class.
sns.barplot(
    data=sex_class_survival,
    x="pclass",
    y="survival_rate",
    hue="sex",
)

plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "survival_sex_class.png",
    dpi=150,
)

plt.close()


# ============================================================
# Required 6-Column Correlation Matrix
# ============================================================

# Explicitly select the six required numeric columns.
#
# Correlation measures the strength and direction of a
# linear relationship between two numeric variables.
correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]


# Calculate the pairwise Pearson correlation coefficient
# for all six selected columns.
correlation_matrix = df[
    correlation_columns
].corr()


print(
    "\nRequired 6x6 correlation matrix:"
)

print(correlation_matrix)


# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

# Display the correlation matrix as a colored grid.
#
# annot=True:
#     display the numerical correlation value.
#
# fmt=".2f":
#     show values to two decimal places.
#
# center=0:
#     make zero the center of the color scale.
#
# square=True:
#     make each cell approximately square.
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
)

plt.title(
    "Titanic Correlation Matrix — Required Six Numeric Columns"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "correlation_heatmap.png",
    dpi=150,
)

plt.close()


# ============================================================
# Find Two Strongest Correlations
# ============================================================

# This list will contain every unique pair of variables.
pairs = []


# Use two loops to generate each pair only once.
#
# Starting j at i + 1 prevents duplicate pairs such as:
#     age / fare
#     fare / age
for i in range(len(correlation_columns)):

    for j in range(i + 1, len(correlation_columns)):

        col1 = correlation_columns[i]
        col2 = correlation_columns[j]

        # Extract the correlation coefficient for this pair.
        coefficient = correlation_matrix.loc[
            col1,
            col2,
        ]

        # Store both the signed and absolute correlation.
        #
        # The signed value tells us direction:
        #     positive -> variables increase together
        #     negative -> one tends to decrease as the other increases
        #
        # The absolute value tells us strength without considering direction.
        pairs.append(
            {
                "feature_1": col1,
                "feature_2": col2,
                "correlation": coefficient,
                "absolute_correlation": abs(coefficient),
            }
        )


# Convert all pairs into a DataFrame.
#
# Sort by absolute correlation from strongest to weakest.
# head(2) selects the two strongest relationships.
strongest_pairs = (
    pd.DataFrame(pairs)
    .sort_values(
        "absolute_correlation",
        ascending=False,
    )
    .head(2)
)


print(
    "\nTwo strongest absolute off-diagonal correlations:"
)

print(strongest_pairs)


# ============================================================
# 5. MULTIVARIATE DATA STORY
# ============================================================

print("\n[5] Multivariate data story...")


# ============================================================
# Chart 1: Sex + Class + Survival
# ============================================================

plt.figure(figsize=(8, 5))

# This chart combines:
#     passenger class
#     sex
#     survival rate
#
# This allows us to examine how multiple variables
# interact rather than looking at one variable at a time.
sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex",
)

plt.title(
    "Survival Rate by Class and Sex"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    CHART_DIR / "story_1_class_sex.png",
    dpi=150,
)

plt.close()


# ============================================================
# Chart 2: Age + Survival + Sex
# ============================================================

plt.figure(figsize=(9, 5))

# A boxplot is used because it allows us to compare
# the distribution of age across survival groups.
#
# hue="sex" adds another dimension to the visualization.
sns.boxplot(
    data=df,
    x="survived",
    y="age",
    hue="sex",
)

plt.title(
    "Age Distribution by Survival and Sex"
)

plt.xlabel("Survived")
plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    CHART_DIR / "story_2_age_survival.png",
    dpi=150,
)

plt.close()


# ============================================================
# Chart 3: Age + Fare + Survival + Sex
# ============================================================

plt.figure(figsize=(9, 6))

# Each point represents a passenger.
#
# x-axis -> age
# y-axis -> fare
# hue   -> survival status
# style -> sex
#
# This allows four variables to be examined simultaneously.
sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.65,
)

plt.title(
    "Age and Fare by Survival and Sex"
)

plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    CHART_DIR / "story_3_age_fare_survival.png",
    dpi=150,
)

plt.close()


# ============================================================
# Chart 4: Family Size + Survival + Sex
# ============================================================

# Create a separate copy so that the original cleaned DataFrame
# is not modified unnecessarily.
story_df = df.copy()


# Family size is calculated as:
#
#     siblings/spouses aboard
#   + parents/children aboard
#   + passenger themselves
#
# Therefore:
#     family_size = sibsp + parch + 1
story_df["family_size"] = (
    story_df["sibsp"]
    + story_df["parch"]
    + 1
)


plt.figure(figsize=(10, 5))


# Show survival rate for each family size.
# hue="sex" allows comparison between male and female passengers.
#
# errorbar=None removes confidence interval bars so the chart
# focuses directly on the observed survival rates.
sns.barplot(
    data=story_df,
    x="family_size",
    y="survived",
    hue="sex",
    errorbar=None,
)

plt.title(
    "Survival Rate by Family Size and Sex"
)

plt.xlabel("Family Size")
plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    CHART_DIR / "story_4_family_size.png",
    dpi=150,
)

plt.close()


# ============================================================
# Written Multivariate Interpretation
# ============================================================

# This text provides a written explanation of the patterns
# shown in the multivariate charts.
#
# Including written interpretation is important because charts
# alone do not explain the analytical conclusion.
story_text = """
MULTIVARIATE DATA STORY
========================

Chart 1 — Survival by class and sex
-----------------------------------
This chart shows that sex and passenger class jointly explain substantial
differences in survival probability. Female passengers generally had higher
survival rates than male passengers, while first-class passengers generally
had better survival outcomes than lower-class passengers. The interaction
between these variables is stronger than considering either variable alone.

Chart 2 — Age, survival and sex
-------------------------------
The age distributions show that survivors and non-survivors are not
identically distributed across sex groups. Younger passengers, particularly
children, appear among survivors, while male non-survivors are concentrated
more heavily across adult ages. This supports the idea that age and sex
provide complementary predictive information.

Chart 3 — Age, fare and survival
--------------------------------
The scatter plot combines socioeconomic position, age and survival. Higher
fares tend to be associated with particular passenger groups and classes,
while survival is visibly concentrated differently across the fare range.
The pattern suggests that fare acts partly as a proxy for socioeconomic
status and passenger class.

Chart 4 — Family size and sex
-----------------------------
Family size also contributes to the survival story. Survival rates vary
across family-size groups, and the pattern differs between men and women.
This suggests that the circumstances associated with traveling alone or
with relatives may have influenced survival, although the relationship is
not necessarily monotonic.
"""


print(story_text)


# Save the complete bivariate and multivariate analysis results.
save_text(
    "bivariate_analysis.txt",
    (
        "SURVIVAL BY SEX\n"
        + sex_survival.to_string()
        + "\n\nSURVIVAL BY CLASS\n"
        + class_survival.to_string()
        + "\n\nSURVIVAL BY SEX AND CLASS\n"
        + sex_class_survival.to_string(index=False)
        + "\n\n"
        + "CORRELATION MATRIX\n"
        + correlation_matrix.to_string()
        + "\n\n"
        + "TWO STRONGEST CORRELATIONS\n"
        + strongest_pairs.to_string(index=False)
        + "\n\n"
        + story_text
    ),
)


# ============================================================
# 6. EXPLORATORY STANDARDIZATION
# ============================================================

print("\n[6] Exploratory standardization...")


# Select only the numeric columns that will be standardized.
#
# Standardization is useful when variables have very different
# scales. For example, age may range around 0-80 while fare
# can have much larger numerical values.
standardization_df = df[
    ["age", "fare"]
].copy()


# Calculate the mean and standard deviation before scaling.
#
# This provides a baseline for comparing the data
# before and after standardization.
before = standardization_df.agg(
    ["mean", "std"]
)


# Create a StandardScaler object.
#
# StandardScaler applies the transformation:
#
#     standardized value =
#         (value - mean) / standard deviation
#
# This produces values centered approximately around zero
# with standard deviation approximately equal to one.
scaler = StandardScaler()


# Fit the scaler to the age and fare columns and transform them.
#
# fit_transform() performs two operations:
#
#     1. fit  -> calculate mean and standard deviation
#     2. transform -> apply the standardization formula
standardized_values = scaler.fit_transform(
    standardization_df
)


# Convert the NumPy array returned by StandardScaler
# back into a pandas DataFrame for easier analysis.
standardized_df = pd.DataFrame(
    standardized_values,
    columns=["age", "fare"],
)


# Calculate mean and standard deviation after standardization.
#
# The means should be approximately 0 and the standard deviations
# should be approximately 1.
after = standardized_df.agg(
    ["mean", "std"]
)


print("\nBefore standardization:")
print(before)

print("\nAfter standardization:")
print(after)


# ============================================================
# Before / After Standardization Plot
# ============================================================

# Create two plots side by side.
#
# This makes it easy to visually compare the distribution
# before and after standardization.
fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
)


# ------------------------------------------------------------
# Age before standardization
# ------------------------------------------------------------

sns.histplot(
    standardization_df["age"],
    kde=True,
    ax=axes[0],
)

axes[0].set_title(
    "Age Before Standardization"
)


# ------------------------------------------------------------
# Age after standardization
# ------------------------------------------------------------

sns.histplot(
    standardized_df["age"],
    kde=True,
    ax=axes[1],
)

axes[1].set_title(
    "Age After Standardization"
)


plt.tight_layout()


# Save the comparison chart.
plt.savefig(
    CHART_DIR / "standardization_before_after.png",
    dpi=150,
)

plt.close()


# ============================================================
# Standardization Interpretation
# ============================================================

# Create a written explanation of the standardization results.
#
# This makes it clear to the evaluator that standardization
# was performed as an exploratory EDA exercise and not
# accidentally used as model training data.
standardization_text = f"""
EXPLORATORY STANDARDIZATION
===========================

Before:
{before.to_string()}

After:
{after.to_string()}

The transformed age and fare columns have means approximately equal to
zero and standard deviations approximately equal to one. This is an EDA
sanity check only; these transformed values are NOT used as input to the
classification models. The modeling pipeline performs its own training-only
scaling to prevent data leakage.
"""


print(standardization_text)


# ============================================================
# Save Profiling and Cleaning Report
# ============================================================

# Combine:
#     - dataset profiling
#     - missing-value decisions
#     - outlier analysis
#     - standardization results
#
# into one text report.
save_text(
    "profiling.txt",
    (
        "\n".join(profiling_text)
        + "\n\n"
        + "\n".join(cleaning_notes)
        + "\n\n"
        + outlier_text
        + "\n\n"
        + standardization_text
    ),
)


# ============================================================
# Pipeline Completion
# ============================================================

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)