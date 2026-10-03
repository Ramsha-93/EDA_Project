# TITANIC DATASET - EXPLORATORY DATA ANALYSIS
# Python for AI - Month 1 Milestone


# In this project, I will use the Titanic dataset to look at
# survival rates, passenger class, and age.
# I will also clean the data and make a few useful charts.


# Import the libraries I need

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# STEP 2: LOAD AND CHECK THE DATA

df = pd.read_csv("titanic.csv")


print("\nFIRST 5 ROWS")
print(df.head())


print("\nLAST 5 ROWS")
print(df.tail())


print("\nDATASET SHAPE")
print(df.shape)


print("\nCOLUMN NAMES")
print(df.columns)


print("\nDATASET INFORMATION")
df.info()


print("\nSTATISTICAL SUMMARY")
print(df.describe())


# Check which columns are numeric and which are categorical
numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\nNUMERIC COLUMNS")
print(numeric_cols)


categorical_cols = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nCATEGORICAL COLUMNS")
print(categorical_cols)


# Short description of the dataset:
# The Titanic dataset contains information about 891 passengers.
# It includes details such as age, gender, passenger class, fare, and survival.
# It has both numeric and categorical columns, and some values are missing.
# This makes it useful for practicing data cleaning and exploratory analysis.

print("\nID COLUMN")
print("PassengerId")

print("\nDATE COLUMNS")
print("None in this dataset")


# PassengerId is an ID column.
# Pclass and Survived are stored as numbers, but they also represent categories.


# List comprehension

numeric_columns_check = [
    column
    for column in df.columns
    if df[column].dtype != "object"
]

print("\nNumeric columns using list comprehension:")
print(numeric_columns_check)


# Research questions

print("\nRESEARCH QUESTIONS")

print("1. Does survival rate differ between male and female passengers?")
print("2. How does passenger class relate to survival?")
print("3. Is age associated with survival?")


# STEP 3 - TASK 1
# Handle missing values

print("\nMISSING VALUES BEFORE CLEANING")

print(df.isna().sum())


# Keep the original data safe and work on a copy.
df_clean = df.copy()


# Cabin has many missing values, so I removed it instead of guessing the missing entries.

df_clean = df_clean.drop(columns=["Cabin"])


# Fill missing Age values with the median.

df_clean["Age"] = df_clean["Age"].fillna(
    df_clean["Age"].median()
)


# Embarked is categorical, so I used its most common value for the missing entry.

df_clean["Embarked"] = df_clean["Embarked"].fillna(
    df_clean["Embarked"].mode()[0]
)


print("\nMISSING VALUES AFTER CLEANING")

print(df_clean.isna().sum())


# STEP 3 - TASK 2
# Create a few new columns

# Add SibSp and Parch to get the total family size.

df_clean["FamilySize"] = (
    df_clean["SibSp"] +
    df_clean["Parch"] +
    1
)


print("\nFAMILY SIZE")

print(
    df_clean[
        ["SibSp", "Parch", "FamilySize"]
    ].head()
)


# Make age groups so the survival rates are easier to compare.

df_clean["AgeGroup"] = pd.cut(
    df_clean["Age"],
    bins=[0, 12, 18, 35, 60, float("inf")],
    labels=[
        "Child",
        "Teenager",
        "Young Adult",
        "Adult",
        "Senior"
    ],
    include_lowest=True
)


# apply() with lambda
# Group passengers into small and large families.

df_clean["FamilyType"] = df_clean["FamilySize"].apply(
    lambda x: "Large Family" if x >= 5 else "Small Family"
)


# apply() with a named function

def age_category(age):


    if age < 18:
        return "Minor"

    elif age < 60:
        return "Adult"

    else:
        return "Senior"


df_clean["AgeCategory"] = df_clean["Age"].apply(age_category)


# NumPy operation
# Use np.where() to create two fare groups.

df_clean["FareLevel"] = np.where(
    df_clean["Fare"] >= 50,
    "High Fare",
    "Low Fare"
)


print("\nNEW FEATURES")

print(
    df_clean[
        [
            "Age",
            "AgeGroup",
            "FamilySize",
            "FamilyType",
            "AgeCategory",
            "FareLevel"
        ]
    ].head()
)


print("\nUpdated dataset shape:")
print(df_clean.shape)


# Filter using two conditions
# Here I am checking adult passengers who paid more than 50.

filtered_passengers = df_clean[
    (df_clean["Age"] >= 18) &
    (df_clean["Fare"] > 50)
]


print("\nFILTERED PASSENGERS")

print(
    filtered_passengers[
        [
            "Name",
            "Age",
            "Fare",
            "Pclass",
            "Survived"
        ]
    ].head(10)
)


# Sort by two columns
# Sort by class first, then show higher fares first within each class.

sorted_df = df_clean.sort_values(
    by=["Pclass", "Fare"],
    ascending=[True, False]
)


print("\nSORTED DATA")

print(
    sorted_df[
        ["Pclass", "Fare", "Name"]
    ].head(10)
)


# STEP 3 - TASK 3
# Distributions
# Age distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df_clean["Age"],
    bins=20,
    kde=True
)

plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


print("\nAge Skewness:")
print(round(df_clean["Age"].skew(), 2))


# Most passengers are around the younger and middle age ranges.
# The distribution is slightly right-skewed because there are fewer older passengers.


# Fare distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df_clean["Fare"],
    bins=20,
    kde=True
)

plt.title("Distribution of Ticket Fare")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


print("\nFare Skewness:")
print(round(df_clean["Fare"].skew(), 2))


# Most passengers paid lower fares, while a small number paid very high fares.
# This makes the Fare distribution strongly right-skewed.


# STEP 3 - TASK 4
# Find fare outliers using IQR

# Find Q1 and Q3.

Q1 = df_clean["Fare"].quantile(0.25)
Q3 = df_clean["Fare"].quantile(0.75)


# Calculate the IQR.

IQR = Q3 - Q1


# Set the lower and upper limits for outliers.

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR


# Get the rows with fares outside these limits.

fare_outliers = df_clean[
    (df_clean["Fare"] < lower_limit) |
    (df_clean["Fare"] > upper_limit)
]


print("\nFARE OUTLIERS")

print("Lower Limit:", round(lower_limit, 2))
print("Upper Limit:", round(upper_limit, 2))

print("Number of Fare Outliers:")
print(len(fare_outliers))


print("\nSome Outlier Rows:")

print(
    fare_outliers[
        [
            "PassengerId",
            "Pclass",
            "Fare",
            "Survived"
        ]
    ].head(10)
)


# Boxplot by passenger class

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x="Pclass",
    y="Fare"
)

plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()
plt.show()


# First-class passengers generally paid higher fares.
# The plot also shows some unusually high fares.


# STEP 3 - TASK 5
# Correlation analysis

numeric_data = df_clean.select_dtypes(
    include=np.number
).drop(
    columns=["PassengerId"],
    errors="ignore"
)


correlation_matrix = numeric_data.corr()


# Correlation heatmap

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numeric Features")
plt.xlabel("Variables")
plt.ylabel("Variables")

plt.tight_layout()
plt.show()


print("\nCORRELATION WITH SURVIVAL")

print(
    correlation_matrix["Survived"]
    .sort_values(ascending=False)
)


# The heatmap shows how the numeric variables are related.
# A correlation does not mean that one variable caused the other.


# STEP 3 - TASK 6
# Group and compare

gender_survival = (
    df_clean
    .groupby("Sex")["Survived"]
    .mean() * 100
)


print("\nSURVIVAL RATE BY GENDER")

print(gender_survival.round(2))


# Survival by passenger class

class_survival = (
    df_clean
    .groupby("Pclass")["Survived"]
    .mean() * 100
)


print("\nSURVIVAL RATE BY PASSENGER CLASS")

print(class_survival.round(2))


# Survival by age group

age_survival = (
    df_clean
    .groupby(
        "AgeGroup",
        observed=True
    )["Survived"]
    .mean() * 100
)


print("\nSURVIVAL RATE BY AGE GROUP")

print(age_survival.round(2))


# groupby() with multiple statistics


class_stats = (
    df_clean
    .groupby("Pclass")["Fare"]
    .agg(
        [
            "mean",
            "min",
            "max",
            "count"
        ]
    )
)


print("\nFARE STATISTICS BY CLASS")

print(class_stats)


# Put the classes in their normal order: 1, 2, 3.

class_stats = class_stats.reindex([1, 2, 3])

print("\nORDERED CLASS STATISTICS")

print(class_stats)


# STEP 4 - Visualize the findings


# Chart 1: Survival rate by gender

plt.figure(figsize=(7, 5))

plt.bar(
    gender_survival.index,
    gender_survival.values,
    color="steelblue"
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")

plt.tight_layout()
plt.show()


# Female passengers had a much higher observed survival rate than male passengers.


# Chart 2: Survival rate by passenger class


plt.figure(figsize=(7, 5))

plt.bar(
    class_survival.index.astype(str),
    class_survival.values,
    color="darkorange"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")

plt.tight_layout()
plt.show()


# First-class passengers had the highest observed survival rate, while third-class had the lowest.


# Chart 3: Survival rate by age group

plt.figure(figsize=(9, 5))

sns.barplot(
    x=age_survival.index,
    y=age_survival.values
)

plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")

plt.tight_layout()
plt.show()


# Survival rates are different across age groups.
# Age alone does not explain survival because other factors may also matter.


# Chart 4: Fare distribution by class

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x="Pclass",
    y="Fare"
)

plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()
plt.show()


# First-class fares are generally much higher than second- and third-class fares.
# There are also some unusually high fares.


# Chart 5: Correlation heatmap

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numeric Features")
plt.xlabel("Variables")
plt.ylabel("Variables")

plt.tight_layout()
plt.show()


# The heatmap gives a quick view of relationships between the numeric columns.


# Requirement 14 - optional merge
# Add readable names for the three passenger classes.

class_lookup = pd.DataFrame({
    "Pclass": [1, 2, 3],
    "ClassName": [
        "First Class",
        "Second Class",
        "Third Class"
    ]
})


merged_df = pd.merge(
    df_clean,
    class_lookup,
    on="Pclass",
    how="left"
)


print("\nMERGED DATA")

print(
    merged_df[
        [
            "Pclass",
            "ClassName",
            "Name"
        ]
    ].head()
)


# Final conclusions

print("\n")
print("FINAL CONCLUSIONS")


print("""
1. Survival and Gender:
The Titanic dataset shows a clear difference in survival rates
between male and female passengers. Female passengers had a
higher observed survival rate than male passengers. However,
the analysis does not prove that gender alone caused this difference.
""")


print("""
2. Survival and Passenger Class:
Survival rates were different across passenger classes.
First-class passengers had a higher observed survival rate,
while third-class passengers had the lowest. The data supports
an association between passenger class and survival, but it
does not explain every factor behind this difference.
""")


print("""
3. Survival and Age:
The dataset contains passengers from different age groups,
and survival rates varied between those groups. However,
age alone is not enough to explain survival because factors
such as gender and passenger class may also be related to
the outcome.
""")


print("Titanic EDA Project Completed Successfully!")
