
# Candidate Placement Prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# 1. LOAD DATA


df = pd.read_csv("Candidate_data.csv")

print("Dataset loaded successfully")
print(df.head())


# 2. CLEAN COLUMN NAMES


df.columns = df.columns.str.strip()

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where target is missing
df = df.dropna(subset=["Shortlisted"])

# 3. CLEAN TEXT COLUMNS


text_columns = [
    "Skills",
    "Education",
    "Certifications"
]

for col in text_columns:
    df[col] = df[col].fillna("None").astype(str).str.strip()

# Experience
df["Experience"] = pd.to_numeric(
    df["Experience"],
    errors="coerce"
)

df["Experience"] = df["Experience"].fillna(
    df["Experience"].median()
)

# Target
df["Shortlisted"] = pd.to_numeric(
    df["Shortlisted"],
    errors="coerce"
)

df = df.dropna(subset=["Shortlisted"])

df["Shortlisted"] = df["Shortlisted"].astype(int)

# 4. CLEAN SKILLS


df["Skills"] = (
    df["Skills"]
    .str.lower()
    .str.split(",")
    .apply(
        lambda x: [
            skill.strip()
            for skill in x
            if skill.strip()
        ]
    )
)


# 5. CREATE SKILL FEATURES


mlb = MultiLabelBinarizer()

skills_encoded = mlb.fit_transform(df["Skills"])

skills_df = pd.DataFrame(
    skills_encoded,
    columns=[
        "Skill_" + skill.replace(" ", "_")
        for skill in mlb.classes_
    ],
    index=df.index
)

# Combine skill features
df = pd.concat(
    [
        df.drop(columns=["Skills"]),
        skills_df
    ],
    axis=1
)


# 6. DEFINE FEATURES AND TARGET


X = df.drop(
    columns=[
        "Candidate_ID",
        "Shortlisted"
    ]
)

y = df["Shortlisted"]


# 7. CATEGORICAL AND NUMERICAL FEATURES

categorical_features = [
    "Education",
    "Certifications"
]

numeric_features = [
    "Experience"
]

skill_features = [
    col for col in X.columns
    if col.startswith("Skill_")
]

categorical_features = categorical_features
numeric_features = numeric_features + skill_features

# 8. PREPROCESSING


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# 9. RANDOM FOREST MODEL


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)


# 10. CREATE PIPELINE


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

# 11. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 12. TRAIN MODEL


pipeline.fit(
    X_train,
    y_train
)

print("\nModel training completed!")


# 13. MODEL PREDICTION


y_pred = pipeline.predict(X_test)


# 14. MODEL ACCURACY


accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# 15. CLASSIFICATION REPORT


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# 
# 16. CONFUSION MATRIX

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# 17. SUCCESS PROBABILITY


probabilities = pipeline.predict_proba(X)

# Probability of class 1
success_probability = probabilities[:, 1]

df["Success_Probability"] = (
    success_probability * 100
).round(2)

# 18. SUCCESS CATEGORY


def success_category(probability):

    if probability >= 80:
        return "High"

    elif probability >= 50:
        return "Medium"

    else:
        return "Low"


df["Success_Category"] = (
    df["Success_Probability"]
    .apply(success_category)
)


# 19. PREDICTION


df["Prediction"] = np.where(
    df["Success_Probability"] >= 50,
    "Likely Shortlisted",
    "Unlikely Shortlisted"
)

# 20. FINAL OUTPUT


output_columns = [
    "Candidate_ID",
    "Education",
    "Certifications",
    "Experience",
    "Shortlisted",
    "Success_Probability",
    "Success_Category",
    "Prediction"
]

# Add skill columns
output_columns = [
    col for col in output_columns
    if col in df.columns
]

result = df[output_columns]


# 21. SAVE OUTPUT


result.to_csv(
    "Candidate_Success_Probability.csv",
    index=False
)

print(
    "\nCandidate_Success_Probability.csv "
    "created successfully!"
)

print("\nFinal Results:")
print(result.head(10))


