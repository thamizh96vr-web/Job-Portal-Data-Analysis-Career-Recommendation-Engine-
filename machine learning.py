import pandas as pd
import numpy as np


df = pd.read_csv("job_data_cleaned.csv")

print(df.shape)
print(df.columns.tolist())
print(df.head())
text_columns = [
    'Source',
    'Job_Title',
    'Company_Name',
    'Industry',
    'Location',
    'Salary_Range',
    'Experience_Required',
    'Skills_Required',
    'Education',
    'Job_Type'
]

for col in text_columns:
    df[col] = df[col].astype(str).str.strip()
# Create Average Salary

df['Min_Salary_LPA'] = (
    df['Salary_Range']
    .str.replace('₹', '', regex=False)
    .str.replace(' LPA', '', regex=False)
    .str.split('-')
    .str[0]
    .astype(float)
)
df['Max_Salary_LPA'] = (
    df['Salary_Range']
    .str.replace('₹', '', regex=False)
    .str.replace(' LPA', '', regex=False)
    .str.split('-')
    .str[1]
    .astype(float)
)
df['Avg_Salary_LPA'] = (
    df['Min_Salary_LPA'] +
    df['Max_Salary_LPA']
) / 2

# Extract experience
df['Experience_Years'] = (
    df['Experience_Required']
    .str.extract(r'(\d+)')
    .astype(float)
)
print(
    df[
        ['Experience_Required',
         'Experience_Years']
    ].head(10)
)
# prepare features
X = df[
    [
        'Skills_Required',
        'Experience_Years',
        'Location',
        'Industry'
    ]
]

y = df['Avg_Salary_LPA']
model_data = pd.concat([X, y], axis=1).dropna()

X = model_data.drop(columns=['Avg_Salary_LPA'])
y = model_data['Avg_Salary_LPA']

# Convert categorical data to numbers
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

categorical_features = [
    'Skills_Required',
    'Location',
    'Industry'
]

numeric_features = [
    'Experience_Years'
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            categorical_features
        ),
        (
            'numeric',
            'passthrough',
            numeric_features
        )
    ]
)
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Linear Regression
from sklearn.linear_model import LinearRegression

linear_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(
    y_test,
    linear_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

r2 = r2_score(
    y_test,
    linear_predictions
)

print("Linear Regression")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        (
            'model',
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

r2 = r2_score(
    y_test,
    rf_predictions
)

print("Random Forest")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Compare models
print("Linear Regression R2:", 
      r2_score(y_test, linear_predictions))

print("Random Forest R2:", 
      r2_score(y_test, rf_predictions))

# Test salary prediction
new_job = pd.DataFrame({
    'Skills_Required': [
        'Python, SQL, Power BI'
    ],
    'Experience_Years': [3],
    'Location': [
        'Bengaluru, Karnataka'
    ],
    'Industry': [
        'Software'
    ]
})

predicted_salary = rf_model.predict(new_job)

print(
    "Predicted Salary:",
    round(predicted_salary[0], 2),
    "LPA"
)
print("X_test columns:")
print(X_test.columns.tolist())

# Create Salary Prediction output
salary_output = X_test.copy()

# Add actual and predicted salary
salary_output["Actual_Salary_LPA"] = y_test.values
salary_output["Predicted_Salary_LPA"] = rf_predictions

# Save CSV
salary_output.to_csv(
    "Salary_Prediction.csv",
    index=False
)

print("Salary_Prediction.csv created successfully")
print(salary_output.head())

# Job Demand Prediction
# prepare date column
df['Posted_Date'] = pd.to_datetime(
    df['Posted_Date']
)

print(df['Posted_Date'].min())
print(df['Posted_Date'].max())
df['Week'] = (
    df['Posted_Date']
    .dt.to_period('W')
    .apply(lambda x: x.start_time)
)

# job-role demand
role_demand = (
    df.groupby(
        ['Week', 'Job_Title']
    )
    .size()
    .reset_index(
        name='Job_Count'
    )
)

print(role_demand.head())

# create lag features
role_demand['Lag_1'] = (
    role_demand
    .groupby('Job_Title')['Job_Count']
    .shift(1)
)
role_demand['Lag_2'] = (
    role_demand
    .groupby('Job_Title')['Job_Count']
    .shift(2)
)
role_demand['Rolling_Mean'] = (
    role_demand
    .groupby('Job_Title')['Job_Count']
    .transform(
        lambda x: x.shift(1).rolling(2).mean()
    )
)
role_demand = role_demand.dropna()

# Random Forest Demand Model
from sklearn.ensemble import RandomForestRegressor

X = role_demand[
    ['Lag_1', 'Lag_2', 'Rolling_Mean']
]

y = role_demand['Job_Count']
split = int(len(role_demand) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]
demand_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

demand_model.fit(
    X_train,
    y_train
)
demand_predictions = demand_model.predict(
    X_test
)

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        demand_predictions
    )
)

print(
    "R2:",
    r2_score(
        y_test,
        demand_predictions
    )
)

#  Predict future job demand

# Get latest values
latest = role_demand.iloc[-1]

lag_1 = latest["Job_Count"]
lag_2 = latest["Lag_1"]
rolling_mean = latest["Rolling_Mean"]

# Create future dates
last_date = pd.to_datetime(df["Posted_Date"]).max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=12,
    freq="MS"
)

future_predictions = []

# Predict next 12 months
for i in range(12):

    future_input = pd.DataFrame({
        "Lag_1": [lag_1],
        "Lag_2": [lag_2],
        "Rolling_Mean": [rolling_mean]
    })

    prediction = demand_model.predict(
        future_input
    )[0]

    # Make sure prediction is not negative
    prediction = max(0, round(prediction))

    future_predictions.append(prediction)

    # Update lag values for next month
    lag_2 = lag_1
    lag_1 = prediction

    # Update rolling mean
    rolling_mean = (lag_1 + lag_2) / 2


# Create Power BI dataset
future_hiring_output = pd.DataFrame({
    "Month": future_dates,
    "Predicted_Jobs": future_predictions
})


# Save CSV
future_hiring_output.to_csv(
    "Future_Hiring.csv",
    index=False
)

print("Future_Hiring.csv created successfully")

print(future_hiring_output)

# Candidate Placement Prediction
# Logistic Regression Placement Model
candidate_df = pd.read_csv(
    "candidate_data.csv"
)

print(candidate_df.head())
X = candidate_df[
    [
        'Skills',
        'Experience',
        'Education',
        'Certifications'
    ]
]

y = candidate_df['Shortlisted']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
candidate_preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            [
                'Skills',
                'Education',
                'Certifications'
            ]
        ),
        (
            'numeric',
            'passthrough',
            ['Experience']
        )
    ]
)
candidate_preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            [
                'Skills',
                'Education',
                'Certifications'
            ]
        ),
        (
            'numeric',
            'passthrough',
            ['Experience']
        )
    ]
)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline(
    steps=[
        (
            'preprocessor',
            candidate_preprocessor
        ),
        (
            'model',
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)


placement_prediction = logistic_model.predict(
    X_test
)

placement_probability = (
    logistic_model.predict_proba(X_test)[:, 1]
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        placement_prediction
    )
)

print(
    classification_report(
        y_test,
        placement_prediction
    )
)

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

rf_classifier = Pipeline(
    steps=[
        (
            'preprocessor',
            candidate_preprocessor
        ),
        (
            'model',
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

rf_classifier.fit(
    X_train,
    y_train
)

rf_prediction = rf_classifier.predict(
    X_test
)

print(
    "Random Forest Accuracy:",
    accuracy_score(
        y_test,
        rf_prediction
    )
)
candidate_output = candidate_df.copy()

candidate_output["Success_Probability"] = (
    rf_classifier.predict_proba(X)[:, 1]
)
candidate_output.to_csv(
    "Candidate_Prediction.csv",
    index=False
)

print("Candidate_Prediction.csv created")

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_test,
    rf_prediction
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title(
    'Candidate Placement Confusion Matrix'
)

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()
