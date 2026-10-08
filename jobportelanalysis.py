import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV
df = pd.read_csv("intern raw data for others.csv")

# Check columns
print("Columns:")
print(df.columns.tolist())

# First 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Unique values
print("\nUnique Values:")
for col in df.columns:
    print(f"\n--- {col} ---")
    print(df[col].unique()[:20])

# Remove extra spaces from text columns
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
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Create minimum salary:
df['Min_Salary_LPA'] = (
    df['Salary_Range']
    .str.replace('₹', '', regex=False)
    .str.replace(' LPA', '', regex=False)
    .str.split('-')
    .str[0]
    .astype(float)
)

#Create maximum salary:
df['Max_Salary_LPA'] = (
    df['Salary_Range']
    .str.replace('₹', '', regex=False)
    .str.replace(' LPA', '', regex=False)
    .str.split('-')
    .str[1]
    .astype(float)
)

#Create average salary:
df['Avg_Salary_LPA'] = (
    df['Min_Salary_LPA'] + df['Max_Salary_LPA']
) / 2
print(
    df[
        [
            'Salary_Range',
            'Min_Salary_LPA',
            'Max_Salary_LPA',
            'Avg_Salary_LPA'
        ]
    ].head(10)
)

# Highest Demand Job Roles
job_demand = (
    df['Job_Title']
    .value_counts()
    .reset_index()
)

job_demand.columns = ['Job_Title', 'Job_Count']

print(job_demand)
# chart creation

plt.figure(figsize=(10, 6))

sns.barplot(
    data=job_demand.head(10),
    x='Job_Count',
    y='Job_Title'
)

plt.title('Top 10 Job Roles by Demand')
plt.xlabel('Number of Job Openings')
plt.ylabel('Job Role')

plt.show()

# Locations With Most Jobs
location_demand = (
    df['Location']
    .value_counts()
    .reset_index()
)

location_demand.columns = ['Location', 'Job_Count']

print(location_demand)

# chart creation
plt.figure(figsize=(10, 6))

sns.barplot(
    data=location_demand.head(10),
    x='Job_Count',
    y='Location'
)

plt.title('Top Locations by Job Openings')
plt.xlabel('Number of Jobs')
plt.ylabel('Location')

plt.show()

# Most Requested Skills
skills = (
    df['Skills_Required']
    .dropna()
    .str.split(',')
    .explode()
    .str.strip()
)

skill_demand = skills.value_counts().reset_index()

skill_demand.columns = ['Skill', 'Count']

print(skill_demand.head(20))

# chart creation
plt.figure(figsize=(10, 7))

sns.barplot(
    data=skill_demand.head(15),
    x='Count',
    y='Skill'
)

plt.title('Top 15 Most Requested Skills')
plt.xlabel('Number of Job Postings')
plt.ylabel('Skill')

plt.show()

#Industries Hiring Most
industry_demand = (
    df['Industry']
    .value_counts()
    .reset_index()
)

industry_demand.columns = ['Industry', 'Job_Count']

print(industry_demand)

# chart
plt.figure(figsize=(10, 6))

sns.barplot(
    data=industry_demand,
    x='Job_Count',
    y='Industry'
)

plt.title('Jobs by Industry')
plt.xlabel('Number of Jobs')
plt.ylabel('Industry')

plt.show()

# Salary Offered by Job Role
salary_by_role = (
    df.groupby('Job_Title')['Avg_Salary_LPA']
    .agg(['count', 'mean', 'min', 'max'])
    .sort_values('mean', ascending=False)
)

print(salary_by_role)
top_salary = (
    df.groupby('Job_Title')['Avg_Salary_LPA']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print(top_salary)

# chart
plt.figure(figsize=(10, 6))

top_salary.sort_values().plot(kind='barh')

plt.title('Top 10 Job Roles by Average Salary')
plt.xlabel('Average Salary (LPA)')
plt.ylabel('Job Role')

plt.show()

# Preferred Experience Level
experience_demand = (
    df['Experience_Required']
    .value_counts()
    .reset_index()
)

experience_demand.columns = [
    'Experience_Level',
    'Job_Count'
]

print(experience_demand)
plt.figure(figsize=(9, 6))

sns.barplot(
    data=experience_demand,
    x='Experience_Level',
    y='Job_Count'
)

# chart
plt.title('Job Openings by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Number of Jobs')

plt.show()
df.to_csv(
    'job_data_cleaned.csv',
    index=False
)

print("Cleaned dataset saved successfully!")