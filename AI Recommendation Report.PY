# AI-Based AnalysiS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
job_df = pd.read_csv("job_data_cleaned.csv")
candidate_df = pd.read_csv("candidate_data.csv")

# 1. Skill Gap Analysis
# Read CSV files
job_df = pd.read_csv("job_data_cleaned.csv")
candidate_df = pd.read_csv("candidate_data.csv")

# Clean column names
job_df.columns = job_df.columns.str.strip()
candidate_df.columns = candidate_df.columns.str.strip()

# Reset index
job_df.reset_index(drop=True, inplace=True)
candidate_df.reset_index(drop=True, inplace=True)

# Function: Find Skill Gap

def get_skill_set(skills):
    return {
        skill.strip().lower()
        for skill in str(skills).split(",")
        if skill.strip()
    }

# Convert skills to sets

candidate_df["Skill_Set"] = candidate_df["Skills"].apply(get_skill_set)

job_df["Skill_Set"] = job_df["Skills_Required"].apply(get_skill_set)

# Skill Gap Analysis


results = []

for candidate in candidate_df.itertuples(index=False):

    candidate_skills = candidate.Skill_Set

    for job in job_df.itertuples(index=False):

        missing_skills = job.Skill_Set - candidate_skills

        results.append({
            "Candidate_ID": candidate.Candidate_ID,
            "Job_ID": job.Job_ID,
            "Job_Title": job.Job_Title,
            "Missing_Skills": ", ".join(sorted(missing_skills))
        })


# Create DataFrame

skill_gap_df = pd.DataFrame(results)



# Save CSV

skill_gap_df.to_csv(
    "Skill_Gap_Analysis.csv",
    index=False
)

# Display result
print("\n Skill Gap Analysis completed!")
print("Candidates:", len(candidate_df))
print("Jobs:", len(job_df))
print("Total combinations:", len(skill_gap_df))

print("\nFirst 20 records:")
print(skill_gap_df.head(20))


# 2. Resume Matching Recommendations
def skill_match(candidate_skills, required_skills):

    candidate = set(candidate_skills)
    required = set(required_skills)

    matched = candidate.intersection(required)

    if len(required) == 0:
        return 0

    return len(matched) / len(required) * 100

# 3. Career Path Suggestions
career_paths = {
    "Data Analyst": [
        "Senior Data Analyst",
        "BI Analyst",
        "Data Scientist"
    ],

    "Junior Data Analyst": [
        "Data Analyst",
        "BI Analyst"
    ],

    "Data Scientist": [
        "Senior Data Scientist",
        "ML Engineer"
    ]
}

current_role = "Data Analyst"

print("Career Suggestions:")
for role in career_paths[current_role]:
    print(role)

# 4. Job Recommendation Engine
def recommend_jobs(candidate_skills, job_df):

    candidate_skills = {
        skill.strip().lower()
        for skill in candidate_skills.split(",")
    }

    recommendations = []

    for _, job in job_df.iterrows():

        required_skills = {
            skill.strip().lower()
            for skill in str(
                job["Skills_Required"]
            ).split(",")
        }

        matched_skills = (
            candidate_skills &
            required_skills
        )

        if len(required_skills) > 0:
            match_score = (
                len(matched_skills) /
                len(required_skills)
            ) * 100
        else:
            match_score = 0

        recommendations.append({
            "Job_Title": job["Job_Title"],
            "Company_Name": job["Company_Name"],
            "Location": job["Location"],
            "Industry": job["Industry"],
            "Matched_Skills": ", ".join(
                matched_skills
            ),
            "Match_Score": round(
                match_score, 2
            )
        })

    result = pd.DataFrame(
        recommendations
    )
    return result.sort_values(
        "Match_Score",
        ascending=False
    )

# 5. INDUSTRY TREND ANALYSIS

# 1. Industry Demand
industry_trend = (
    job_df.groupby("Industry")
    .size()
    .sort_values(ascending=False)
)

print("\nIndustry Demand:")
print(industry_trend)


# Bar chart - Industry Demand
industry_trend.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Industry Hiring Demand")
plt.xlabel("Industry")
plt.ylabel("Number of Jobs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Skills by Industry
industry_skill = (
    job_df.groupby(
        ["Industry", "Skills_Required"]
    )
    .size()
    .reset_index(name="Job_Count")
)

print("\nSkills by Industry:")
print(industry_skill.head(20))

# 3. Convert Posted Date
job_df["Posted_Date"] = pd.to_datetime(
    job_df["Posted_Date"],
    errors="coerce"
)


# Remove invalid dates
job_df = job_df.dropna(
    subset=["Posted_Date"]
)

# 4. Monthly Industry Trend

industry_monthly = (
    job_df.groupby([
        job_df["Posted_Date"].dt.to_period("M"),
        "Industry"
    ])
    .size()
    .reset_index(name="Job_Count")
)

# Convert Period to datetime
industry_monthly["Posted_Date"] = (
    industry_monthly["Posted_Date"]
    .dt.to_timestamp()
)

print("\nMonthly Industry Trend:")
print(industry_monthly.head(20))

# 5. Industry Trend Line Chart

plt.figure(figsize=(12, 6))

for industry in industry_monthly["Industry"].unique():

    temp = industry_monthly[
        industry_monthly["Industry"] == industry
    ]

    plt.plot(
        temp["Posted_Date"],
        temp["Job_Count"],
        marker="o",
        label=industry
    )


plt.title("Industry Hiring Trends Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# AI-Based Questions
#1. Which skills will be most demanded in the next 3–5 years?
job_df["Posted_Date"] = pd.to_datetime(job_df["Posted_Date"])

skills = (
    job_df["Skills_Required"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
    .str.title()
)

skill_demand = skills.value_counts().reset_index()
skill_demand.columns = ["Skill", "Job_Count"]

print(skill_demand.head(15))
# 2. Which industries are expected to grow?
# Make sure Posted_Date is datetime
job_df["Posted_Date"] = pd.to_datetime(
    job_df["Posted_Date"],
    errors="coerce"
)

# Remove rows with invalid dates
job_df = job_df.dropna(
    subset=["Posted_Date"]
)


# Create Month
# IMPORTANT: Convert Period to Timestamp
job_df["Month"] = (
    job_df["Posted_Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# Industry monthly demand
industry_trend = (
    job_df.groupby(
        ["Month", "Industry"]
    )
    .size()
    .reset_index(name="Job_Count")
)
print(industry_trend.head(20))
industry_trend.to_csv(
    "Industry_Growth.csv",
    index=False
)


# Make Job_Count numeric
industry_trend["Job_Count"] = pd.to_numeric(
    industry_trend["Job_Count"],
    errors="coerce"
)


# Remove invalid values
industry_trend = industry_trend.dropna(
    subset=["Job_Count"]
)


print("\nIndustry Trend:")
print(industry_trend.head(20))

# INDUSTRY TREND GRAPH
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=industry_trend,
    x="Month",
    y="Job_Count",
    hue="Industry",
    marker="o"
)

plt.title("Industry Hiring Trend")
plt.xlabel("Month")
plt.ylabel("Number of Jobs")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# AI CANDIDATE-JOB MATCHING
# Keep only required columns
candidates = candidate_df[
    [
        "Candidate_ID",
        "Skills",
        "Experience",
        "Education"
    ]
].copy()

jobs = job_df[
    [
        "Job_ID",
        "Job_Title",
        "Skills_Required",
        "Experience_Required",
        "Education"
    ]
].copy()

# Convert skills into sets


def skill_set(value):

    if pd.isna(value):
        return set()

    return {
        skill.strip().lower()
        for skill in str(value).split(",")
        if skill.strip()
    }


candidates["Skill_Set"] = (
    candidates["Skills"].apply(skill_set)
)

jobs["Skill_Set"] = (
    jobs["Skills_Required"].apply(skill_set)
)

# Convert experience to numeric
candidates["Experience"] = pd.to_numeric(
    candidates["Experience"],
    errors="coerce"
).fillna(0)

jobs["Experience_Required"] = pd.to_numeric(
    jobs["Experience_Required"],
    errors="coerce"
).fillna(0)

# Convert DataFrames to records
job_records = jobs[
    [
        "Job_ID",
        "Job_Title",
        "Skill_Set",
        "Experience_Required",
        "Education"
    ]
].to_dict("records")


candidate_records = candidates[
    [
        "Candidate_ID",
        "Skill_Set",
        "Experience",
        "Education"
    ]
].to_dict("records")


# Matching


results = []


for candidate in candidate_records:

    # Current candidate values
    candidate_skills = candidate["Skill_Set"]
    candidate_exp = candidate["Experience"]
    candidate_education = str(
        candidate["Education"]
    ).strip().lower()


    for job in job_records:

        # Current job values
        job_skills = job["Skill_Set"]
        required_exp = job["Experience_Required"]
        job_education = str(
            job["Education"]
        ).strip().lower()


        # 1. Skill Score
        if len(job_skills) == 0:

            skill_score = 0

        else:

            matched = (
                candidate_skills.intersection(
                    job_skills
                )
            )

            skill_score = (
                len(matched) /
                len(job_skills)
            ) * 100


       
        # 2. Experience Score
  

        if required_exp <= 0:

            experience_score = 100

        else:

            experience_score = (
                min(
                    candidate_exp / required_exp,
                    1
                ) * 100
            )


       
        # 3. Education Score


        if (
            candidate_education
            == job_education
        ):

            education_score = 100

        else:

            education_score = 0


      
        # 4. Final Match Score


        match_score = (
            skill_score * 0.60
            +
            experience_score * 0.25
            +
            education_score * 0.15
        )


       
        # Store result
      

        results.append({

            "Candidate_ID":
                candidate["Candidate_ID"],

            "Job_ID":
                job["Job_ID"],

            "Job_Title":
                job["Job_Title"],

            "Skill_Score":
                round(skill_score, 2),

            "Experience_Score":
                round(experience_score, 2),

            "Education_Score":
                education_score,

            "Match_Score":
                round(match_score, 2)
        })



# Create DataFrame


matching_df = pd.DataFrame(results)


print("\nAI Matching completed!")

print(
    matching_df.head(20)
)


# RECOMMENDATION


def recommendation(score):

    if score >= 80:
        return "Highly Recommended"

    elif score >= 60:
        return "Recommended"

    elif score >= 40:
        return "Partially Suitable"

    else:
        return "Not Suitable"


matching_df["Recommendation"] = (
    matching_df["Match_Score"]
    .apply(recommendation)
)


print("\nRecommendations:")

print(
    matching_df.head(20)
)

# TOP 3 JOB RECOMMENDATIONS


top_jobs = (
    matching_df
    .sort_values(
        ["Candidate_ID", "Match_Score"],
        ascending=[True, False]
    )
    .groupby("Candidate_ID")
    .head(3)
    .reset_index(drop=True)
)


print("\nTop 3 Job Recommendations:")

print(
    top_jobs.head(20)
)


# EXPORT RESULTS



matching_df.to_csv(
    "AI_Job_Matching.csv",
    index=False
)

top_jobs.to_csv(
    "Career_Recommendations.csv",
    index=False
)


print(
    "AI_Job_Matching.csv created successfully"
)

print(
    "Career_Recommendations.csv created successfully"
)
# Emerging Technologies


# Emerging technologies to identify
emerging_technologies = [
    "Artificial Intelligence",
    "AI",
    "Generative AI",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Natural Language Processing",
    "Computer Vision",
    "Large Language Models",
    "LLM",
    "Cloud Computing",
    "AWS",
    "Azure",
    "Google Cloud",
    "Big Data",
    "Blockchain",
    "Data Science",
    "IoT",
    "Cybersecurity"
]

# Split skills into separate rows
skills = (
    job_df["Skills_Required"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

# Count each skill
skill_count = skills.value_counts().reset_index()

skill_count.columns = ["Technology", "Job_Count"]

# Filter only emerging technologies
emerging_df = skill_count[
    skill_count["Technology"].str.lower().isin(
        [x.lower() for x in emerging_technologies]
    )
].copy()

# Calculate percentage of job demand
total_jobs = len(job_df)

emerging_df["Demand_Percentage"] = (
    emerging_df["Job_Count"] / total_jobs * 100
).round(2)

# Sort highest demand first
emerging_df = emerging_df.sort_values(
    by="Job_Count",
    ascending=False
)

# Display result
print("\nEMERGING TECHNOLOGIES")
print(emerging_df)

# Save result for Power BI
emerging_df.to_csv(
    "Emerging_Technologies.csv",
    index=False
)

print("\nEmerging_Technologies.csv created successfully!")
