drop database job_portal;
create database job_portal;
use job_portal;
CREATE TABLE job_data (
    Job_ID VARCHAR(20),
    Source VARCHAR(50),
    Job_Title VARCHAR(100),
    Company_Name VARCHAR(100),
    Industry VARCHAR(100),
    Location VARCHAR(150),
    Skills_Required TEXT,
    Education VARCHAR(100),
    Job_Type VARCHAR(50),
    Posted_Date DATE,
    Salary_Range_in_LPA DECIMAL(10,2),
    Experience_Required_in_Years DECIMAL(5,2)
);
SELECT COUNT(*) AS Total_Records
FROM job_data;
SELECT Job_ID, COUNT(*) AS Duplicate_Count
FROM job_data
GROUP BY Job_ID
HAVING COUNT(*) > 1;
SELECT
    SUM(Job_ID IS NULL) AS Missing_Job_ID,
    SUM(Job_Title IS NULL) AS Missing_Job_Title,
    SUM(Company_Name IS NULL) AS Missing_Company,
    SUM(Industry IS NULL) AS Missing_Industry,
    SUM(Location IS NULL) AS Missing_Location,
    SUM(Skills_Required IS NULL) AS Missing_Skills,
    SUM(Education IS NULL) AS Missing_Education,
    SUM(Job_Type IS NULL) AS Missing_Job_Type,
    SUM(Posted_Date IS NULL) AS Missing_Date,
    SUM(Salary_Range_in_LPA IS NULL) AS Missing_Salary,
    SUM(Experience_Required_in_Years IS NULL) AS Missing_Experience
FROM job_data;
CREATE TABLE job_data_backup AS
SELECT * FROM job_data;
SET SQL_SAFE_UPDATES = 0;

DELETE j1
FROM job_data j1
JOIN job_data j2
ON j1.Job_ID = j2.Job_ID
AND j1.Job_ID > j2.Job_ID;

SET SQL_SAFE_UPDATES = 1;
UPDATE job_data
SET Job_Type =
CASE
    WHEN LOWER(TRIM(Job_Type)) = 'full-time' THEN 'Full Time'
    WHEN LOWER(TRIM(Job_Type)) = 'part-time' THEN 'Part Time'
    WHEN LOWER(TRIM(Job_Type)) = 'contract' THEN 'Contract'
    WHEN LOWER(TRIM(Job_Type)) = 'internship' THEN 'Internship'
    ELSE TRIM(Job_Type)
END;
#EDA Analysis
#Total number of job
SELECT COUNT(*) AS Total_Jobs
FROM job_data;
#Total company name
SELECT COUNT(DISTINCT Company_Name) AS Total_Companies
FROM job_data;
#Total industries
SELECT COUNT(DISTINCT industry) AS Total_Industries
FROM job_data;
#Total locations
SELECT COUNT(DISTINCT Location) AS Total_Locations
FROM job_data;
# Highest demand job Role
SELECT
    Job_Title,
    COUNT(*) AS Job_Openings
FROM job_data
GROUP BY Job_Title
ORDER BY Job_Openings DESC;

# Most most job opening location
SELECT
    Location,
    COUNT(*) AS Job_Openings
FROM job_data
GROUP BY Location
ORDER BY Job_Openings DESC;

#Most requested skills
SELECT Skills_Required,COUNT(*) AS job_openings
FROM job_data
GROUP BY Skills_Required
ORDER BY Job_Openings DESC;

#The most hiring industries
SELECT industry,COUNT(*) AS job_openings
FROM job_data
GROUP BY industry
ORDER BY job_Openings DESC; 

#salary ranges are offered for different roles
SELECT job_title,ROUND(AVG(Salary_Range_in_LPA) ,2) AS Average_salary FROM job_data
GROUP BY job_title
ORDER BY Average_salary;

# most preferred experience levels
SELECT
    Experience_Required_in_Years,
    COUNT(*) AS Job_Openings
FROM job_data
GROUP BY Experience_Required_in_Years
ORDER BY Job_Openings DESC;















 
