"""
Transform Employee Attrition data to Position Fill Times format
This script converts IBM HR Attrition dataset to a format compatible with our dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Read the attrition dataset
print("Reading the Attrition dataset...")
# Use encoding to handle potential BOM characters in the first line
attrition_df = pd.read_csv('attached_assets/EmployeeAttrition.csv', encoding='utf-8-sig')

# Check if the data was loaded properly
print(f"Loaded {len(attrition_df)} employee records")

# Print to verify what columns we actually have
print("Columns in dataset:", attrition_df.columns.tolist())

# Create a subset of employees who left (attrition = 'Yes')
# Make sure the 'Attrition' column exists
if 'Attrition' in attrition_df.columns:
    departed_employees = attrition_df[attrition_df['Attrition'] == 'Yes'].copy()
    print(f"Found {len(departed_employees)} employees who left the company")
else:
    print("Error: 'Attrition' column not found in the dataset. Using all records instead.")
    departed_employees = attrition_df.copy()

# Sample down to a reasonable number of records if needed
if len(departed_employees) > 100:
    departed_employees = departed_employees.sample(100, random_state=42)
    print(f"Sampled down to {len(departed_employees)} employee records")

# Create a new dataframe for position fill times
position_fill_df = pd.DataFrame()

# Generate position IDs
position_fill_df['position_id'] = ['POS' + str(i).zfill(3) for i in range(1, len(departed_employees) + 1)]

# Map departments to our simplified structure
department_mapping = {
    'Sales': 'Sales',
    'Research & Development': 'Engineering',
    'Human Resources': 'HR'
}

# Check if 'Department' column exists
if 'Department' in departed_employees.columns:
    # Map departments - ensure no null values
    position_fill_df['department'] = departed_employees['Department'].map(department_mapping).fillna('Other')
else:
    print("Warning: 'Department' column not found. Using 'Other' as default.")
    position_fill_df['department'] = 'Other'

# Check if 'JobRole' column exists
if 'JobRole' in departed_employees.columns:
    # Use job roles directly - ensure no null values
    position_fill_df['job_role'] = departed_employees['JobRole'].fillna('Unspecified Role')
else:
    print("Warning: 'JobRole' column not found. Using 'Unspecified Role' as default.")
    position_fill_df['job_role'] = 'Unspecified Role'

# Map job level to seniority
level_to_seniority = {
    1: 'Junior',
    2: 'Mid-level',
    3: 'Senior', 
    4: 'Senior',
    5: 'Executive'
}

# Check if 'JobLevel' column exists
if 'JobLevel' in departed_employees.columns:
    position_fill_df['seniority_level'] = departed_employees['JobLevel'].map(level_to_seniority).fillna('Mid-level')
else:
    print("Warning: 'JobLevel' column not found. Using 'Mid-level' as default.")
    position_fill_df['seniority_level'] = 'Mid-level'

# Create reason for leaving based on available data
def get_reason(row):
    reasons = []
    
    # Check each potential reason field
    if 'JobSatisfaction' in row and not pd.isna(row['JobSatisfaction']) and row['JobSatisfaction'] <= 2:
        reasons.append('Dissatisfaction')
    
    if 'WorkLifeBalance' in row and not pd.isna(row['WorkLifeBalance']) and row['WorkLifeBalance'] <= 2:
        reasons.append('Work-Life Balance')
    
    if 'EnvironmentSatisfaction' in row and not pd.isna(row['EnvironmentSatisfaction']) and row['EnvironmentSatisfaction'] <= 2:
        reasons.append('Work Environment')
    
    if 'MonthlyIncome' in row and not pd.isna(row['MonthlyIncome']) and row['MonthlyIncome'] < 5000:
        reasons.append('Better Opportunity')
    
    if 'YearsAtCompany' in row and not pd.isna(row['YearsAtCompany']) and row['YearsAtCompany'] > 15:
        reasons.append('Retirement')
        
    # If we found reasons, randomly pick one, otherwise use a default reason
    if reasons:
        return random.choice(reasons)
    else:
        return random.choice(['Personal Reasons', 'Relocation', 'Career Change', 'Resignation'])

position_fill_df['reason_for_leaving'] = departed_employees.apply(get_reason, axis=1)

# Generate plausible vacancy dates - distribute over the past year
base_date = datetime.now() - timedelta(days=365)
position_fill_df['vacancy_date'] = [
    (base_date + timedelta(days=random.randint(0, 300))).strftime('%Y-%m-%d') 
    for _ in range(len(departed_employees))
]

# Generate fill times that make sense for the role level
def get_fill_time(seniority):
    if seniority == 'Junior':
        return random.randint(5, 30)
    elif seniority == 'Mid-level':
        return random.randint(15, 45)
    elif seniority == 'Senior':
        return random.randint(30, 90)
    else:  # Executive
        return random.randint(60, 120)

position_fill_df['days_to_fill'] = position_fill_df['seniority_level'].apply(get_fill_time)

# Calculate filled date based on vacancy date and days to fill
position_fill_df['vacancy_date'] = pd.to_datetime(position_fill_df['vacancy_date'])
position_fill_df['filled_date'] = position_fill_df['vacancy_date'] + pd.to_timedelta(position_fill_df['days_to_fill'], unit='d')
position_fill_df['filled_date'] = position_fill_df['filled_date'].dt.strftime('%Y-%m-%d')
position_fill_df['vacancy_date'] = position_fill_df['vacancy_date'].dt.strftime('%Y-%m-%d')

# Add hiring manager - use a few common names
hiring_managers = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson']
position_fill_df['hiring_manager'] = [random.choice(hiring_managers) for _ in range(len(departed_employees))]

# Add candidate source
sources = ['Internal Referral', 'LinkedIn', 'External Job Board', 'Company Website', 
           'Recruiting Agency', 'Career Fair', 'Headhunter', 'Direct Application']
position_fill_df['candidate_source'] = [random.choice(sources) for _ in range(len(departed_employees))]

# Check for and fill any missing values
for col in position_fill_df.columns:
    null_count = position_fill_df[col].isna().sum()
    if null_count > 0:
        print(f"Found {null_count} missing values in column {col}, filling them...")
        
        if col == 'department':
            position_fill_df[col] = position_fill_df[col].fillna('Other')
        elif col == 'job_role':
            position_fill_df[col] = position_fill_df[col].fillna('Unspecified Role')
        elif col == 'seniority_level':
            position_fill_df[col] = position_fill_df[col].fillna('Mid-level')
        elif col == 'reason_for_leaving':
            position_fill_df[col] = position_fill_df[col].fillna('Resignation')
        elif col == 'candidate_source':
            position_fill_df[col] = position_fill_df[col].fillna('External Job Board')
        elif col == 'hiring_manager':
            position_fill_df[col] = position_fill_df[col].fillna('Smith')

# Reorder columns to match expected dashboard format
position_fill_df = position_fill_df[[
    'department', 'job_role', 'position_id', 'vacancy_date', 'filled_date', 
    'days_to_fill', 'candidate_source', 'seniority_level', 'hiring_manager', 'reason_for_leaving'
]]

# Create a sample dataset based on the IBM Attrition dataset
position_fill_df.to_csv('data/position_fill_times.csv', index=False)

print(f"Transformed data saved to data/position_fill_times.csv")
print(f"Created {len(position_fill_df)} position records")

# Print some stats about the transformed data
print("\nData Distribution:")
print(f"Departments: {position_fill_df['department'].value_counts().to_dict()}")
print(f"Seniority Levels: {position_fill_df['seniority_level'].value_counts().to_dict()}")
print(f"Reasons for Leaving: {position_fill_df['reason_for_leaving'].value_counts().to_dict()}")
print(f"Average Fill Time: {position_fill_df['days_to_fill'].mean():.1f} days")