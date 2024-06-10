import pandas as pd
import numpy as np
import random

# Paths ke file CSV yang di-upload
file_paths = [
    'diff_datasets/Diabetes.csv',
    'diff_datasets/diabetes_binary_health_indicators_BRFSS2021.csv',
    'diff_datasets/diabetes_dataset__2019.csv',
    'diff_datasets/Diabetes Classification.csv'
]

cols_to_use = [
    'age',
    'glucose',
    'bmi',
    'gender',
    'blood_pressure',
    'smoker',
    'outcome',
]

outputdf = None

row_list = []
dataframes = []

for path in file_paths:
    df = pd.read_csv(path)
    dataframes.append(df)
    
def random_age_from_category(category):
    if category == "less than 40":
        return random.randint(20, 39)
    elif category == "40-49":
        return random.randint(40, 49)
    elif category == "50-59":
        return random.randint(50, 59)
    elif category == "60 or older":
        return random.randint(60, 90)
    return None

def get_bloodpressure_from_number(number):
    if number < 60:
        return 0
    elif number >= 60 and number < 80:
        return 1
    elif number >= 80:
        return 2
    
def get_bloodpressure_from_category(category):
    if category.lower() == 'low':
        return 0
    elif category.lower() == 'normal':
        return 1
    elif category.lower() == 'high':
        return 2
    

for i, df in enumerate(dataframes):
    df.columns = [col.lower() for col in df.columns]
    
    # Age
    if i == 2:
        df['age'] = df['age'].apply(lambda x: random_age_from_category(x))
    age = df['age']
    
    # Glucose
    if i != 0:
        df['glucose'] = 0
            
    # BMI
    if i == 3:
        df['bmi'] = df['bmi'].apply(lambda x: float(x))
    bmi = df['bmi']
    
    # Gender (0 male, 1 female)
    match i:
        case 0: 
            df['gender'] = 1
        case 1:
            df['gender'] = df['sex']
        case 2:
            df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)
        case 3:
            df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)

    # Blood Pressure (0 low, 1 normal, 2 high)
    match i:
        case 0:
            df['blood_pressure'] = df['bloodpressure'].apply(lambda x: get_bloodpressure_from_number(x))
        case 1:
            df['blood_pressure'] = df['highbp'].apply(lambda x: 2 if x == 1 else 0)
        case 2:
            df['blood_pressure'] = df['bplevel'].apply(lambda x: get_bloodpressure_from_category(x))
        case 3:
            df['blood_pressure'] = df['blood pressure'].apply(lambda x: get_bloodpressure_from_category(x))
            
    # Smoker (0 no, 1 yes)
    match i:
        case 0:
            df['smoker'] = 0 # use Constant Imputation (no)
        case 2:
            df['smoker'] = df['smoking'].apply(lambda x: 1 if x == 'yes' else 0)
        case 3:
            df['smoker'] = df['smoking'].apply(lambda x: 1 if x == 'Yes' else 0)            
    
    # Outcome (0 no, 1 yes)
    match i:
        case 0:
            df['outcome'] = df['outcome']
        case 1:
            df['outcome'] = df['diabetes_binary'].apply(lambda x: int(x))
        case 2:
            df['outcome'] = df['diabetic'].apply(lambda x: 1 if x == 'yes' else 0)
        case 3:
            df['outcome'] = df['diagnosis'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    for index, row in df.iterrows():
        row_dict = {col: row[col] for col in cols_to_use if col in df.columns}
        row_list.append(row_dict)
    
outputdf = pd.DataFrame(row_list)

# print(outputdf)

outputdf.to_csv('outputs/cleaned_data.csv', index=False)