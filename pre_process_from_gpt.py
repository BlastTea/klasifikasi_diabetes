import pandas as pd
import numpy as np

# Paths ke file CSV yang di-upload
file_paths = [
    'diff_datasets/Diabetes.csv',
    'diff_datasets/diabetes_binary_5050split_health_indicators_BRFSS2021.csv',
    'diff_datasets/diabetes_dataset__2019.csv',
    'diff_datasets/Dataset of Diabetes .csv',
    'diff_datasets/Diabetes Classification.csv'
]

# Membaca dan menggabungkan semua data
dataframes = []
for path in file_paths:
    df = pd.read_csv(path)
    dataframes.append(df)

# Menyelaraskan nama kolom
for df in dataframes:
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('diastolic_blood_pressure', 'blood_pressure').str.replace('plasma_glucose', 'glucose').str.replace('fbs', 'glucose')

# Menggabungkan semua DataFrame menjadi satu dengan mempertimbangkan kolom yang umum
all_data = pd.concat(dataframes, ignore_index=True, sort=False)

# Menyeleksi kolom yang diperlukan dan melakukan imputasi
cols_to_use = ['age', 'bmi', 'glucose', 'outcome', 'blood_pressure', 'smoker', 'gender', 'exercise']
for col in cols_to_use:
    if col in all_data.columns:
        if all_data[col].dtype == 'object':  # Imputasi untuk data kategorikal
            all_data[col].fillna(all_data[col].mode()[0], inplace=True)
        else:  # Imputasi untuk data numerik
            all_data[col].fillna(all_data[col].median(), inplace=True)
    else:
        print(f"Kolom '{col}' tidak ditemukan di data yang tersedia. Periksa ketersediaan kolom.")

# Menghapus baris dengan nilai target 'outcome' yang hilang
all_data.dropna(subset=['outcome'], inplace=True)

# Menyimpan hasil ke file CSV
output_file_path = 'outputs/Cleaned_Diabetes_Data.csv'
all_data.to_csv(output_file_path, index=False)

# Tampilkan informasi dan beberapa baris pertama dari data yang sudah bersih
print(all_data.info())
print(all_data.head())
