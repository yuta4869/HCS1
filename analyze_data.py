import pandas as pd
import os

def analyze_excel(file_path, sheet_name=None):
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        print(f"--- Analyzing {os.path.basename(file_path)} ---")
        if sheet_name:
             df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
             # Try to read the first sheet or specific sheets if known
             xls = pd.ExcelFile(file_path)
             print(f"Sheets: {xls.sheet_names}")
             df = pd.read_excel(file_path, sheet_name=xls.sheet_names[0])
        
        print("Head:")
        print(df.head())
        print("Column Names:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        print("\nSummary Statistics (Grouped by Condition if available):")
        
        # Guess condition column
        cond_col = None
        for col in df.columns:
            if 'Condition' in col or 'condition' in col or '条件' in col:
                cond_col = col
                break
        
        if cond_col:
            # Group by condition and calculate mean for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            grouped = df.groupby(cond_col)[numeric_cols].mean()
            print(grouped)
        else:
            print("No Condition column found. Summary of all data:")
            print(df.describe())

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Analyze HRV Results
# analyze_excel('/Users/user/Research/HCS_ver4.0/results/Combined_AllSubjects.xlsx')

# Inspect Questionnaire File
analyze_excel('/Users/user/Research/HCS_ver4.0/実験後アンケート1（回答）.xlsx')
