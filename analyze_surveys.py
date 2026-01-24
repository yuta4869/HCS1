import pandas as pd
import os

files = {
    'Fixed': '/Users/user/Research/HCS_ver4.0/実験後アンケート1（回答）.xlsx',
    'PID': '/Users/user/Research/HCS_ver4.0/実験後アンケート4（回答）.xlsx',
    'Robust': '/Users/user/Research/HCS_ver4.0/実験後アンケート7（回答）.xlsx'
}

questions = {
    2: 'Satisfaction',
    3: 'Stress',
    4: 'Fatigue',
    5: 'Enjoyment',
    6: 'Security',
    7: 'Discomfort',
    8: 'Boredom',
    9: 'Surprise',
    10: 'Excitement',
    11: 'Understood',
    12: 'Empathy_Received', # 相手が自分の話に共感してくれた
    13: 'Empathy_Gave',     # 相手に共感できた
    14: 'Listened_To',      # 相手がどれくらい話を聞いていたか
    15: 'Adapted',          # 相手は自分の状態に合わせて話し方を変えていたか
    16: 'Length_Felt',
    17: 'Wait_Time',
    18: 'Repeat_Desire',
    19: 'Intonation_Change'
}

panas_pa_indices = [21, 23, 26, 27, 29, 31, 32, 33]
panas_na_indices = [20, 22, 24, 25, 28, 30, 34, 35]

results = {}

print(f"{'Metric':<20} | {'Fixed':<8} | {'PID':<8} | {'Robust':<8}")
print("-" * 50)

# Load dataframes first
dfs = {}
for condition, path in files.items():
    if os.path.exists(path):
        try:
            dfs[condition] = pd.read_excel(path)
        except Exception as e:
            print(f"Error loading {condition}: {e}")
            dfs[condition] = None
    else:
        print(f"File not found for {condition}")
        dfs[condition] = None

# Calculate means for questions
for idx, name in questions.items():
    row = [name]
    for cond in ['Fixed', 'PID', 'Robust']:
        df = dfs[cond]
        if df is not None:
            # Column indices are 0-based
            val = df.iloc[:, idx].mean()
            row.append(f"{val:.2f}")
        else:
            row.append("N/A")
    print(f"{row[0]:<20} | {row[1]:<8} | {row[2]:<8} | {row[3]:<8}")

print("-" * 50)

# Calculate PANAS
row_pa = ["PANAS_PA"]
row_na = ["PANAS_NA"]

for cond in ['Fixed', 'PID', 'Robust']:
    df = dfs[cond]
    if df is not None:
        # Calculate sum for each subject first, then mean
        pa_scores = df.iloc[:, panas_pa_indices].sum(axis=1)
        na_scores = df.iloc[:, panas_na_indices].sum(axis=1)
        
        row_pa.append(f"{pa_scores.mean():.2f}")
        row_na.append(f"{na_scores.mean():.2f}")
    else:
        row_pa.append("N/A")
        row_na.append("N/A")

print(f"{row_pa[0]:<20} | {row_pa[1]:<8} | {row_pa[2]:<8} | {row_pa[3]:<8}")
print(f"{row_na[0]:<20} | {row_na[1]:<8} | {row_na[2]:<8} | {row_na[3]:<8}")
