import pandas as pd # You might need: pip install pandas

try:
    df = pd.read_csv('logs/usage_stats.csv')
    summary = df['Gesture'].value_counts()
    print("--- Vision Assistant Usage Summary ---")
    print(summary)
    print("--------------------------------------")
except FileNotFoundError:
    print("No logs found. Wave at the camera first!")