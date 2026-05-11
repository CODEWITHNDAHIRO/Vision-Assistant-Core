import csv
from datetime import datetime
import os

# Create a logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

def log_gesture(gesture_name):
    file_path = 'logs/usage_stats.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Add header if it's a new file
        if not file_exists:
            writer.writerow(['Timestamp', 'Gesture'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gesture_name])

# Inside your start_assistant() loop, call this:
# if results.multi_hand_landmarks:
#     log_gesture("Hand_Detected")