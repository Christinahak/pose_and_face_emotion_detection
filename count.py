import json
from collections import Counter

data_file = 'collected_state_data.jsonl' # Or your actual filename
state_counts = Counter()

try:
    with open(data_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'target_state' in data:
                    state_counts[data['target_state']] += 1
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line.strip()}")

    print("Data Distribution:")
    for state, count in state_counts.items():
        print(f"- {state}: {count} instances")

except FileNotFoundError:
    print(f"Error: Data file not found at {data_file}")