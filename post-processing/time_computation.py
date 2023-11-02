import json

filename='./results/jsonl/all_data_2609.jsonl'
total_time = 0
line_count = 0
with open(filename, 'r') as file:
        for line in file:
        # Parse the JSON object from the line
            data = json.loads(line)
            
            # Check if the "time" key exists in the JSON object
            if "time" in data:
                # Add the time value to the total_time
                total_time += data["time"]
                
                # Increment the line count
                line_count += 1
if line_count > 0:
    mean_time = total_time / line_count

    print(f"Mean time over all lines: {mean_time/60}")
else:
    print("No 'time' values found in the JSON Lines file.")

print(line_count)