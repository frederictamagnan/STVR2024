import glob
import json

def merge_files():
    # Define the path to the directory containing the input files
    directory_path = './results/jsonl'

    # Find all files in the directory that start with "output_"
    file_paths = glob.glob(f'{directory_path}/output_*')

    # Create an empty list to store the data from all files
    all_data = []

    # Load the data from each file and append it to the list
    for file_path in file_paths:
        with open(file_path) as f:
            data = f.readlines()
        all_data.extend([json.loads(line) for line in data])

    # Dump the list of data to a new file
    with open('./results/jsonl/all_data_1010.jsonl', 'w') as f:
        for data in all_data:
            json.dump(data, f)
            f.write('\n')

def modify():

    input_file = './results/jsonl/all_data_07093.jsonl'
    output_file = './results/jsonl/all_data_v2.jsonl'

    with open(input_file, 'r') as reader, open(output_file, 'w') as writer:
        for line in reader:
            data = json.loads(line)
            data['name_exp'] = data['clustering_pipeline'] + '_' + data['sample_heuristic']
            writer.write(json.dumps(data) + '\n')

def conc():
    input_file1 =  './results/jsonl/all_data_07093.jsonl'
    input_file2 =  './results/jsonl/all_data_07092.jsonl'
    output_file =  './results/jsonl/all_data_07094.jsonl'

    with open(output_file, 'w') as writer:
        # Process first input file
        with open(input_file1, 'r') as reader:
            for line in reader:
                writer.write(line)

        # Process second input file
        with open(input_file2, 'r') as reader:
            for line in reader:
                writer.write(line)


def delete_groupname():

    target_name_exp = "AEplusRS"
   


    # Define the value you want to filter by
    

    # Specify the input and output file names
    input_file = './results/jsonl/all_data_2609.jsonl'
    output_file ='./results/jsonl/all_data_26092.jsonl'

    # Open the input and output files
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                # Load each line as a JSON object
                data = json.loads(line)
                
                # Check if the "name_exp" matches the target value
                if data['name_exp'] != target_name_exp:
                    # If not, write the line to the output file
                    outfile.write(json.dumps(data) + '\n')
            except json.JSONDecodeError as e:
                # Handle any JSON decoding errors here (skip or log)
                print(f"JSON decoding error: {e}")

    print(f"Rows with 'name_exp' equal to '{target_name_exp}' removed. Filtered data saved to '{output_file}'.")


if __name__=="__main__":
    merge_files()
    # delete_groupname()
    # conc()