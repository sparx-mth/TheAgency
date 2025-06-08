import os
import json


def combine_json_files(base_path, output_file):
    combined_results = {}

    # Traverse the directory structure
    for root, dirs, files in os.walk(base_path):
        if 'ConfigFiles' in dirs:
            config_files_path = os.path.join(root, 'ConfigFiles')
            summery_file_path = os.path.join(config_files_path, 'Summery.json')

            if os.path.isfile(summery_file_path):
                main_folder = os.path.basename(root)

                # Read the Summery.json file
                with open(summery_file_path, 'r') as f:
                    summery_data = json.load(f)

                # Add the summery data under the field named after the main folder
                combined_results[main_folder] = summery_data

    # Write the combined results to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=4)


# Specify the base path and output file
base_path = 'C:/Users/Danielco/AppData/LocalLow/AsgardSystems/SwarmUI/Results/final_20240728213826345'
output_file = 'Combined_Results.json'

# Combine the JSON files
combine_json_files(base_path, output_file)
