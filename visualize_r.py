import subprocess
import re
import matplotlib.pyplot as plt
import os
import shutil # For creating/checking the output directory

# --- Configuration ---
cpp_executable = "./main"
resolution = 20
# datasets = ["dataset/simple","dataset/SinaNet","dataset/Cora","dataset/CiteSeer","dataset/PubMed","dataset/Reddit"]
datasets = []
base_dataset_directory = "dataset"
for item in os.listdir(base_dataset_directory):
    item_path = os.path.join(base_dataset_directory, item)
    if os.path.isdir(item_path):
        datasets.append(item_path)
output_plot_directory = "plots_r"
# --- Main Script ---

# Ensure the C++ executable exists and is executable
if not (os.path.exists(cpp_executable) and os.access(cpp_executable, os.X_OK)):
    print(f"Error: C++ executable '{cpp_executable}' not found or not executable.")
    print("Please check the path and permissions.")
    exit()

# Create the output directory if it doesn't exist
if not os.path.exists(output_plot_directory):
    os.makedirs(output_plot_directory)
    print(f"Created output directory: {output_plot_directory}")

for dataset_path in datasets:
    print(f"\nProcessing dataset: {dataset_path}...")

    # Construct the command
    command = [cpp_executable, str(resolution), dataset_path, str(resolution)]

    try:
        # Run the C++ program
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        # print("\n--- C++ Program Output ---") # Optional: uncomment to see full output
        # print(output)
        # print("--------------------------\n")

        # --- Parse the output ---
        community_max_distances = []
        global_max_distance_str = None
        average_max_distance_str = None

        # Regex to find community max distances
        community_regex = r"Community \d+: .*? Max distance = ([\d.]+)"
        # Regex to find global max distance
        global_max_regex = r"Global maximum distance: ([\d.]+)"
        # Regex to find average max distance
        average_max_regex = r"Average max distance per community: ([\d.]+)"

        for line in output.splitlines():
            match_community = re.search(community_regex, line)
            if match_community:
                community_max_distances.append(float(match_community.group(1)))

            match_global = re.search(global_max_regex, line)
            if match_global:
                global_max_distance_str = match_global.group(1)

            match_average = re.search(average_max_regex, line)
            if match_average:
                average_max_distance_str = match_average.group(1)

        if not community_max_distances:
            print(f"No community max distances found for dataset: {dataset_path}. Skipping plot.")
            continue

        # Convert summary stats to float, if found
        global_max_val = float(global_max_distance_str) if global_max_distance_str else None
        average_max_val = float(average_max_distance_str) if average_max_distance_str else None

        # --- Plotting ---
        plt.figure(figsize=(12, 7)) # Slightly adjusted size for saving
        plt.hist(community_max_distances, bins='auto', color='skyblue', edgecolor='black', alpha=0.7)

        legend_labels = []
        if global_max_val is not None:
            plt.axvline(global_max_val, color='r', linestyle='dashed', linewidth=2)
            legend_labels.append(f'Global Max: {global_max_val:.4f}')
        if average_max_val is not None:
            plt.axvline(average_max_val, color='g', linestyle='dashed', linewidth=2)
            legend_labels.append(f'Avg. Max: {average_max_val:.4f}')

        if legend_labels:
            plt.legend(legend_labels)

        sanitized_dataset_name = os.path.basename(dataset_path).replace('/', '_').replace('\\', '_')
        plot_filename = f"histogram_{sanitized_dataset_name}.png"
        plot_filepath = os.path.join(output_plot_directory, plot_filename)

        plt.title(f'Histogram of Community Max Distances for {os.path.basename(dataset_path)}')
        plt.xlabel('Max Distance in Community')
        plt.ylabel('Number of Communities')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()

        plt.savefig(plot_filepath)
        print(f"Plot saved to: {plot_filepath}")
        plt.close() # Close the figure to free memory

    except subprocess.CalledProcessError as e:
        print(f"Error running C++ program for dataset {dataset_path}:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
    except FileNotFoundError:
        print(f"Error: The C++ executable '{cpp_executable}' was not found.")
        print("Please ensure the path is correct and the file exists.")
        break # Stop if the executable is not found
    except Exception as e:
        print(f"An unexpected error occurred while processing {dataset_path}: {e}")

print(f"\nAll datasets processed. Plots are saved in '{output_plot_directory}'.")