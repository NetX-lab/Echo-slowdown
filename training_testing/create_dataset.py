import os
import pandas as pd

# Define the input folder where the merged file CSV files are stored
input_folder_path = 'input/train_csv'

# Define the output file name for features and target
output_file_name = 'output/train_dataset.csv'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]

# Initialize an empty list to hold DataFrames
dfs = []

# Loop through the CSV files and append each to the list
for file in csv_files:
    file_path = os.path.join(input_folder_path, file)
    try:
        df = pd.read_csv(file_path)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Check if any DataFrames were read
if dfs:
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)

    # merged_df.to_csv('testttttt.csv', index=False)

    # threshold = 1e-5
    # merged_df = merged_df[merged_df['overlap_ratio'].abs() > threshold] # remove all rows with overlap_ratio close to 0

    # merged_df.to_csv('testttttt2.csv', index=False)

    # Set display options to show full content
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)  # Don't wrap DataFrame

    # Print the head of the merged DataFrame
    print("Merged DataFrame head:")
    print(merged_df.head())

    # Separate features and target
    X = merged_df.drop(columns=['id', 'kShortName', 'kDemangledName', 'duration', 'overlap_ratio', 'slowdown', 'ID', 'Kernel Name', 'SM'])
    y = merged_df['slowdown']

    # Combine X and y into a single DataFrame for saving
    final_df = pd.concat([X, y], axis=1)

    # Save the features and target to a CSV file
    final_df.to_csv(output_file_name, index=False)

    # Print a confirmation message
    print(f"Features and target saved to {output_file_name}")

    # Optional: Print shapes of X and y
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
else:
    print("No DataFrames to merge.")