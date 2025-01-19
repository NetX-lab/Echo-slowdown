import pandas as pd
import os
import glob
import argparse
import json

# Set up argument parser
parser = argparse.ArgumentParser(description='Merge Excel and CSV based on kernel names.')
parser.add_argument('--output', type=str, default='merged_file.csv', help='Path to save the merged CSV file.')
args = parser.parse_args()

cwd = os.getcwd()

slowdown_file = 'input/slowdown_stats_output_device_0.xlsx'
features_file = 'input/kernel_metric_output.csv'
output_path = args.output

slowdown_kernelname_col = 'kShortName'
# features_kernelname_col = 'Function Name'
features_kernelname_col = 'Kernel Name'

# Read the CSV files into DataFrames
df1 = pd.read_excel(slowdown_file)
df2 = pd.read_csv(features_file)

print('df1.head()', df1.head())

# Initialize pointers for both dataframes
pointer1 = 0
pointer2 = 0
merged_data = []

def cmp_name(name1, name2):
    if name1 == name2:
        return True
    return False

# Loop until one of the DataFrames is fully traversed
while pointer1 < len(df1) and pointer2 < len(df2):
    name1 = df1.loc[pointer1, slowdown_kernelname_col]
    name2 = df2.loc[pointer2, features_kernelname_col]
    
    # print(name1, name2)

    if cmp_name(name1, name2):
        # If the names match, merge the rows
        merged_row = pd.concat([df1.loc[pointer1], df2.loc[pointer2]], axis=0)
        merged_data.append(merged_row)
        pointer1 += 1
        pointer2 += 1
    else:
        # Move pointer1 for now to match the next name in df1
        pointer1 += 1

if pointer2 < len(df2):
    print('last unmatched', pointer2, df2.loc[pointer2, features_kernelname_col])
else:
    print('All found')

# Convert the merged data list to a DataFrame
merged_df = pd.DataFrame(merged_data)

# Save the merged result to a new CSV at the specified location
merged_df.to_csv(output_path, index=False)

print(f"Merged file saved to: {output_path}")