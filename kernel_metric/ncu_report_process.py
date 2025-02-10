import pandas as pd
import glob
import os

# Get the current directory
current_directory = os.getcwd()
temp_directory = os.path.join(current_directory, 'temp')
output_directory = os.path.join(current_directory, 'output')

# Search for file in the current directory ending with '_details.csv'
details_csv = glob.glob(os.path.join(temp_directory, "*_details.csv"))[0]
kshortname_csv = glob.glob(os.path.join(temp_directory, "*_kshortname.csv"))[0]

details_df = pd.read_csv(details_csv)
kshortname_df = pd.read_csv(kshortname_csv)

"""
Column name

Index(['ID', 'Process ID', 'Process Name', 'Host Name',
       'thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg',
       'Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg',
       'Kernel Name', 'Context', 'Stream', 'Block Size', 'Grid Size', 'Device',
       'CC', 'Section Name', 'Metric Name', 'Metric Unit', 'Metric Value',
       'Rule Name', 'Rule Type', 'Rule Description', 'Estimated Speedup Type',
       'Estimated Speedup'],
      dtype='object')
"""

"""
Section + Metric names

unique_section_metric_name = df[['Section Name', 'Metric Name']].drop_duplicates()
unique_section_metric_name = [tuple(x) for x in unique_section_metric_name.to_numpy()]
for t in unique_section_metric_name:
    print(t)

('GPU Speed Of Light Throughput', 'DRAM Frequency')
('GPU Speed Of Light Throughput', 'SM Frequency')
('GPU Speed Of Light Throughput', 'Elapsed Cycles')
('GPU Speed Of Light Throughput', 'Memory Throughput')
('GPU Speed Of Light Throughput', 'DRAM Throughput')
('GPU Speed Of Light Throughput', 'Duration')
('GPU Speed Of Light Throughput', 'L1/TEX Cache Throughput')
('GPU Speed Of Light Throughput', 'L2 Cache Throughput')
('GPU Speed Of Light Throughput', 'SM Active Cycles')
('GPU Speed Of Light Throughput', 'Compute (SM) Throughput')
('SpeedOfLight', nan)
('Memory Workload Analysis', 'Memory Throughput')
('Memory Workload Analysis', 'Mem Busy')
('Memory Workload Analysis', 'Max Bandwidth')
('Memory Workload Analysis', 'L1/TEX Hit Rate')
('Memory Workload Analysis', 'L2 Compression Success Rate')
('Memory Workload Analysis', 'L2 Compression Ratio')
('Memory Workload Analysis', 'L2 Hit Rate')
('Memory Workload Analysis', 'Mem Pipes Busy')
('Launch Statistics', 'Block Size')
('Launch Statistics', 'Function Cache Configuration')
('Launch Statistics', 'Grid Size')
('Launch Statistics', 'Registers Per Thread')
('Launch Statistics', 'Shared Memory Configuration Size')
('Launch Statistics', 'Driver Shared Memory Per Block')
('Launch Statistics', 'Dynamic Shared Memory Per Block')
('Launch Statistics', 'Static Shared Memory Per Block')
('Launch Statistics', '# SMs')
('Launch Statistics', 'Threads')
('Launch Statistics', 'Uses Green Context')
('Launch Statistics', 'Waves Per SM')
('LaunchStats', nan)
('Occupancy', 'Block Limit SM')
('Occupancy', 'Block Limit Registers')
('Occupancy', 'Block Limit Shared Mem')
('Occupancy', 'Block Limit Warps')
('Occupancy', 'Theoretical Active Warps per SM')
('Occupancy', 'Theoretical Occupancy')
('Occupancy', 'Achieved Occupancy')
('Occupancy', 'Achieved Active Warps Per SM')
('Occupancy', nan)

"""


unique_ids = details_df['ID'].unique()

output_dictionaries = []

for id in unique_ids:
    df_kernel = details_df[details_df['ID'] == id]

    # Filter out data based on conditions and check if the result is not empty
    def get_metric_value(section_name, metric_name):
        filtered_df = df_kernel[(df_kernel['Section Name'] == section_name) & (df_kernel['Metric Name'] == metric_name)].reset_index(drop=True)
        if not filtered_df.empty:
            return filtered_df.at[0, 'Metric Value']
        else:
            return None  # or a default value like 0 or 'N/A'

    new_row = {
        'ID': df_kernel.reset_index(drop=True).at[0, 'ID'],
        'Compute throughput': get_metric_value('GPU Speed Of Light Throughput', 'Compute (SM) Throughput'),
        'SM': get_metric_value('Launch Statistics', '# SMs'),
        'Memory throughput': get_metric_value('GPU Speed Of Light Throughput', 'Memory Throughput'),
        'DRAM throughput': get_metric_value('GPU Speed Of Light Throughput', 'DRAM Throughput'),
        'Achieved occupancy': get_metric_value('Occupancy', 'Achieved Occupancy'),
        'Maximum occupancy': get_metric_value('Occupancy', 'Theoretical Occupancy'),
        'L1 hit rate': get_metric_value('Memory Workload Analysis', 'L1/TEX Hit Rate'),
        'L2 hit rate': get_metric_value('Memory Workload Analysis', 'L2 Hit Rate'),
    }

    output_dictionaries.append(new_row)

df_output = pd.DataFrame(output_dictionaries)

kshortname_df = kshortname_df.drop_duplicates(subset='ID', keep='first')
kshortname_df_filtered = kshortname_df[['ID', 'Kernel Name']]

merged_df = pd.merge(kshortname_df_filtered, df_output, on='ID', how='inner')

merged_df.to_csv(os.path.join(output_directory, 'kernel_metric_output.csv'), index=False)