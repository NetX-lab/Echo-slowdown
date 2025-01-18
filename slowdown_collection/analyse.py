import sqlite3
import shutil
import os
import pandas as pd
import re
import glob

def prepare_res_directory():
    res_dir = './res'
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)
    
def process_sqlite_file(original_db, comm_conn, compute_conn):
    processing_db = 'processing-db.sqlite'
    shutil.copyfile(original_db, processing_db)
    print(f"Database {original_db} has been backed up as {processing_db}")
    conn = sqlite3.connect(processing_db)
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN cudaAPIName TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN kShortName TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN kDemangledName TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN duration INTEGER;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN overlap_time INTEGER DEFAULT 0;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN overlap_ratio REAL DEFAULT 0;")

        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET cudaAPIName = (
            SELECT value FROM StringIds
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
            ON r.nameId = StringIds.id
            AND CUPTI_ACTIVITY_KIND_KERNEL.correlationId = r.correlationId);
        ''')

        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET kShortName = (
            SELECT value FROM StringIds WHERE shortName = StringIds.id);
        ''')

        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET kDemangledName = (
            SELECT value FROM StringIds WHERE demangledName = StringIds.id);
        ''')

        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET duration = end - start;
        ''')

        cursor.execute('''
        WITH Overlaps AS (
            SELECT 
                c.deviceId AS device_id,
                c.start AS compute_start,
                c.end AS compute_end,
                SUM(
                    CASE
                        WHEN c.start < m.end AND c.end > m.start THEN
                            MIN(c.end, m.end) - MAX(c.start, m.start)
                        ELSE 0
                    END
                ) AS total_overlap_time
            FROM CUPTI_ACTIVITY_KIND_KERNEL c
            JOIN CUPTI_ACTIVITY_KIND_KERNEL m
            ON c.deviceId = m.deviceId
            AND m.kShortName LIKE '%nccl%'
            AND c.kShortName NOT LIKE '%nccl%'
            GROUP BY c.deviceId, c.start, c.end
        )
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL
        SET overlap_time = (
            SELECT total_overlap_time 
            FROM Overlaps 
            WHERE Overlaps.device_id = CUPTI_ACTIVITY_KIND_KERNEL.deviceId
            AND Overlaps.compute_start = CUPTI_ACTIVITY_KIND_KERNEL.start
            AND Overlaps.compute_end = CUPTI_ACTIVITY_KIND_KERNEL.end
        );
        ''')

        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL
        SET overlap_ratio = CAST(overlap_time AS REAL) / duration
        WHERE duration > 0;
        ''')

        conn.commit()

        base_name = os.path.splitext(os.path.basename(original_db))[0]

        query_other = '''
        SELECT kShortName, kDemangledName, deviceId, duration, overlap_time, overlap_ratio
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE kShortName IS NOT NULL 
            AND kShortName NOT LIKE '%nccl%'
            AND kDemangledName IS NOT NULL
            AND duration > 0
        '''

        df_other = pd.read_sql_query(query_other, conn)
        grouped = df_other.groupby('deviceId')
        
        for device_id, group in grouped:
            group['id'] = range(1, len(group) + 1)
            cols = ['id'] + [col for col in group.columns if col != 'id']
            group = group[cols]

            table_name = f"{base_name}_device_{device_id}"
            group.to_sql(table_name, compute_conn, if_exists='replace', index=False)

            print(f"Data for device {device_id} has been saved to table {table_name} in the compute database.")

        print(f"Data from {original_db} has been successfully saved to separate tables in the main databases.")

    finally:
        conn.close()
        if os.path.exists(processing_db):
            os.remove(processing_db)
            print(f"Processing completed, temporary database file {processing_db} has been deleted \n")

def calculate_slowdown(compute_conn, slowdown_conn):# yet to be optimized
    cursor = compute_conn.cursor()

    # Get all table names from the compute database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Regular expression to match table names and extract the base name
    ws1_pattern = re.compile(r'^(.*)_ws1_device_(\d+)$')  # Matches tables ending in "_ws1_device_{n}"
    other_pattern = re.compile(r'^(.*)_ws\d+_device_(\d+)$')  # Matches tables like "_ws{n}_device_{n}"

    # Group tables by base name
    base_tables = {}
    for table in tables:
        name = table[0]
        ws1_match = ws1_pattern.match(name)
        other_match = other_pattern.match(name)

        if ws1_match:
            base_name = ws1_match.group(1)
            device_id = ws1_match.group(2)
            if base_name not in base_tables:
                base_tables[base_name] = {'ws1': {}, 'others': {}}
            base_tables[base_name]['ws1'][device_id] = name
        elif other_match:
            base_name = other_match.group(1)
            device_id = other_match.group(2)
            if base_name not in base_tables:
                base_tables[base_name] = {'ws1': {}, 'others': {}}
            base_tables[base_name]['others'][device_id] = name
    
    # Check for cases where ws1 is missing for some device_ids
    for base_name, table_group in base_tables.items():
        for device_id in table_group['others']:
            if device_id not in table_group['ws1']:
                print(f"Warning: No matching ws1 table for base name: {base_name} and device {device_id}")
    
    # Calculate slowdown for each base name
    for base_name, table_group in base_tables.items():
        for device_id, ws1_table in table_group['ws1'].items():
            if device_id not in table_group['others']:
                continue  # Skip this device_id if there's no corresponding "other" table

            other_table = table_group['others'][device_id]
            slowdown_table = f"{other_table}_slowdown"

            # SQL query to calculate the slowdown
            calculate_slowdown_sql = f"""
            SELECT
                t1.id,
                t1.kShortName,
                t1.kDemangledName,
                t1.duration AS ground_truth,  -- Duration from ws1 table (ground truth)
                t2.duration,
                t2.overlap_ratio,
                CASE
                    WHEN t2.overlap_ratio = 0 THEN 0  
                    ELSE ((CAST(t2.duration AS REAL) - CAST(t1.duration AS REAL)) / (CAST(t2.overlap_ratio AS REAL) * CAST(t1.duration AS REAL)))
                END AS slowdown
            FROM {ws1_table} t1
            JOIN {other_table} t2 ON t1.id = t2.id AND t1.deviceId = t2.deviceId
            WHERE t1.kDemangledName = t2.kDemangledName;
            """


            # Use pandas to run the SQL query and store the result in a DataFrame
            df_slowdown = pd.read_sql_query(calculate_slowdown_sql, compute_conn)

            # Save the DataFrame to the slowdown database
            df_slowdown.to_sql(slowdown_table, slowdown_conn, if_exists='replace', index=False)
            
            # Save the DataFrame to an Excel file
            excel_filename = f'output/slowdown_{base_name}_device_{device_id}.xlsx'
            df_slowdown.to_excel(excel_filename, index=False)
            print(f"Slowdown result saved to {excel_filename}")

            print(f"Slowdown table {slowdown_table} created in slowdown_results.sqlite.")

    # Commit changes to the slowdown database
    slowdown_conn.commit()


def calculate_and_save_averages(compute_conn):
    cursor = compute_conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    results_conn = sqlite3.connect('./res/results.sqlite')

    device_groups = {}
    for table_name_tuple in tables:
        table_name = table_name_tuple[0]
        match = re.search(r'device_(\d+)', table_name)
        if match:
            device_id = match.group(1)
            if device_id not in device_groups:
                device_groups[device_id] = []
            device_groups[device_id].append(table_name)

    for device_id, device_tables in device_groups.items():
        ground_truth_table = f'ground_truth_device_{device_id}'
        overlap_table = f'overlap_device_{device_id}'
        ground_truth_exists = False
        overlap_exists = False

        # Check if both tables exist before running the query
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (ground_truth_table,))
        if cursor.fetchone():
            ground_truth_exists = True

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (overlap_table,))
        if cursor.fetchone():
            overlap_exists = True

        if ground_truth_exists and overlap_exists:
            combined_query = f'''
            SELECT gt.kShortName, gt.kDemangledName, gt.deviceId, gt.ground_truth, 
                   ol.avg_duration, ol.overlap_rate
            FROM {ground_truth_table} AS gt
            LEFT JOIN {overlap_table} AS ol
            ON gt.kShortName = ol.kShortName AND 
               gt.kDemangledName = ol.kDemangledName AND 
               gt.deviceId = ol.deviceId
            '''
            combined_df = pd.read_sql_query(combined_query, results_conn)

            # Ensure that there are no divisions by zero
            combined_df = combined_df[(combined_df['overlap_rate'] > 0) & (combined_df['ground_truth'] > 0)]

            # Calculate slowdown and append it to the DataFrame
            combined_df['slowdown'] = (combined_df['avg_duration'] - (1 - combined_df['overlap_rate']) * combined_df['ground_truth']) / (combined_df['overlap_rate'] * combined_df['ground_truth']) - 1

            combined_df['slowdown'] = combined_df['slowdown'].replace([float('inf'), -float('inf')], float('nan'))

            combined_df.to_excel(f'slowdown_device_{device_id}.xlsx', index=False)
            print(f"Combined durations with slowdown have been saved to 'slowdown_device_{device_id}.xlsx'")

        else:
            print(f"Warning: Missing required table(s) for device {device_id}. Ground truth exists: {ground_truth_exists}, Overlap exists: {overlap_exists}")

    results_conn.close()

def main():
    prepare_res_directory()
    
    sqlite_files = glob.glob("./temp/*.sqlite")

    comm_conn = sqlite3.connect('./res/comm_results.sqlite')
    compute_conn = sqlite3.connect('./res/compute_results.sqlite')
    slowdown_db = sqlite3.connect('./res/slowdown_results.sqlite')

    for sqlite_file in sqlite_files:
        print(sqlite_file)
        process_sqlite_file(sqlite_file, comm_conn, compute_conn)

    # Calculate the duration slowdown between tables in compute.sqlite and save to slowdown_results.sqlite
    calculate_slowdown(compute_conn, slowdown_db)

    # calculate_and_save_averages(compute_conn) 

    comm_conn.close()
    compute_conn.close()
    slowdown_db.close()

if __name__ == '__main__':
    main()