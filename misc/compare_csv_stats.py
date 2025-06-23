import pandas as pd

def count_rows_in_csv(file_path):
    df = pd.read_csv(file_path)
    
    empty_data_rows = df[df.isnull().any(axis=1)]
    
    string_columns = df.select_dtypes(include=['object']).columns
    empty_string_rows = df[df[string_columns].apply(lambda x: x.str.strip() == '').any(axis=1)]
    
    empty_data_rows = pd.concat([empty_data_rows, empty_string_rows]).drop_duplicates()
    
    if not empty_data_rows.empty:
        print(f"Rows with empty or null data in {file_path}:")
        print(empty_data_rows)
    else:
        print(f"No rows with empty or null data in {file_path}")
    
    duplicate_rows = df[df.duplicated(subset=['Account', 'Account.1', 'Timestamp', 'Amount Received'], keep=False)]
    
    if not duplicate_rows.empty:
        print(f"Duplicate rows in {file_path}:")
        print(duplicate_rows)
    else:
        print(f"No duplicate rows based on Account, Account.1, Timestamp and Amount Received in {file_path}")
    
    return len(df)

def count_total_rows(csv_file1, csv_file2):
    count1 = count_rows_in_csv(csv_file1)
    count2 = count_rows_in_csv(csv_file2)
    
    print(f"augmented: {count1}")
    print(f"normal: {count2}")
    
    return count1 + count2

# csv_file1 = r'C:\Users\Temp\Research_Project\project_code\data\augmented.csv'
# csv_file2 = r'C:\Users\Temp\Research_Project\project_code\data\HI-Small_Trans.csv'
csv_file1 = input("Enter path to first CSV file: ")
csv_file2 = input("Enter path to second CSV file: ")

total_rows = count_total_rows(csv_file1, csv_file2)
print(f"Total rows in both CSV files: {total_rows}")
