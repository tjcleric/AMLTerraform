import pandas as pd

def combine_csv_files(file_list, outputfile):
    print("Reading splits...")
    dfs = [pd.read_csv(f, dtype=str) for f in file_list]

    combined_df = pd.concat(dfs, ignore_index=True)

    print("Saving to CSV...")
    combined_df.to_csv(outputfile, index=False)

    print(f"CSV files successfully combined and saved as {outputfile}")

filepath = "data/"

# combine_csv_files(
#     [f"{filepath}val_split.csv", f"{filepath}test_split.csv", f"{filepath}train_split.csv"],
#     f"{filepath}formatted_transactions1.csv"
# )

# combine_csv_files(
#     [f"{filepath}unformatted_remaining_split.csv", f"{filepath}unformatted_test_split.csv"],
#     f"{filepath}unformatted.csv"
# )

# combine_csv_files(
#     [f"{filepath}unformatted_remaining_split.csv", f"{filepath}augmented.csv"],
#     f"{filepath}unformatted.csv"
# )
