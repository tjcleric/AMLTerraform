import csv
import pandas as pd

# def separate_by_edgeid(input_full, input_test, output_testcsv, output_remainingcsv):
#     print("Reading CSVs...")
#     df_full = pd.read_csv(input_full)
#     df_test_ids = pd.read_csv(input_test)

#     test_indices = set(df_test_ids["EdgeID"].astype(int))

#     mask = df_full.index.isin(test_indices)

#     df_test = df_full[mask]
#     df_remaining = df_full[~mask]

#     print("Saving to CSVs...")
#     df_test.to_csv(output_testcsv, index=False)
#     df_remaining.to_csv(output_remainingcsv, index=False)

#     print(f"Wrote {len(df_test)} rows to {output_testcsv}")
#     print(f"Wrote {len(df_remaining)} rows to {output_remainingcsv}")

import pandas as pd

def separate_by_edgeid(input_full, input_test, output_testcsv, output_remainingcsv):
    print("Reading CSVs...")
    df_full = pd.read_csv(input_full, dtype=str)
    df_test_ids = pd.read_csv(input_test)
    test_indices = df_test_ids["EdgeID"].astype(int).tolist()
    # df_test = df_full.iloc[test_indices]
    df_test = df_full.iloc[test_indices].sort_index()
    df_remaining = df_full.drop(index=test_indices)

    print("Saving to CSVs...")
    df_test.to_csv(output_testcsv, index=False)
    df_remaining.to_csv(output_remainingcsv, index=False)
    print(f"Wrote {len(df_test)} rows to {output_testcsv}")
    print(f"Wrote {len(df_remaining)} rows to {output_remainingcsv}")

filepath = "data"
input_full = f"{filepath}/HI-Small_Trans.csv"
input_test = f"{filepath}/test_split.csv"
output_testcsv = f"{filepath}/unformatted_test_split.csv"
output_remainingcsv = f"{filepath}/unformatted_remaining_split.csv"

separate_by_edgeid(input_full, input_test, output_testcsv, output_remainingcsv)






