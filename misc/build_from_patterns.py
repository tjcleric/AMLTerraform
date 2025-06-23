import pandas as pd
import time

columns = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering"
]

# input_file = 'patterns.txt'
input_file = input("Enter path to txt file containing patterns")

epoch_time = int(time.time())
output_file = f'patterns_{epoch_time}.csv'

transactions = []
with open(input_file, 'r') as f:
    inside_block = False
    for line in f:
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            inside_block = True
            continue
        elif line.startswith("END LAUNDERING ATTEMPT"):
            inside_block = False
            continue
        elif inside_block and line:
            transactions.append(line.split(','))

df = pd.DataFrame(transactions, columns=columns)

df.to_csv(output_file, index=False)
print(f"Saved {len(df)} transactions to {output_file}")
