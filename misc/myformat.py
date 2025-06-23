import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f, join, sort
import os

def format_transactions(in_path, out_path = None):
    if out_path is None:
        out_path = os.path.join(os.path.dirname(in_path), "formatted_transactions.csv")

    raw = dt.fread(in_path, columns=dt.str32)

    currency = dict()
    paymentFormat = dict()
    account = dict()

    def get_dict_val(name, collection):
        if name in collection:
            val = collection[name]
        else:
            val = len(collection)
            collection[name] = val
        return val

    header = "EdgeID,from_id,to_id,Timestamp," \
             "Amount Sent,Sent Currency,Amount Received,Received Currency," \
             "Payment Format,Is Laundering\n"

    first_ts = -1

    with open(out_path, 'w') as writer:
        writer.write(header)
        for i in range(raw.nrows):
            datetime_object = datetime.strptime(raw[i, "Timestamp"], '%Y/%m/%d %H:%M')
            ts = datetime_object.timestamp()

            if first_ts == -1:
                start_time = datetime(datetime_object.year, datetime_object.month, datetime_object.day)
                first_ts = start_time.timestamp() - 10
                print(f"Selected first timestamp: {datetime_object} (UNIX: {first_ts})")

            ts -= first_ts

            cur1 = get_dict_val(raw[i, "Receiving Currency"], currency)
            cur2 = get_dict_val(raw[i, "Payment Currency"], currency)
            fmt = get_dict_val(raw[i, "Payment Format"], paymentFormat)

            from_acc_id_str = raw[i, "From Bank"] + raw[i, 2]
            to_acc_id_str = raw[i, "To Bank"] + raw[i, 4]

            from_id = get_dict_val(from_acc_id_str, account)
            to_id = get_dict_val(to_acc_id_str, account)

            amount_received = float(raw[i, "Amount Received"])
            amount_paid = float(raw[i, "Amount Paid"])
            is_laundering = int(raw[i, "Is Laundering"])

            line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % (
                i, from_id, to_id, ts, amount_paid, cur2, amount_received, cur1, fmt, is_laundering
            )

            writer.write(line)

    formatted = dt.fread(out_path)
    formatted = formatted[:, :, sort(3)]
    formatted.to_csv(out_path)

    return out_path
