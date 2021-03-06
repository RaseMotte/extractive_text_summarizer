#!/usr/bin/python3

import sys
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: Expecting one argument, the csv file to analyse.")
        exit(1)

    fname = sys.argv[1]
    res = pd.read_csv(fname, index_col=0)

    max_row_id = res[["avg_rl_r", "sampling_size"]].idxmax()
    row = res.iloc[max_row_id]
    print(row["summarizer_name"].values)
    print(row)
