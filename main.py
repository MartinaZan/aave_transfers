import os
import pickle
from utils import *
from RQ1 import run_RQ1
from RQ2 import run_RQ2
from RQ3 import run_RQ3

###################################################################################

# IDs for aave_transfers JSON files
START_ID = 1
END_ID = 69

###################################################################################

def main():
    # Import data
    df_transfers = import_data_range(START_ID, END_ID)
    df_transfers = df_transfers.sort_values("timestamp").reset_index(drop=True)

    # Compute graph metrics (only if they haven't been computed before)
    if not os.path.exists("metrics.json") or not os.path.exists("snapshots.pkl"):
        metrics, snapshots = compute_metrics(df_transfers)

        with open(f"metrics.json", "w") as f:
            json.dump(metrics, f, default=str)
        with open(f"snapshots.pkl", "wb") as f:
                pickle.dump(snapshots, f)
    else:
        with open("metrics.json") as f:
            metrics = json.load(f)
        with open("snapshots.pkl", "rb") as f:
            snapshots = pickle.load(f)

    # Answer RQ1, RQ2, RQ3
    df_transfers["value"] = pd.to_numeric(df_transfers["value"], errors="coerce")
    run_RQ1(metrics, df_transfers)
    run_RQ2(df_transfers, snapshots)
    run_RQ3(metrics)

if __name__ == '__main__':
    main()