import pandas as pd

def dicts_to_csv(hyperparam, avg_meter, filename):
    all_data = {**hyperparam, **{f"{key}_avg_val": avg_meter[key].val for key in avg_meter.keys()}}
    new_row = pd.DataFrame([all_data])
    
    # Append new_row to CSV, without writing header if file already exists
    with open(filename, 'a') as f:
        new_row.to_csv(f, header=f.tell() == 0, index=False)