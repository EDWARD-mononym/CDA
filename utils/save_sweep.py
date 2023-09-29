import pandas as pd

def dicts_to_csv(hyperparam, avg_meter, filename, headings):
    all_data = {**hyperparam, **{f"{key}_avg_val": avg_meter[key].val for key in avg_meter.keys()}}
    df = pd.DataFrame([all_data])
    df.to_csv(filename, index=False)