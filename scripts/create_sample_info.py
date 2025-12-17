import pandas as pd
import numpy as np

n_samples = 96
location = 8
well_ids = [f'{location}{let}{id}' for id in range(1, 13) for let in 'ABCDEFGH'] 
sample_ids = [f"{i+1}" for i in range(n_samples)]
df = pd.DataFrame({
    "sample ID": sample_ids,
    "well ID": well_ids
})
df.to_csv('config/info/sample_info.csv', index=False)
