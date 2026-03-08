import random

def train_datasplit(train_data_csv, train_data_encode, nums):
    groups = train_data_csv.groupby('target_sequence')
    F_data = {}
    for target, group_df in groups:
        indices = group_df.index.tolist()
        random.shuffle(indices)
        encoded_samples = [train_data_encode[i] for i in indices]
        set1 = encoded_samples[:nums]
        set2 = encoded_samples[nums:]  
        F_data[target] = [set1, set2]
    return F_data

