import random


def train_datasplit(mode,train_data_csv,train_data_encode,spt_num,seed,no_shuffle=False):
    """
    notice: you should run create_data.py to get the train_data_csv and train_data_encode file
    
    Parameters:
        param train_data_csv: a csv file eg.[compound_iso_smiles,target_sequence,affinity]
        param train_data_encode: a file has been encoded based on the train_data_csv file  
              eg. Data(x, edge_index, y, target, c_size)
        param spt_num: the size of support set;
        param qry_num: the size of query set;
        
    Returns:
        a dictionary ,
        e.g. [[support samples],[query samples]]
    """
    random.seed(seed)
    # to_num=spt_num+qry_num

    indices_dict = {}
    for value in train_data_csv['target_sequence'].unique():
        indices = train_data_csv.index[train_data_csv['target_sequence'] == value].tolist()
        indices_dict[value] = indices
    
    F_idx_data={}
    F_data={} 
    for k in indices_dict.keys():
        F_data[k]=[[],[]]  
        if no_shuffle:
            F_idx_data[k]=[[],[]]
            for v in indices_dict[k]:
                F_data[k][0].append(train_data_encode[v])
                F_idx_data[k][0].append(v)
        else:
            for v in indices_dict[k]:
                F_data[k][0].append(train_data_encode[v])

    if mode=='num':
        for i in F_data.keys():
            if no_shuffle:
                combined = list(zip(F_data[i][0], F_idx_data[i][0]))
                # random.shuffle(combined)
                F_data[i][0], F_idx_data[i][0] = zip(*combined)

                F_data[i][0]=list(F_data[i][0])
                F_data[i][1]=F_data[i][0][spt_num:]   
                F_data[i][0]=F_data[i][0][:spt_num]  

                F_idx_data[i][0]=list(F_idx_data[i][0])
                F_idx_data[i][1]=F_idx_data[i][0][spt_num:]   
                F_idx_data[i][0]=F_idx_data[i][0][:spt_num]
                # random.shuffle(F_idx_data[i][1])
            else:
                random.shuffle(F_data[i][0])
                F_data[i][1]=F_data[i][0][spt_num:]  
                F_data[i][0]=F_data[i][0][:spt_num]

    elif mode=='ratio':
        for i in F_data.keys():
            random.shuffle(F_data[i][0])
            num=len(F_data[i][0])
            n_spt=int(num*spt_num)
            F_data[i][1]=F_data[i][0][n_spt:]  
            F_data[i][0]=F_data[i][0][:n_spt]   
    
    return F_data

