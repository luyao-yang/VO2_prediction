import numpy as np
from torch.utils.data import Dataset
#TODO
import pdb

class TFT_Dataset(Dataset):
    def __init__(self, data, id_column, time_column, target_column, 
                 input_columns, encoder_steps, decoder_steps):
        """
          data (pd.DataFrame): dataframe containing raw data
          entity_column (str): name of column containing entity data
          time_column (str): name of column containing date data
          target_column (str): name of column we need to predict
          input_columns (list): list of string names of columns used as input
          encoder_steps (int): number of known past time steps used for forecast. Equivalent to size of LSTM encoder
          decoder_steps (int): number of input time steps used for each forecast date. Equivalent to the width N of the decoder
        """
        
        self.encoder_steps = encoder_steps
             
        inputs = []
        outputs = []
        entity = []
        time = []
        
        
        # 对于16个entity，不停的遍历每个entity，每个entity都是一个user id
        # encoder_steps: 175  decoder_step: 180
        for e in data[id_column].unique():
          
          entity_group = data[data[id_column]==e]
          # print(entity_group)
            
          # train: 1096
          # test: 365
          data_time_steps = len(entity_group) # 对于每一个entity，有多少个day，时间细粒度的数量 
          
          # pdb.set_trace()
          if data_time_steps >= decoder_steps:
            x = entity_group[input_columns].values.astype(np.float32)
            inputs.append(np.stack([x[i:data_time_steps - (decoder_steps - 1) + i, :] for i in range(decoder_steps)], axis=1))
            # print(len(inputs)) 16
            # print(inputs[0].shape) # (917, 180, 10) (186, 180, 10)

            y = entity_group[[target_column]].values.astype(np.float32)
            outputs.append(np.stack([y[i:data_time_steps - (decoder_steps - 1) + i, :] for i in range(decoder_steps)], axis=1))
            # print(len(outputs)) # 16
            # print(outputs[0].shape) # (917, 180, 1)

            e = entity_group[[id_column]].values.astype(np.float32)
            entity.append(np.stack([e[i:data_time_steps - (decoder_steps - 1) + i, :] for i in range(decoder_steps)], axis=1))
            # print(entity[0].shape) # (917, 180, 1)

            t = entity_group[[time_column]].values.astype(np.int64)
            time.append(np.stack([t[i:data_time_steps - (decoder_steps - 1) + i, :] for i in range(decoder_steps)], axis=1))
            # print(time[0].shape) # (917, 180, 1)

        # pdb.set_trace()
        self.inputs = np.concatenate(inputs, axis=0)
        self.outputs = np.concatenate(outputs, axis=0)[:,encoder_steps:,:]
        self.entity = np.concatenate(entity, axis=0)
        self.time = np.concatenate(time, axis=0)
        self.active_inputs = np.ones_like(outputs)
        # print(self.inputs.shape) # (14672, 180, 10)
        # print(self.outputs.shape) # (14672, 5, 1) = (917*16, 5, 1)
        # print(self.outputs)
     

        self.sampled_data = {
            'inputs': self.inputs,
            'outputs': self.outputs[:, self.encoder_steps:, :],
            'active_entries': np.ones_like(self.outputs[:, self.encoder_steps:, :]),
            'time': self.time,
            'identifier': self.entity
        }
        
    def __getitem__(self, index):
        s = {
        'inputs': self.inputs[index],
        'outputs': self.outputs[index], 
        'active_entries': np.ones_like(self.outputs[index]), 
        'time': self.time[index],
        'identifier': self.entity[index]
        }

        return s


    def __len__(self):
        return self.inputs.shape[0]