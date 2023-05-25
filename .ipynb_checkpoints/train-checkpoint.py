from tft_model import  QuantileLoss
from tft_model import TemporalFusionTransformer
from TFT_dataset import TFT_Dataset

import time
import json
import torch
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim

import pdb


def read_data(params):
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    # Dataset variables
    input_columns = ["ID", "Age", "Height", "Weight", "BMI", "BSA", "Temperature",
                     "Humidity","WorkLoad","HR","HRR","RER","VE","VT","BF","VO2/kg"]
    target_column = "VO2/kg"
    id_column = "ID"
    time_column = "Time"
    col_to_idx = {col: idx for idx, col in enumerate(input_columns)}

    encoder_steps = params['encoder_steps']
    decoder_steps = params['decoder_steps']
    training_data = TFT_Dataset(train_df, id_column, time_column, target_column, input_columns,encoder_steps, decoder_steps)
    testing_data = TFT_Dataset(test_df, id_column, time_column, target_column, input_columns, encoder_steps, decoder_steps)

    return training_data, testing_data, col_to_idx


def train(params,train_dataloader,device):
    
    model = TemporalFusionTransformer(params,device)
    model.to(device)
    criterion = QuantileLoss(params["quantiles"])
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    print_every_k = 100
    losses = []

    for epoch in range(params["epochs"]):
        t0 = time.time()
        print(f"===== Epoch {epoch+1} =========")
        epoch_loss = 0.0
        running_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            # print(batch.keys())
            # batch: dict_keys(['inputs', 'outputs', 'active_entries', 'time', 'identifier'])
            labels = batch['outputs'][:,:,0].flatten().float().to(device)
            # print(batch['outputs'].shape) # [64,180,1]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # pdb.set_trace()
            # inputs.shape=[64,180,15]
            outputs, attention_weights = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()

            if (i+1) % print_every_k == 0:
                # print(f"Mini-batch {i+1} average loss: {round(running_loss / print_every_k, 5)}")
                running_loss = 0.0
        
        t1 = time.time()
        # print(f"\nEpoch trained for {round(t1-t0, 2)} seconds")
        # print("\nEpoch loss:", round(epoch_loss / (i+1), 5), "\n")
        losses.append(epoch_loss / (i+1))


def main():
    with open('config.json','r') as f:
        params = json.load(f)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_data, testing_data, col_to_idx = read_data(params)

    params['col_to_idx'] = col_to_idx
    batch_size = params['batch_size']

    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=2, shuffle=False)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, num_workers=2, shuffle=False)
    
    train(params,train_dataloader,DEVICE)

if __name__ == '__main__':
        main()
        