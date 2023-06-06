from tft_model import  QuantileLoss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pdb


def eval(test_dataloader,model):
    plots_save_path = "./results"

    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)
    
    out_df = pd.DataFrame(columns=['p10', 'p50', 'p90', 'identifier'])

    start_id = -1
    QUANTILES= [0.1, 0.5, 0.9]
    encoder_steps=400
    
    for i, batch in enumerate(test_dataloader):

        outputs, attention_weights = model(batch)
        bs = batch["outputs"].shape[0]

        process_map = {f"p{int(q * 100)}": outputs.reshape(bs, encoder_steps, 3)[:, :, i].cpu().detach().numpy()[::5,:].reshape(-1)[:bs] for i, q in enumerate(QUANTILES)}

        # Change time ticks
        tmp = pd.DataFrame(data=process_map, index=pd.to_datetime(batch['time'][:, encoder_steps - 1, 0]))
        # pdb.set_trace()

        # tmp = pd.DataFrame(data=process_map, index=batch['time'][:,encoder_steps - 1,0])
        tmp["labels"] = batch["outputs"].reshape(-1)[::encoder_steps]
        tmp["identifier"] = batch['identifier'][:, 0, 0]

        out_df = pd.concat([out_df, tmp])
        
        e = int(batch['identifier'][0, 0, 0].numpy())
        if batch['identifier'][0, 0, 0].numpy() != start_id:
            
            print("="*20)
            print(f"Plotting interpreation plots for a batch of entity {e}:")
            
            # pdb.set_trace()
            id_mask = batch['identifier'][:, 0, 0]==e
            
            #Plotting multi-head attention
            plt.figure(figsize=(15,10))
            sns.lineplot(x=pd.to_datetime(batch["time"][0,:,0].numpy()),
                        y=batch["inputs"][0,:,0].numpy(), color="blue")
            ax2 = plt.twinx()

            sns.lineplot(x=pd.to_datetime(batch["time"][0,:,0].numpy()),
                        y=attention_weights['multihead_attention'][0].cpu().detach().numpy()[:, :].mean(axis=1),
                        ax=ax2, color="orange")
            
            # plt.xlabel("Time")
            plt.savefig(os.path.join(plots_save_path,str(e)+".png"))
            # plt.show()
            
            
            # 可解释性描述权重的可视化代码
            past_inputs = ["WorkLoad","HR","HRR","RER","VE","VT","BF"]

            # Plotting past weights
            plt.figure(figsize=(15,4))
            sns.barplot(x=past_inputs, 
                        y=attention_weights['past_weights'][id_mask,:,:].mean(dim=(0,1)).cpu().detach().numpy(), 
                        palette="crest")
            plt.savefig(os.path.join(plots_save_path,str(e)+"_weights.png"))
            # plt.show()
     
            plt.figure(figsize=(20,10))
            # pdb.set_trace()
            index = out_df[out_df["identifier"]==e].index
            index = np.array([int(str(x)[-3:]) for x in index])
            index = [i-index[0] for i in index]
            sns.lineplot(x=index, y=out_df[out_df["identifier"]==e].labels, color="black",label="ground truth")
            sns.lineplot(x=index, y=out_df[out_df["identifier"]==e].p10, color="red",label="model 1")
            sns.lineplot(x=index, y=out_df[out_df["identifier"]==e].p50, color="orange",label="model 2")
            sns.lineplot(x=index, y=out_df[out_df["identifier"]==e].p90, color="green",label="model 3")
            # plt.legend(labels=["ground truth","model1","model2","model3"])


            x_locations = index
            x_labels = index
            plt.xticks(x_locations,x_labels)

            plt.legend()


            plt.fill_between(index, out_df[out_df["identifier"]==e].p10, out_df[out_df["identifier"]==e].p90, alpha=0.3)
            plt.xlabel("Time (Seconds)")
            
            plt.savefig(os.path.join("./results",str(e)+"_line.png"))
            plt.show()

            start_id = e