from tft_model import  QuantileLoss

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pdb


def eval(test_dataloader,model):
    out_df = pd.DataFrame(columns=['p10', 'p50', 'p90', 'identifier'])

    start_id = -1
    QUANTILES= [0.1, 0.5, 0.9]
    
    for i, batch in enumerate(test_dataloader):

        outputs, attention_weights = model(batch)
        bs = batch["outputs"].shape[0]

        process_map = {f"p{int(q * 100)}": outputs.reshape(bs, 175, 3)[:, :, i].cpu().detach().numpy()[::5,:].reshape(-1)[:bs] for i, q in enumerate(QUANTILES)}

        tmp = pd.DataFrame(data=process_map, index=pd.to_datetime(batch['time'][:, 175 - 1, 0]))
        tmp["labels"] = batch["outputs"].reshape(-1)[::175]
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
            # pdb.set_trace()
            # 没有问题，因为y值为nan，所以画不出来
            sns.lineplot(x=pd.to_datetime(batch["time"][0,:,0].numpy()),
                        y=attention_weights['multihead_attention'][0].cpu().detach().numpy()[:, :].mean(axis=1),
                        ax=ax2, color="orange")
            plt.savefig("a.png")
            plt.show()
            
            
            # 可解释性描述权重的可视化代码
            past_inputs = ["Time"]
            # future_inputs = ["day_of_week", "day_of_month", "week_of_year", "month", "days_from_start"]
            

            # Plotting past weights
            # plt.figure(figsize=(15,4))
            # sns.barplot(x=past_inputs, 
            #             y=attention_weights['past_weights'][id_mask,:,:].mean(dim=(0,1)).cpu().detach().numpy(), 
            #             palette="crest")
            # plt.show()

            # Plotting future weights
            # plt.figure(figsize=(15,4))
            # sns.barplot(x=future_inputs, 
            #             y=attention_weights['future_weights'][id_mask,:,:].mean(dim=(0,1)).cpu().detach().numpy(), 
            #             palette="crest")
            # plt.show()
            start_id = e
     
            
        for i in range(5):
    plt.figure(figsize=(20,10))
    sns.lineplot(x=test[test["Entity"]==i].iloc[:ENCODER_STEPS-1]["date"], y=test[test["Entity"]==i].iloc[:ENCODER_STEPS-1]["traffic"])
    sns.lineplot(x=out_df[out_df["identifier"]==i].index, y=out_df[out_df["identifier"]==i].labels, color="black")
    sns.lineplot(x=out_df[out_df["identifier"]==i].index, y=out_df[out_df["identifier"]==i].p10, color="red")
    sns.lineplot(x=out_df[out_df["identifier"]==i].index, y=out_df[out_df["identifier"]==i].p50, color="orange")
    sns.lineplot(x=out_df[out_df["identifier"]==i].index, y=out_df[out_df["identifier"]==i].p90, color="green")

    plt.fill_between(out_df[out_df["identifier"]==i].index, out_df[out_df["identifier"]==i].p10, out_df[out_df["identifier"]==i].p90, alpha=0.3)

    plt.show()