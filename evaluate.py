import pandas as pd


out_df = pd.DataFrame(columns=['p10', 'p50', 'p90', 'identifier'])

start_id = -1

for i, batch in enumerate(test_dataloader):

    outputs, attention_weights = model(batch)
    bs = batch["outputs"].shape[0]

    process_map = {f"p{int(q * 100)}": outputs.reshape(bs, 5, 3)[:, :, i].cpu().detach().numpy()[::5,:].reshape(-1)[:bs] for i, q in enumerate(QUANTILES)}

    tmp = pd.DataFrame(data=process_map, index=pd.to_datetime(batch['time'][:, ENCODER_STEPS - 1, 0]))
    tmp["labels"] = batch["outputs"].reshape(-1)[::5]
    tmp["identifier"] = batch['identifier'][:, 0, 0]

    out_df = pd.concat([out_df, tmp])
    
    e = int(batch['identifier'][0, 0, 0].numpy())
    if batch['identifier'][0, 0, 0].numpy() != start_id:
        
        print("="*20)
        print(f"Plotting interpreation plots for a batch of entity {e}:")
        
        id_mask = batch['identifier'][:, 0, 0]==e
        
        #Plotting multi-head attention
        plt.figure(figsize=(15,10))
        sns.lineplot(x=pd.to_datetime(batch["time"][0,:,0].numpy()),
                     y=batch["inputs"][0,:,0].numpy(), color="blue")
        ax2 = plt.twinx()
        sns.lineplot(x=pd.to_datetime(batch["time"][0,:,0].numpy()),
                     y=attention_weights['multihead_attention'][0].cpu().detach().numpy()[:, 175:].mean(axis=1),
                     ax=ax2, color="orange")
        plt.show()
        
        past_inputs = ["day_of_week", "day_of_month", "week_of_year", "month", 'log_vol', 'days_from_start', "open_to_close"]
        future_inputs = ["day_of_week", "day_of_month", "week_of_year", "month", "days_from_start"]
        
        # Plotting past weights
        plt.figure(figsize=(15,4))
        sns.barplot(x=past_inputs, 
                    y=attention_weights['past_weights'][id_mask,:,:].mean(dim=(0,1)).cpu().detach().numpy(), 
                    palette="crest")
        plt.show()

        # Plotting future weights
        plt.figure(figsize=(15,4))
        sns.barplot(x=future_inputs, 
                    y=attention_weights['future_weights'][id_mask,:,:].mean(dim=(0,1)).cpu().detach().numpy(), 
                    palette="crest")
        plt.show()
        start_id = e
        print()