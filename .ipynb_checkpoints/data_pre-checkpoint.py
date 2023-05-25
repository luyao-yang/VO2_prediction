import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pdb

def fit_preprocessing(train, real_columns, categorical_columns):
    real_scalers = StandardScaler().fit(train[real_columns].values)

    categorical_scalers = {}
    num_classes = []
    for col in categorical_columns:
        srs = train[col].apply(str) 
        categorical_scalers[col] = LabelEncoder().fit(srs.values)
        num_classes.append(srs.nunique())

    return real_scalers, categorical_scalers


def transform_inputs(df, real_scalers, categorical_scalers, real_columns, categorical_columns):
    out = df.copy()
    out[real_columns] = real_scalers.transform(df[real_columns].values)

    ##TODO 想明白这一步是干嘛，是否需要这一步 
    # initiate the workload
    for col in categorical_columns:
        string_df = df[col].apply(str)
        out[col] = categorical_scalers[col].transform(string_df)

    return out

# 平均2s采样samples，处理最原始的数据
def data_preprocess(folder_path, data_save_path):
    # Set the total dataframe
    CPET_files = [] 

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        # Read the time series from an Excel file
        df = pd.read_excel(file_path, parse_dates=['Time'])

        # Set the index of the DataFrame to the time column
        df.set_index('Time', inplace=True)

        ## Resample the time series to a frequency of 2 seconds 
        # using the interploting method "linear"
        df_resampled = df.resample('2S').mean().interpolate(method="linear")
        ## using the interploting method "spline" can be computationally expensive \\
        ## and may require more tuning of the interpolation parameters
        # df_resampled = df.resample('2S').mean().interpolate(method="spline", order=3)
        ## using the interploting method "polynomial"
        # df_resampled = df.resample('2S').mean().interpolate(method="polynomial", order=2)

        ##change the workLoad to categorical variants
        df_resampled.loc[df_resampled['WorkLoad'] <= 10.0, 'WorkLoad'] = 0.0
        df_resampled.loc[(df_resampled['WorkLoad'] <= 30.0) & (df_resampled['WorkLoad'] > 10.0), 'WorkLoad'] = 20.0
        df_resampled.loc[(df_resampled['WorkLoad'] <= 50.0) & (df_resampled['WorkLoad'] > 30.0), 'WorkLoad'] = 40.0
        df_resampled.loc[(df_resampled['WorkLoad'] <= 70.0) & (df_resampled['WorkLoad'] > 50.0), 'WorkLoad'] = 60.0
        df_resampled.loc[(df_resampled['WorkLoad'] <= 90.0) & (df_resampled['WorkLoad'] > 70.0), 'WorkLoad'] = 80.0


        df_resampled.index = (df_resampled.index - df_resampled.index.min()).total_seconds()
        save_file_path = os.path.join(data_save_path,file+".csv")
        df_resampled.to_csv(save_file_path)

        CPET_files.append(df_resampled)

    final_CPET_files = pd.concat(CPET_files)
    save_file_path = os.path.join(data_save_path, "CPET_files.csv")
    final_CPET_files.to_csv(save_file_path)

def main():

    # Set the folder path
    folder_path = "./CPET-2-proc"
    data_save_path = "./data"

    # 2s抽取一定的数据
    # data_preprocess(folder_path, data_save_path)

    CPET_files = pd.read_csv(os.path.join(data_save_path,"CPET_files.csv"))
    CPET_files = CPET_files.drop(columns=['VO2'])

    # train_ids, test_ids = train_test_split(CPET_files['ID'].unique(), test_size=0.2, random_state=42)

    # Create the train and test DataFrames based on the split IDs
    # train_df = CPET_files[CPET_files['ID'].isin(train_ids)]
    # test_df = CPET_files[CPET_files['ID'].isin(test_ids)]

    real_columns = ['HR','HRR','RER','VE','VT','BF','VO2/kg']
    categorical_columns = ['ID','WorkLoad','Age','Height','Weight','BMI','BSA','Temperature','Humidity']

    # train_df.to_csv("./train_o.csv",index=None)
    # test_df.to_csv("./test_0.csv",index=None)

    # TODO ??
    real_scalers, categorical_scalers = fit_preprocessing(CPET_files, real_columns, categorical_columns)
    # train_df = transform_inputs(train_df, real_scalers, categorical_scalers, reanl_columns, categorical_columns)
    # real_scalers, categorical_scalers = fit_preprocessing(test_df, real_columns, categorical_columns)
    all_df = transform_inputs(CPET_files, real_scalers, categorical_scalers, real_columns, categorical_columns)

    pdb.set_trace()
    train_ids, test_ids = train_test_split(all_df['ID'].unique(), test_size=0.2, random_state=42)
    train_df = all_df[all_df['ID'].isin(train_ids)]
    test_df = all_df[all_df['ID'].isin(test_ids)]

    train_df.to_csv("./train.csv",index=None)
    test_df.to_csv("./test.csv",index=None)


if __name__=="__main__":
    main()





