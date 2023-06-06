import os
import pandas as pd
import numpy as np

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
        # df = pd.read_excel(file_path, parse_dates=['Time'])
        df = pd.read_excel(file_path,header=0)
        # df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df['Time'] = pd.to_datetime(df['Time'], unit='s')

        # Set the index of the DataFrame to the time column
        
        df.set_index('Time', inplace=True)

        
        df_resampled = df.resample('2S').mean().interpolate(method="linear")
        ## using the interploting method "spline" can be computationally expensive \\
        ## and may require more tuning of the interpolation parameters
        # df_resampled = df.resample('2S').mean().interpolate(method="spline", order=3)
        ## using the interploting method "polynomial"
        # df_resampled = df.resample('2S').mean().interpolate(method="polynomial", order=2)

        ##change the workLoad to categorical variants
        # pdb.set_trace()
        # bins = pd.IntervalIndex.from_tuples([(i, i + 20) for i in range(-10, 330, 20)])
        bins = [-1, 10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330]
        df_resampled['WorkLoad'] = pd.cut(df_resampled['WorkLoad'], bins=bins, labels=np.arange(0,340,20))


        df_resampled.index = (df_resampled.index - df_resampled.index.min()).total_seconds()
        save_file_path = os.path.join(data_save_path,file+".csv")
        df_resampled.to_csv(save_file_path)

        CPET_files.append(df_resampled)

    final_CPET_files = pd.concat(CPET_files)
    save_file_path = os.path.join(data_save_path, "CPET_files.csv")
    final_CPET_files.to_csv(save_file_path)

def main():

    # Set the folder path 24 files
    # folder_path = "./CPET-2-proc"
    # data_save_path = "./data_2"

    folder_path = "./CPET-1-proc/Health"
    data_save_path = "./data_2"

    # 2s抽取一定的数据
    # data_preprocess(folder_path, data_save_path)

    # pdb.set_trace()
    CPET_files_2 = pd.read_csv(os.path.join("./data_2","CPET_files.csv"))
    CPET_files_2 = CPET_files_2.drop(columns=['VO2/kg','BSA','Temperature','Humidity'])
    CPET_files_3 = pd.read_csv(os.path.join("./data_3","CPET_files.csv"))
    CPET_files = pd.concat([CPET_files_2, CPET_files_3], ignore_index=True)

    # 去除掉BSA, Temperature, humidity, VO2/kg 这几个参数
    # CPET_files = CPET_files.drop(columns=['VO2/kg','BSA','Temperature','Humidity'])

    # train_ids, test_ids = train_test_split(CPET_files['ID'].unique(), test_size=0.2, random_state=42)

    # Create the train and test DataFrames based on the split IDs
    # train_df = CPET_files[CPET_files['ID'].isin(train_ids)]
    # test_df = CPET_files[CPET_files['ID'].isin(test_ids)]

    real_columns = ['HR','HRR','RER','VE','VT','BF','VO2']
    # categorical_columns = ['ID','WorkLoad','Age','Height','Weight','BMI','BSA','Temperature','Humidity']
    categorical_columns = ['ID','WorkLoad','Age','Height','Weight','BMI']

    # train_df.to_csv("./train_o.csv",index=None)
    # test_df.to_csv("./test_0.csv",index=None)

    # TODO ?? 解释为什么这么做是对的
    real_scalers, categorical_scalers = fit_preprocessing(CPET_files, real_columns, categorical_columns)
    # train_df = transform_inputs(train_df, real_scalers, categorical_scalers, reanl_columns, categorical_columns)
    # real_scalers, categorical_scalers = fit_preprocessing(test_df, real_columns, categorical_columns)
    all_df = transform_inputs(CPET_files, real_scalers, categorical_scalers, real_columns, categorical_columns)

    # pdb.set_trace()
    train_ids, test_ids = train_test_split(all_df['ID'].unique(), test_size=0.2, random_state=42)
    train_df = all_df[all_df['ID'].isin(train_ids)]
    test_df = all_df[all_df['ID'].isin(test_ids)]

    train_df.to_csv("./train1.csv",index=None)
    test_df.to_csv("./test1.csv",index=None)


if __name__=="__main__":
    main()





