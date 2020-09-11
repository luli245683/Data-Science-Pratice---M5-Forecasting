# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:36:05 2020

@author: luli
"""

import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime as dt
######################################################################################
#  File parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
parser.add_argument('--data_path', type=str, default='../data/', help='The path of input files.')
parser.add_argument('--output_path', type=str, default='../dump/', help='The path of input files.')
parser.add_argument('--seed', type=int, default=2020, help='')
parser.add_argument('--firstDay', type=int, default=1200, help='')
parser.add_argument('--lastDay', type=int, default=1913, help='')
parser.add_argument('--max_lags', type=int, default=57, help='')
######################################################################################
#  Model parameters
######################################################################################
args = parser.parse_args()
np.random.seed(2020)
cat_cols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

calendarDTypes = {"event_name_1": "category", 
                      "event_name_2": "category", 
                      "event_type_1": "category", 
                      "event_type_2": "category", 
                      "weekday": "category", 
                      'wm_yr_wk': 'int16', 
                      "wday": "int16",
                      "month": "int16", 
                      "year": "int16", 
                      "snap_CA": "float32", 
                      'snap_TX': 'float32', 
                      'snap_WI': 'float32' }

priceDTypes = {"store_id": "category", 
               "item_id": "category", 
               "wm_yr_wk": "int16",
               "sell_price":"float32"}

params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        "metric": ["rmse"],
        'verbosity': 1,
        'num_iterations' : 1200,
        'num_leaves': 2**11-1,
        "min_data_in_leaf":  2**12-1,
}


######################################################################################
# The End of Hyperparameters
######################################################################################
def prepare_dataset(is_train = True):
    
    start_day = max(1 if is_train else args.lastDay-args.max_lags, args.firstDay)
    num_cols = [f"d_{day}" for day in range(start_day, args.lastDay+1)]
    
    calendar = label_encoding(pd.read_csv(args.data_path + 'calendar.csv', encoding='utf-8', dtype=calendarDTypes),calendarDTypes)
    calendar["date"] = pd.to_datetime(calendar["date"])
    print('calendar has %d rows and %d columns' % (calendar.shape[0], calendar.shape[1]))
    
    

    sell_prices = label_encoding(pd.read_csv(args.data_path + 'sell_prices.csv', encoding='utf-8', dtype=priceDTypes),priceDTypes)
    print('sell_prices has %d rows and %d columns' % (sell_prices.shape[0], sell_prices.shape[1]))
    
    
    
    salesDTypes = {numCol: 'float32' for numCol in num_cols}
    salesDTypes.update({catCol: 'category' for catCol in cat_cols if catCol != 'id'})
    
    
    sales_train_validation = label_encoding(pd.read_csv(args.data_path + "sales_train_validation.csv", encoding='utf-8', \
                                                         usecols = cat_cols + num_cols, dtype = salesDTypes),salesDTypes)
    print('sales_train_validation has %d rows and %d columns' % (sales_train_validation.shape[0], sales_train_validation.shape[1]))

    
    if not is_train:
        for day in range(args.lastDay+1,args.lastDay+29):
            sales_train_validation[f'd_{day}'] = np.nan


    return merge_df(sales_train_validation , sell_prices , calendar)
    

# Label encoding
def label_encoding(df, col_type_dict):
    
    for col, col_dtype in col_type_dict.items():
        if col_dtype == 'category':
            df[col] = df[col].cat.codes.astype('int16')
            df[col] -= df[col].min()
    
    return df

def merge_df(sales_train_validation, sell_prices , calendar):
    

    df = pd.melt(sales_train_validation, id_vars=cat_cols, 
                 value_vars = [col for col in sales_train_validation.columns if col.startswith("d_")], var_name='d', value_name='sold')
    df = pd.merge(df, calendar, on='d', copy = False)
    df = pd.merge(df, sell_prices, on=['store_id','item_id','wm_yr_wk'], copy = False)
    
    
    return df
    
def get_features(df):
    lags = [7,28]
    lag_cols = [f'lag_{lag}' for lag in lags]
    windows = [7, 28]
    
    #lag features
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[["id","sold"]].groupby("id")["sold"].shift(lag)
        
    #Window Features
    for window in windows:
        for lag, lag_col in zip(lags, lag_cols):
            df[f"rmean_{lag}_{window}"] = df[["id", lag_col]].groupby("id")[lag_col].transform(lambda x: x.rolling(window).mean())
            df[f"rstd_{lag}_{window}"] = df[["id", lag_col]].groupby("id")[lag_col].transform(lambda x: x.rolling(window).std())
            
    dateFeatures = {"wday": "weekday",
                    "week": "weekofyear",
                    "month": "month",
                    "quarter": "quarter",
                    "year": "year",
                    "mday": "day",
                    "dayofweek":"dayofweek"}

    for featName, featFunc in dateFeatures.items():
        if featName in df.columns:
            df[featName] = df[featName].astype("int16")
        else:
            df[featName] = getattr(df["date"].dt, featFunc).astype("int16")
    


def get_training_data(df):
    
    catfeats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id', 'event_name_1', 'event_name_2' , 'event_type_1', 'event_type_2']
    none_use_cols = ['id' , 'date' , 'sold' , 'd' , 'wm_yr_wk' , 'weekday']
    train_cols = df.columns[~df.columns.isin(none_use_cols)]
    X_train = df[train_cols]
    y_train = df["sold"]
    # print(X_train.info())
    valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)
    train_inds = np.setdiff1d(X_train.index.values, valid_inds)
    
    train_data = lgb.Dataset(X_train.loc[train_inds], label = y_train.loc[train_inds], 
                            categorical_feature = catfeats, free_raw_data = False)
    valid_data = lgb.Dataset(X_train.loc[valid_inds], label = y_train.loc[valid_inds],
                            categorical_feature = catfeats, free_raw_data = False)
    
    return train_data , valid_data , train_cols

'''
faster way to create lag features and date features
''' 
def create_lag_features_for_test(df, day):
    print("start to create lag features for testing data:", df.shape)
    # create lag feaures just for single day (faster)
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):

        df.loc[df.date == day, lag_col] = df.loc[df.date ==day-dt.timedelta(days=lag), 'sold'].values  
    # print(df.shape)
    windows = [7, 28]
    for window in windows:
        for lag in lags:
            df_window = df[(df.date <= day-dt.timedelta(days=lag)) & (df.date > day-dt.timedelta(days=lag+window))]
            df_window_grouped = df_window.groupby("id").agg({'sold':'mean'}).reindex(df.loc[df.date==day,'id'])
            df.loc[df.date == day,f"rmean_{lag}_{window}"] = df_window_grouped.sold.values 
    # print(df.shape)
    
def create_date_features_for_test(df):
    # copy of the code from `create_dt()` above
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
        "dayofweek":"dayofweek"
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in df.columns:
            df[date_feat_name] = df[date_feat_name].astype("int16")
        else:
            df[date_feat_name] = getattr(
                df["date"].dt, date_feat_func).astype("int16")
def predict(model, test_df,train_cols):
    print(train_cols)
    create_date_features_for_test(test_df)
    
    cols = [f'F{i}' for i in range(1,29)]
    first_date = dt.datetime(2016,4,25)
    
    
    
    for delta in range(0,28):
        day = first_date + dt.timedelta(days=delta)
        print(delta, day)
        tst = test_df[(test_df.date >= day - dt.timedelta(days=args.max_lags)) & (test_df.date <=day)].copy()
        
        
        # get_features(tst)
        create_lag_features_for_test(tst,day)
        
        tst = tst.loc[tst.date == day , train_cols]
        # print(tst.info())
        
        test_df.loc[test_df.date == day, "sold"] = model.predict(tst)
        
    submission = test_df.loc[test_df.date>=first_date, ["id", "sold"]].copy()
    submission["F"] = [f"F{rank}" for rank in submission.groupby("id")["id"].cumcount()+1]
    submission = submission.set_index(["id", "F" ]).unstack()["sold"][cols].reset_index()
    submission.fillna(0., inplace = True)
    submission.sort_values("id", inplace = True)
    submission.reset_index(drop=True, inplace = True)
    # submission.to_csv(args.output_path+"submission.csv",index=False)
        
    submission2 = submission.copy()
    submission2["id"] = submission2["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([submission, submission2], axis=0, sort=False)
    sub.to_csv(args.output_path+"submission.csv",index=False)
    return submission

if __name__ == '__main__':
    
    
    df = prepare_dataset(is_train=True)
    # print(df.shape)

    get_features(df)
    # df.dropna(inplace=True)
    # print(df.info(),df.shape)
    train_data , valid_data , train_cols = get_training_data(df)
    
    # m_lgb =lgb.train(params, train_data, valid_sets = [valid_data], verbose_eval=20) 
    
    # print("save trained file.")
    # m_lgb.save_model(args.output_path + "model.lgb")
    
    m_lgb = lgb.Booster(model_file=args.output_path + "model.lgb")
    test_df = prepare_dataset(is_train=False)
    print(test_df.shape)
    submission = predict(m_lgb, test_df,train_cols)