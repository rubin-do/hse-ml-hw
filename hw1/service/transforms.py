import pickle
import pandas as pd
import numpy as np
import re


with open('../pickles/medians.pkl', 'rb') as f:
    inferred_medians = pickle.load(f)


def transform_mileage(mileage):
    if isinstance(mileage, float):
        return mileage
    mile_raw = mileage.replace(" ", "")
    if mile_raw[-4:] == "kmpl":
        return float(mile_raw[:-4])
    elif mile_raw[-5:] == "km/kg":
        return float(mile_raw[:-5])
    return float(mile_raw)


def transform_power(power):
    if isinstance(power, float):
        return power
    power_raw = power.replace(" ", "")
    if power_raw[-3:] == "bhp":
        if power_raw[:-3] == "":
            return 1.0
        return float(power_raw[:-3])
    return float(power_raw)


def extract_numbers(s):
    if isinstance(s, float):
        return [s, float('NaN')]
    
    mult = 1
    if 'kgm' in s.lower():
        mult = 10
        
    l = list(map(
        lambda x: float(x.replace(',', '')),
        re.findall('([0-9\.\,]+)', s)))
    l[0] *= mult
    return l


def one_hot_encoding(df):
    df_new = df.copy()
    cols = list(df.select_dtypes(include='object')) + ['seats']
    for col in cols:
        df_new.drop(columns=[col], inplace=True)
        df_new = pd.concat([df_new, pd.get_dummies(df[col], drop_first=True).astype(float)], axis=1)
    
    df_new.columns = df_new.columns.astype(str)
    return df_new


def transform(df, required_features):
    df.drop(columns=['selling_price', 'name'], inplace=True)
    # transform mileage
    df['mileage'] = df['mileage'].apply(transform_mileage)
    
    # transform engine
    df['engine'] = df['engine'].apply(lambda x: x if isinstance(x, float) else float("".join([c for c in str(x) if c.isnumeric()])))
    
    # transform power
    df['max_power'] = df['max_power'].apply(transform_power)

    df['max_torque_rpm'] = df['torque'].apply(lambda x: extract_numbers(x)[-1])
    df['torque'] = df['torque'].apply(lambda x: extract_numbers(x)[0])

    for col in df.select_dtypes(include='number'):
        df[col] = df[col].fillna(inferred_medians[col])
    
    df['engine'] = df['engine'].apply(int)
    df['seats'] = df['seats'].apply(int)

    df_new = one_hot_encoding(df)

    for col in required_features:
        if col not in df_new.columns:
            df_new[col] = np.zeros(df_new.shape[0])
    
    df_new = df_new[required_features]

    return df_new
