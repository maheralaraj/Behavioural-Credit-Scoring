import pandas as pd
import numpy as np
import glob
import os
import pickle


def additional_info():
    client_df = pd.read_csv('other_data/client_info.csv', low_memory=False)
    country_df = pd.read_csv('other_data/country_list.csv', low_memory=False)
    salary_df = pd.read_csv('other_data/salary_new.csv', thousands=r' ', dtype={'Salary.New':float}, low_memory=False)
    client_df = client_df.loc[:, ['Key', 'Branch ID', 'Open date MM/YY', 'Birth_Year', 'Nationality']]
    client_df = client_df.merge(salary_df, on=['Key'], how='left')
    df = client_df.merge(country_df, left_on=['Nationality'], right_on=['alpha-2'], how='left').drop(['Nationality', 'alpha-2'], 1)
    BI_dummies = pd.get_dummies(df['Branch ID'])
    df = pd.concat([df, BI_dummies], axis=1)
    region_dummies = pd.get_dummies(df['region'])
    df = pd.concat([df, region_dummies], axis=1)
    df = df.drop(['Branch ID', 'region'], 1)
    df.insert(2, 'Is Equal', np.where(df['Open date MM/YY'] == '1/1/1990', True, False))
    start = pd.to_datetime('1/1/1990')
    df['Open date MM/YY'] = pd.to_datetime(df['Open date MM/YY'])
    df['Open date MM/YY'] = (df['Open date MM/YY']-start)/np.timedelta64(1, 'Y')
    df['Open date MM/YY'].mask(df['Is Equal'], other=np.nan, inplace=True)
    df.insert(5, 'Is salary empty', pd.isnull(df['Salary.New']).astype(int))
    for i in ['Salary.New','Birth_Year','Open date MM/YY']:
        df[i] = (df[i]-df[i].mean())/df[i].std()
    return df


path = r'transactions'
merchant_code_file = 'other_data/merchant_codes.csv'
merchant_code = pd.read_csv(merchant_code_file, low_memory=False)
merchant_code = merchant_code.loc[:, ['MERCHANT.CAT.CODE', 'MCG CODE']]
merchant_code = merchant_code.groupby('MERCHANT.CAT.CODE').agg(np.min)

ALLOWED_TRANSACTION_CODES = [1, 2, 101, 102, 110, 130, 135, 136, 140, 605, 620, 10]
all_files = glob.glob(os.path.join(path, "*.csv"))
for i in all_files:
    print(i)
df_from_each_file = (pd.read_csv(f, thousands=r' ', dtype={'TRANS AMT IN AED-Signed':float,'TRANSACTION POST DATE': str, 'TRANSACTION DATE': str}, low_memory=False) for f in all_files)
df = pd.concat(df_from_each_file, ignore_index=True)
df.Key = pd.to_numeric(df.Key, errors='coerce')
df = df[df.Key.notna()]
df = df.loc[:, ['Key', 'TRANSACTION DATE', 'TRANSACTION CODE', 'TRANSACTION POST DATE', 'MERCHANT CAT CODE', 'TRANS AMT IN AED-Signed']]
df = df.loc[df['TRANSACTION CODE'].isin(ALLOWED_TRANSACTION_CODES)]
df.loc[df['TRANSACTION POST DATE'] == '0', 'TRANSACTION POST DATE'] = None
mask = pd.isnull(df['TRANSACTION POST DATE'])
tmp = df.loc[mask, 'TRANSACTION DATE']
df.loc[mask, 'TRANSACTION POST DATE'] = tmp
df = df.drop(df[df['TRANSACTION DATE'] == '0'].index)
df['TRANSACTION DATE'] = pd.to_datetime(df['TRANSACTION DATE'].str.rjust(8, '0'), format='%d%m%Y')
df['TRANSACTION POST DATE'] = pd.to_datetime(df['TRANSACTION POST DATE'].str.rjust(8, '0'), format='%d%m%Y')
df = df.sort_values(['TRANSACTION DATE'], ascending=True)
res = df[['Key', 'TRANS AMT IN AED-Signed', 'TRANSACTION CODE', 'TRANSACTION DATE', 'MERCHANT CAT CODE']]
res.insert(2, "DAY OF THE WEEK", df['TRANSACTION DATE'].dt.dayofweek + 1)
res.insert(3, "DAY OF THE MONTH", df['TRANSACTION DATE'].dt.day)
res.insert(4, "MONTH", df['TRANSACTION DATE'].dt.month)
res.insert(5, "YEAR", df['TRANSACTION DATE'].dt.year)
res.insert(6, "DAYS TILL POST DATE", (df['TRANSACTION POST DATE'] - df['TRANSACTION DATE']).dt.days)
res.insert(7, "IS POSTDATE NULL", mask.astype(int))
res = res.merge(merchant_code, left_on=['MERCHANT CAT CODE'], right_on=merchant_code.index, how='left').drop(
    ['MERCHANT CAT CODE'], 1)
print(res.shape)

with open("pickle/df_cleaned.pickle", "wb") as fp:
    pickle.dump(res, fp)

with open("pickle/df_additional.pickle", "wb") as fp:
    pickle.dump(additional_info(), fp)



