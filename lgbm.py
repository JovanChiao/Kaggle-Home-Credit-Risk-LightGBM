from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

warnings.simplefilter(action='ignore')

# Determine the data format of columns ending in “L” and “T”
str_L = ['credtype_587L', 'familystate_726L', 'inittransactioncode_279L', 'status_219L', 'conts_type_509L',
         'familystate_447L', 'gender_992L', 'housetype_905L', 'housingtype_772L', 'maritalst_703L',
         'role_1084L', 'role_993L', 'sex_738L', 'type_25L', 'addres_role_871L', 'bankacctype_710L', 'cardtype_51L',
         'credtype_322L', 'disbursementtype_67L', 'inittransactioncode_186L', 'lastst_736L', 'paytype1st_925L',
         'paytype_783L', 'twobodfilling_608L', 'typesuite_864L', 'empl_industry_691L', 'requesttype_4525192L',
         'equalityempfrom_62L', 'credacc_cards_status_52L', 'credacc_status_367L', 'empl_employedtotal_800L']
bool_L = ['isbidproduct_390L', 'contaddr_matchlist_1032L', 'contaddr_smempladdr_334L', 'isreference_387L',
          'remitter_829L', 'safeguarantyflag_411L', 'equalitydataagreement_891L', 'isbidproduct_1095L',
          'isbidproductrequest_292L', 'isdebitcard_729L', 'opencred_647L', 'isdebitcard_527L']
str_T = ['incometype_1044T', 'relationshiptoclient_415T', 'relationshiptoclient_642T', 'relatedpersons_role_762T',
         'riskassesment_302T']

# 1.Data preprocessing
# Data filtering
def filter_cols(df):

    for col in df.columns:  # remove columns with >95% missing values
        if col not in ['target', 'case_id', 'WEEK_NUM']:
            rate_null = df[col].isnull().mean()
            if rate_null > 0.95:
                df = df.drop(col, axis=1)

    for col in df.columns:  # remove duplicate values
        if (col not in ['target', 'case_id', 'WEEK_NUM']) and (type(df[col][0]) == str):
            freq = df[col].nunique()
            if (freq == 1):
                df = df.drop(col, axis=1)

    for col in df.columns:  # remove non-input variables
        if (col[-1] not in ["P", "A", "L", "M"]) and (('month_' in col) or ('year_' in col)):  # 删除用于聚合的变量
            df = df.drop(col, axis=1)

    return df

# Standardization of data formats
def set_data_type(df):
    for col in df.columns:
        if col in ['case_id', 'WEEK_NUM', 'num_group1', 'num_group2']:
            df[col] = df[col].astype(int)
        elif col in ['data_decision']:
            df[col] = pd.to_datetime(df[col])
        elif (col[-1] in ('P', 'A', 'L', 'T')) and (col not in (str_L+bool_L+str_T)):
            df[col] = df[col].astype(np.float32)
        elif (col[-1] in 'M') or (col in (str_L + str_T)):
            df[col] = df[col].astype(str)
        elif col[-1] in 'D':
            df[col] = pd.to_datetime(df[col])
        elif col in bool_L:
            df[col] = pd.Series(df[col]).apply(lambda x: 1 if x is True else (0 if x is False else np.nan))
    return df

# Converting String Data to Numeric Data with Label Encoding
def str_to_int(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


# 2.Data aggregation
def agg_str(df):  # String, Boolean data: the number of pluralities
    cols = [col for col in df.columns if col.endswith("M") or (col in (str_L+str_T)) or (col in bool_L)]
    if len(cols) == 0:
        return pd.DataFrame([])
    mode = pd.DataFrame([df[col].agg('mode') for col in cols]).iloc[0]
    return pd.DataFrame({col: [mode_col] for col, mode_col in zip(cols, mode)})


def agg_num(df):  # Numerical data: the maximum value, minimum value, sum, mean, median and standard deviation
    cols = [col for col in df.columns if (col[-1] in ('P', 'A', 'L', 'T')) and (col not in (str_L+bool_L+str_T))]
    max = [df[col].agg('max') for col in cols]
    min = [df[col].agg('min') for col in cols]
    sum = [df[col].agg('sum') for col in cols]
    mean = [df[col].agg('mean') for col in cols]
    median = [df[col].agg('median') for col in cols]
    std = [df[col].agg('std') for col in cols]
    row_names = ['max', 'min', 'sum', 'mean', 'median', 'std']
    new_col_names = [f"{r}_{c}" for r in row_names for c in cols]
    df_flattened = pd.DataFrame([max, min, sum, mean, median, std]).values.flatten()

    return pd.DataFrame([df_flattened], columns=new_col_names)


def agg_date(df):  # Date data: the maximum and minimum values, and divided into year, month, day return
    cols = [col for col in df.columns if col.endswith("D")]
    max = [df[col].agg('max') for col in cols]
    min = [df[col].agg('min') for col in cols]
    max_year = [date.year for date in max]
    max_month = [date.month for date in max]
    max_day = [date.day for date in max]
    min_year = [date.year for date in min]
    min_month = [date.month for date in min]
    min_day = [date.day for date in min]

    row_names = ['max_year', 'max_month', 'max_day', 'min_year', 'min_month', 'min_day']
    new_col_names = [f"{r}_{c}" for r in row_names for c in cols]
    df_flattened = pd.DataFrame([max_year, max_month, max_day, min_year, min_month, min_day]).values.flatten()

    return pd.DataFrame([df_flattened], columns=new_col_names)


def agg_date2_1(df):  # Aggregate date data from depth=2 to depth=1
    cols = [col for col in df.columns if col.endswith("D")]
    max = [df[col].agg('max') for col in cols]
    min = [df[col].agg('min') for col in cols]
    df_flattened = pd.DataFrame([max, min]).values.flatten()

    return pd.DataFrame([df_flattened])


# 3.Data grouping
# depth=0, without grouping, the date-type data will be converted to the year, month, day after the return;
def dep0(df):
    cols = [col for col in df.columns if col.endswith("D")]
    for col in cols:
        df[f'year_{col}'] = [date.year for date in list(df[col])]
        df[f'month_{col}'] = [date.month for date in list(df[col])]
        df[f'day_{col}'] = [date.day for date in list(df[col])]
    df.drop(columns=cols, inplace=True)

    return df

# depth=1, according to “case_id” grouping, data aggregation;
def dep1(df, index=None):
    df_id = list({idcase: group.reset_index(drop=True) for idcase, group in df.groupby('case_id')}.values())
    num_agg = []
    strbool_agg = []
    date_agg = []
    id_case = []
    for i in range(len(df_id)):
        num_agg.append(agg_num(df_id[i]))
        strbool_agg.append(agg_str(df_id[i]))
        if index == 1:
            date_agg.append(agg_date(df_id[i]))
        elif index == 2:
            date_agg.append(agg_date2_1(df_id[i]))
        id_case.append(df_id[i]['case_id'].iloc[0])
    num_agg = pd.concat(num_agg, ignore_index=True)
    strbool_agg = pd.concat(strbool_agg, ignore_index=True)
    date_agg = pd.concat(date_agg, ignore_index=True)
    merge_df = pd.concat([num_agg, strbool_agg, date_agg], axis=1)
    merge_df.insert(0, 'case_id', id_case)
    return merge_df


# depth=2, according to “case_id” and “num_group1” for grouping, data aggregation
def dep2(df):
    df_group = []
    id_groups = df.groupby('case_id')
    for id_value, id_group_df in id_groups:
        group_result = []
        group_groups = id_group_df.groupby('num_group1')
        for group_value, group_df in group_groups:
            group_result.append(group_df)
        df_group.append(group_result)
    num_agg = []
    strbool_agg = []
    date_agg = []
    id_case = []
    group1 = []
    for idcase in range(len(df_group)):
        df_id = df_group[idcase]
        for group in range(len(df_id)):
            num_agg.append(agg_num(df_id[group]))
            strbool_agg.append(agg_str(df_id[group]))
            date_agg.append(agg_date(df_id[group]))
            id_case.append(df_id[group]['case_id'].iloc[0])
            group1.append(df_id[group]['num_group1'].iloc[0])
    num_agg = pd.concat(num_agg, ignore_index=True)
    strbool_agg = pd.concat(strbool_agg, ignore_index=True)
    date_agg = pd.concat(date_agg, ignore_index=True)
    merge_df = pd.concat([num_agg, strbool_agg, date_agg], axis=1)
    merge_df.insert(0, 'case_id', id_case)
    merge_df.insert(1, 'num_group1', group1)
    return merge_df

# 4.Read data
def read_file(paths, depth=None):
    files = []
    for path in glob(str(paths)):
        df = pd.read_csv(path)
        df = filter_cols(df)
        df = set_data_type(df)
        if depth == 1:
            df = dep1(df, 1)
        elif depth == 2:
            df = dep2(df)
        elif depth == 0:
            df = dep0(df)
        files.append(df)
    files.sort(key=lambda x: len(x.columns), reverse=True)
    df = pd.concat(files, ignore_index=True)

    return df

root = Path("D:/data/kaggle")
train_path = root / "train"

data_train = {
    "df_base": read_file(train_path / "train_base.csv"),
    "depth_0": [
        read_file(train_path / "train_static_cb_0.csv", 0),
        read_file(train_path / "train_static_0_*.csv", 0),
    ],
    "depth_1": [
        read_file(train_path / "train_applprev_1_*.csv", 1),
        read_file(train_path / "train_tax_registry_a_1.csv", 1),
        read_file(train_path / "train_tax_registry_b_1.csv", 1),
        read_file(train_path / "train_tax_registry_c_1.csv", 1),
        read_file(train_path / "train_credit_bureau_a_1_*.csv", 1),
        read_file(train_path / "train_credit_bureau_b_1.csv", 1),
        read_file(train_path / "train_other_1.csv", 1),
        read_file(train_path / "train_person_1.csv", 1),
        read_file(train_path / "train_deposit_1.csv", 1),
        read_file(train_path / "train_debitcard_1.csv", 1),
    ],
    "depth_2": [
        read_file(train_path / "train_credit_bureau_b_2.csv", 2),
        read_file(train_path / "train_credit_bureau_a_2_*.csv", 2),
        read_file(train_path / "train_applprev_2.csv", 2),
        read_file(train_path / "train_person_2.csv", 2)
    ]
}


# 5.Constructing the input dataset
depth0 = pd.concat(data_train['depth_0'], ignore_index=True)
depth1 = pd.concat(data_train['depth_1'], ignore_index=True)
depth2 = pd.concat(data_train['depth_2'], ignore_index=True)
depth2 = dep1(depth2, 2)

depth0 = str_to_int(depth0)
depth1 = str_to_int(depth1)
depth2 = str_to_int(depth2)

data_input = data_train["df_base"]
for df in [depth0, depth1, depth2]:
    data_input = pd.merge(data_input, df, on='case_id', how='left')

data_input = data_input.drop_duplicates(subset='case_id')

X = data_input.drop(columns=['target', 'date_decision', 'MONTH', 'WEEK_NUM'])
y = data_input['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 6.Training Model
lgb = LGBMClassifier()
parameters = {'n_estimators': range(100, 1000, 200), 'subsample': [0.8, 0.9, 1.0],
              'colsample_bytree': [0.8, 0.9, 1.0], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
grid_search = GridSearchCV(lgb, parameters, cv=3)
lgb.fit(X_train.values, y_train.values)
y_pred = lgb.predict(X_test.values)

print("Accuracy Rate：" + str(accuracy_score(y_test, y_pred)))