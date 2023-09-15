import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from functools import lru_cache
from functools import partial

#数据读入与处理
df=pd.read_csv("/Users/hanqiyu/Desktop/9.6崭新的数据_含档位/transformer/1_数据输入.csv")
df['日期'] = pd.to_datetime(df['日期'])
df=df.drop(columns=['油站名称','油品名称'])
# 定义一个用于计算第一个非空值的函数
first = lambda x: x.iloc[0]
# 定义聚合函数
aggregation_functions = {
    '油站编码': first,
    '油品编码': first,
    '日期': first,
    '历史销量': 'sum',
    '当天温度': 'mean',
    '当天油价': 'mean',
    '节假日': first,
    '当时天气': first
}
result_df = df.groupby(['油站编码', '油品编码', '日期'], as_index=False).agg(aggregation_functions)
result_df['item_id'] = result_df.apply(lambda row: f"{row['油站编码']}_{row['油品编码']}", axis=1)
item_id_counts = result_df['item_id'].value_counts()

train_df = result_df[result_df['日期'] < '2023-06-01']
validation_df = result_df[result_df['日期'] < '2023-06-15']#验证集比训练集多15天
test_df = result_df#测试集比训练集多30天

def create_samples(df):
    grouped = df.groupby(['油站编码', '油品编码'])
    samples = []
    for (station_code, product_code), group in grouped:
        sample = {
            'start': group['日期'].min().strftime('%Y-%m-%d'), 
            'target': group['历史销量'].tolist(),
            'feat_static_cat': [],
            'feat_dynamic_real': [group['当天温度'].tolist(), group['当天油价'].tolist()], 
            #, group['节假日'].tolist(), group['当时天气'].tolist()   这些非字符串的动态的变量先放放，后面再弄
            'item_id': f"{station_code}_{product_code}"
        }
        samples.append(sample)
    return samples

# 使用 create_samples 函数创建样本集
train_samples = create_samples(train_df)
validation_samples = create_samples(validation_df)
test_samples = create_samples(test_df)

# 创建 DatasetDict
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(train_samples)),
    'test': Dataset.from_pandas(pd.DataFrame(test_samples)),
    'validation': Dataset.from_pandas(pd.DataFrame(validation_samples)),
})
