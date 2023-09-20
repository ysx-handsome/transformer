import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from functools import lru_cache
from functools import partial
from gluonts.time_feature import get_lags_for_frequency
from gluonts.time_feature import time_features_from_frequency_str
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from transformers import PretrainedConfig #PretrainedConfig类，它是用于存储预训练模型的配置信息的类
from gluonts.transform.sampler import InstanceSampler
from typing import Optional
from typing import Iterable

import torch
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches

from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    get_lags_for_frequency,
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)

from typing import Iterable

import torch
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches


#数据读入与处理
file_path = '1_数据输入.csv'
df=pd.read_csv(file_path)
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

