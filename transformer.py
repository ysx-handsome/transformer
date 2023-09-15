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

freq = 'D'
lags_sequence = get_lags_for_frequency(freq)#一个表示滞后序列的列表。给定频率的 lags(滞后): 这将决定模型“回头看”的程度，也会作为附加特征
print(lags_sequence)
time_features = time_features_from_frequency_str(freq)#时间
print(time_features)

@lru_cache(10_000)#使函数具有缓存功能，可以缓存最近调用的10,000个不同的输入及其结果，以减少重复计算和提高效率
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch


train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]
train_dataset.set_transform(partial(transform_start_field, freq=freq))
test_dataset.set_transform(partial(transform_start_field, freq=freq))

prediction_length=14
config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,#预测的时间长度
    context_length=prediction_length * 2,#用于预测的历史数据的时间长度，这里设置为预测长度的两倍
    
    lags_sequence=lags_sequence,#一个表示滞后序列的列表
    num_time_features=len(time_features) + 1,#时间特征的数量
    num_static_categorical_features=1,#静态分类特征的数量
    cardinality=[len(train_dataset)],#表示静态分类特征可能的不同值的数量，这里是训练数据集的长度
    embedding_dimension=[2],#嵌入维度
    
    # transformer params:
    encoder_layers=4,#编码器和解码器的层数，这里都设置为4
    decoder_layers=4,
    d_model=32,#模型的维度，这里设置为32
)

model = TimeSeriesTransformerForPrediction(config)#创建模型实例


##上面的绝对没问题
#143-355有问题

def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    #根据config配置来确定是否需删除某些字段（例如，如果某个特定的特征数量为0），并将这些字段的名称添加到remove_field_names列表中
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # 创建一个Chain对象，它是一个Transformation链，可以将多个转换序列在一起执行
    return Chain(
        # step 1: 移除不需要的字段
        [RemoveFields(field_names=remove_field_names)]
        # step 2: 将某些字段转换为NumPy数组
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: 添加"观察到的值"指示器：这个转换器处理目标字段中的NaN值，填充缺失值并生成一个表示哪些值是被观测到的掩码。
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: 添加时间特征：根据数据集的频率添加时间特征，这些特征可以作为位置编码来使用
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: 添加年龄特征：添加一个表示时间序列在其生命周期中的位置的时间特征
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: 垂直堆叠所有时间特征
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: 重命名字段
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )
def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(
        stream, is_train=True
    )
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )
def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test")

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )
train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
)

test_dataloader = create_test_dataloader(
    config=config,
    freq=freq,
    data=test_dataset,
    batch_size=64,
)
batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape, v.type())

outputs = model(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"]
    if config.num_static_categorical_features > 0
    else None,
    static_real_features=batch["static_real_features"]
    if config.num_static_real_features > 0
    else None,
    future_values=batch["future_values"],
    future_time_features=batch["future_time_features"],
    future_observed_mask=batch["future_observed_mask"],
    output_hidden_states=True,
)
#对这个批次前向传播会报错，感觉参数可能有问题
