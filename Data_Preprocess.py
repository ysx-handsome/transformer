import pandas as pd
import time
import numpy as np
import datetime
from icecream import ic


#数据读入与处理
file_path = '/Users/hanqiyu/Desktop/transformer/要发论文/原始数据.csv'
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

df_input=result_df[['日期','历史销量','当天温度','当天油价','节假日','当时天气','item_id']]


# encoding the timestamp data cyclically. See Medium Article.
def process_data(df):#source):
    #df = pd.read_csv(source)

    # timestamps = [ts.split('+')[0] for ts in df['日期']]
    # timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').day) for t in timestamps])
    # timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').month) for t in timestamps])
    # timestamps_year = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').year) for t in timestamps])
    timestamps_year = np.array([ts.year for ts in df['日期']])
    timestamps_month = np.array([ts.month for ts in df['日期']])
    timestamps_day = np.array([ts.day for ts in df['日期']])

    # hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    # df['sin_hour'] = np.sin(2 * np.pi * timestamps_hour / hours_in_day)
    # df['cos_hour'] = np.cos(2 * np.pi * timestamps_hour / hours_in_day)
    df['sin_day'] = np.sin(2 * np.pi * timestamps_day / days_in_month)
    df['cos_day'] = np.cos(2 * np.pi * timestamps_day / days_in_month)
    df['sin_month'] = np.sin(2 * np.pi * timestamps_month / month_in_year)
    df['cos_month'] = np.cos(2 * np.pi * timestamps_month / month_in_year)
    df['year'] = timestamps_year
    return df


dataset = process_data(df_input)




# #数据日期无法对齐，且日期均有缺失
# df['日期'] = pd.to_datetime(df['日期'])
# grouped = df.groupby('item_id')
# result_list = []

# # 遍历每个分组（即每个item_id）
# for name, group in grouped:
#     group = group.sort_values('日期')
    
#     min_date = group['日期'].min()
#     max_date = group['日期'].max()
#     num=len(group)
    
#     full_range = pd.date_range(min_date, max_date)
    
#     has_missing_dates = not set(group['日期']) == set(full_range)
#     result_list.append({'num':num,'item_id': name, 'min_date': min_date, 'max_date': max_date, 'has_missing_dates': has_missing_dates})

# # 将结果列表转换为DataFrame
# result_df = pd.DataFrame(result_list)
# # 输出结果
# result_df.to_excel("/Users/hanqiyu/Desktop/transformer/要发论文/result.xlsx")




##切割数据集
df['日期'] = pd.to_datetime(df['日期'])
grouped = df.groupby('item_id')
# 初始化一个变量来存储最小的数据条数
min_count = float('inf')

# 初始化两个空列表来存储所有组的截断数据
train_data_list = []
test_data_list = []

# 第一次遍历：找到最小的数据条数
for name, group in grouped:
    count = len(group)
    if count < min_count:
        min_count = count
        
N=12
# 第二次遍历：删除多余的数据并存储结果
for name, group in grouped:
    group = group.sort_values('日期', ascending=False)  # 按日期降序排列，这样我们可以删除日期最大的数据
    train_truncated_group = group.iloc[:min_count-N]  # 只保留前min_count条数据
    test_truncated_group = group.iloc[:min_count]  # 只保留前min_count条数据
    train_data_list.append(train_truncated_group)
    test_data_list.append(test_truncated_group)
train_df = pd.concat(train_data_list)
test_df = pd.concat(test_data_list)

# 输出结果
train_df.to_csv("/Users/hanqiyu/Desktop/transformer/要发论文/train.csv")
test_df.to_csv("/Users/hanqiyu/Desktop/transformer/要发论文/test.csv")



#对item_id的处理
# #unique_item_ids = df['item_id'].unique()
# item_id_to_index = {'MD02_300863': 0, 'MD04_300863': 1, 'MD04_300873': 2, 'MD04_300874': 3, 'MD06_300863': 4, 'MD06_300873': 5, 'MD06_300874': 6, 'MD08_300863': 7, 'MD08_300873': 8, 'MD08_300874': 9, 'MD09_300863': 10, 'MD09_300873': 11, 'MD09_300874': 12, 'MD0A_300863': 13, 'MD0A_300873': 14, 'MD0A_300874': 15, 'MD0B_300863': 16, 'MD0B_300873': 17, 'MD0B_300874': 18, 'MD0C_300863': 19, 'MD0C_300873': 20, 'MD0C_300874': 21, 'MD0D_300873': 22, 'MD0D_300874': 23, 'MD0E_300863': 24, 'MD0E_300873': 25, 'MD0E_300874': 26, 'MD0G_300863': 27, 'MD0G_300873': 28, 'MD0G_300874': 29, 'MD0H_300863': 30, 'MD0H_300873': 31, 'MD0H_300874': 32, 'MD0I_300863': 33, 'MD0I_300873': 34, 'MD0I_300874': 35, 'MD0J_300863': 36, 'MD0J_300873': 37, 'MD0J_300874': 38, 'MD0K_300863': 39, 'MD0K_300873': 40, 'MD0K_300874': 41, 'MD0L_300863': 42, 'MD0L_300873': 43, 'MD0L_300874': 44, 'MD0M_300863': 45, 'MD0M_300873': 46, 'MD0M_300874': 47, 'MD20_300863': 48, 'MD20_300873': 49, 'MD20_300874': 50, 'MD21_300863': 51, 'MD21_300873': 52, 'MD21_300874': 53, 'MD23_300863': 54, 'MD23_300873': 55, 'MD23_300874': 56, 'MD24_300863': 57, 'MD24_300873': 58, 'MD24_300874': 59, 'MD25_300863': 60, 'MD25_300873': 61, 'MD25_300874': 62, 'MD26_300863': 63, 'MD26_300873': 64, 'MD26_300874': 65, 'MD27_300863': 66, 'MD27_300873': 67, 'MD27_300874': 68, 'MD28_300863': 69, 'MD28_300873': 70, 'MD28_300874': 71, 'MD29_300863': 72, 'MD29_300873': 73, 'MD29_300874': 74, 'MD2A_300863': 75, 'MD2A_300873': 76, 'MD2A_300874': 77, 'MD2B_300863': 78, 'MD2B_300873': 79, 'MD2B_300874': 80, 'MD40_300863': 81, 'MD40_300873': 82, 'MD40_300874': 83, 'MD41_300863': 84, 'MD41_300873': 85, 'MD41_300874': 86, 'MD42_300863': 87, 'MD42_300873': 88, 'MD42_300874': 89, 'MD43_300863': 90, 'MD43_300873': 91, 'MD43_300874': 92, 'MD44_300863': 93, 'MD44_300873': 94, 'MD44_300874': 95, 'MD46_300863': 96, 'MD46_300873': 97, 'MD46_300874': 98, 'MD47_300863': 99, 'MD47_300873': 100, 'MD47_300874': 101, 'MD48_300863': 102, 'MD48_300873': 103, 'MD48_300874': 104, 'MD49_300863': 105, 'MD49_300873': 106, 'MD49_300874': 107, 'MD60_300863': 108, 'MD60_300873': 109, 'MD60_300874': 110, 'MD63_300863': 111, 'MD63_300873': 112, 'MD63_300874': 113, 'MD64_300863': 114, 'MD64_300873': 115, 'MD64_300874': 116, 'MD65_300863': 117, 'MD65_300873': 118, 'MD65_300874': 119, 'MD66_300863': 120, 'MD66_300873': 121, 'MD66_300874': 122, 'MD68_300863': 123, 'MD68_300873': 124, 'MD68_300874': 125, 'MD6B_300863': 126, 'MD6B_300873': 127, 'MD6B_300874': 128, 'MD6E_300863': 129, 'MD6E_300873': 130, 'MD6E_300874': 131, 'MD6F_300863': 132, 'MD6F_300873': 133, 'MD6F_300874': 134, 'MD80_300863': 135, 'MD80_300873': 136, 'MD80_300874': 137, 'MD82_300863': 138, 'MD82_300873': 139, 'MD82_300874': 140, 'MD83_300863': 141, 'MD83_300873': 142, 'MD83_300874': 143, 'MD84_300863': 144, 'MD84_300873': 145, 'MD84_300874': 146, 'MD85_300863': 147, 'MD85_300873': 148, 'MD85_300874': 149, 'MD86_300863': 150, 'MD86_300873': 151, 'MD86_300874': 152, 'MD88_300863': 153, 'MD88_300873': 154, 'MD88_300874': 155, 'MD89_300863': 156, 'MD89_300873': 157, 'MD89_300874': 158, 'MD8B_300863': 159, 'MD8B_300873': 160, 'MD8B_300874': 161, 'MD8C_300863': 162, 'MD8C_300873': 163, 'MD8C_300874': 164, 'MD8D_300863': 165, 'MD8D_300873': 166, 'MD8D_300874': 167, 'MD8E_300863': 168, 'MD8E_300873': 169, 'MD8E_300874': 170, 'MD8F_300863': 171, 'MD8F_300873': 172, 'MD8F_300874': 173, 'MD8H_300863': 174, 'MD8H_300873': 175, 'MD8H_300874': 176, 'MD8J_300863': 177, 'MD8J_300873': 178, 'MD8J_300874': 179, 'MD8K_300863': 180, 'MD8K_300873': 181, 'MD8K_300874': 182, 'MD8L_300863': 183, 'MD8L_300873': 184, 'MD8L_300874': 185, 'MDC0_300863': 186, 'MDC0_300873': 187, 'MDC0_300874': 188, 'MDC2_300863': 189, 'MDC2_300873': 190, 'MDC2_300874': 191, 'MDC3_300863': 192, 'MDC3_300873': 193, 'MDC3_300874': 194, 'MDC4_300863': 195, 'MDC4_300873': 196, 'MDC4_300874': 197, 'MDC5_300863': 198, 'MDC5_300873': 199, 'MDC5_300874': 200, 'MDC6_300863': 201, 'MDC6_300873': 202, 'MDC6_300874': 203, 'MDC7_300863': 204, 'MDC7_300873': 205, 'MDC7_300874': 206, 'MDC8_300863': 207, 'MDC8_300873': 208, 'MDC8_300874': 209, 'MDCA_300863': 210, 'MDCA_300873': 211, 'MDCA_300874': 212, 'MDCB_300863': 213, 'MDCB_300873': 214, 'MDCB_300874': 215, 'MDCC_300863': 216, 'MDCC_300873': 217, 'MDCC_300874': 218, 'MDCE_300863': 219, 'MDCE_300873': 220, 'MDCE_300874': 221, 'MDCF_300863': 222, 'MDCF_300873': 223, 'MDCF_300874': 224, 'MDCG_300863': 225, 'MDCG_300873': 226, 'MDCG_300874': 227, 'MDCH_300863': 228, 'MDCH_300873': 229, 'MDCH_300874': 230, 'MDCI_300863': 231, 'MDCI_300873': 232, 'MDCI_300874': 233, 'MDCJ_300863': 234, 'MDCJ_300873': 235, 'MDCJ_300874': 236, 'MDCL_300863': 237, 'MDCL_300873': 238, 'MDCL_300874': 239}
# #{item_id: index for index, item_id in enumerate(unique_item_ids)}
# index_to_item_id = {0: 'MD02_300863', 1: 'MD04_300863', 2: 'MD04_300873', 3: 'MD04_300874', 4: 'MD06_300863', 5: 'MD06_300873', 6: 'MD06_300874', 7: 'MD08_300863', 8: 'MD08_300873', 9: 'MD08_300874', 10: 'MD09_300863', 11: 'MD09_300873', 12: 'MD09_300874', 13: 'MD0A_300863', 14: 'MD0A_300873', 15: 'MD0A_300874', 16: 'MD0B_300863', 17: 'MD0B_300873', 18: 'MD0B_300874', 19: 'MD0C_300863', 20: 'MD0C_300873', 21: 'MD0C_300874', 22: 'MD0D_300873', 23: 'MD0D_300874', 24: 'MD0E_300863', 25: 'MD0E_300873', 26: 'MD0E_300874', 27: 'MD0G_300863', 28: 'MD0G_300873', 29: 'MD0G_300874', 30: 'MD0H_300863', 31: 'MD0H_300873', 32: 'MD0H_300874', 33: 'MD0I_300863', 34: 'MD0I_300873', 35: 'MD0I_300874', 36: 'MD0J_300863', 37: 'MD0J_300873', 38: 'MD0J_300874', 39: 'MD0K_300863', 40: 'MD0K_300873', 41: 'MD0K_300874', 42: 'MD0L_300863', 43: 'MD0L_300873', 44: 'MD0L_300874', 45: 'MD0M_300863', 46: 'MD0M_300873', 47: 'MD0M_300874', 48: 'MD20_300863', 49: 'MD20_300873', 50: 'MD20_300874', 51: 'MD21_300863', 52: 'MD21_300873', 53: 'MD21_300874', 54: 'MD23_300863', 55: 'MD23_300873', 56: 'MD23_300874', 57: 'MD24_300863', 58: 'MD24_300873', 59: 'MD24_300874', 60: 'MD25_300863', 61: 'MD25_300873', 62: 'MD25_300874', 63: 'MD26_300863', 64: 'MD26_300873', 65: 'MD26_300874', 66: 'MD27_300863', 67: 'MD27_300873', 68: 'MD27_300874', 69: 'MD28_300863', 70: 'MD28_300873', 71: 'MD28_300874', 72: 'MD29_300863', 73: 'MD29_300873', 74: 'MD29_300874', 75: 'MD2A_300863', 76: 'MD2A_300873', 77: 'MD2A_300874', 78: 'MD2B_300863', 79: 'MD2B_300873', 80: 'MD2B_300874', 81: 'MD40_300863', 82: 'MD40_300873', 83: 'MD40_300874', 84: 'MD41_300863', 85: 'MD41_300873', 86: 'MD41_300874', 87: 'MD42_300863', 88: 'MD42_300873', 89: 'MD42_300874', 90: 'MD43_300863', 91: 'MD43_300873', 92: 'MD43_300874', 93: 'MD44_300863', 94: 'MD44_300873', 95: 'MD44_300874', 96: 'MD46_300863', 97: 'MD46_300873', 98: 'MD46_300874', 99: 'MD47_300863', 100: 'MD47_300873', 101: 'MD47_300874', 102: 'MD48_300863', 103: 'MD48_300873', 104: 'MD48_300874', 105: 'MD49_300863', 106: 'MD49_300873', 107: 'MD49_300874', 108: 'MD60_300863', 109: 'MD60_300873', 110: 'MD60_300874', 111: 'MD63_300863', 112: 'MD63_300873', 113: 'MD63_300874', 114: 'MD64_300863', 115: 'MD64_300873', 116: 'MD64_300874', 117: 'MD65_300863', 118: 'MD65_300873', 119: 'MD65_300874', 120: 'MD66_300863', 121: 'MD66_300873', 122: 'MD66_300874', 123: 'MD68_300863', 124: 'MD68_300873', 125: 'MD68_300874', 126: 'MD6B_300863', 127: 'MD6B_300873', 128: 'MD6B_300874', 129: 'MD6E_300863', 130: 'MD6E_300873', 131: 'MD6E_300874', 132: 'MD6F_300863', 133: 'MD6F_300873', 134: 'MD6F_300874', 135: 'MD80_300863', 136: 'MD80_300873', 137: 'MD80_300874', 138: 'MD82_300863', 139: 'MD82_300873', 140: 'MD82_300874', 141: 'MD83_300863', 142: 'MD83_300873', 143: 'MD83_300874', 144: 'MD84_300863', 145: 'MD84_300873', 146: 'MD84_300874', 147: 'MD85_300863', 148: 'MD85_300873', 149: 'MD85_300874', 150: 'MD86_300863', 151: 'MD86_300873', 152: 'MD86_300874', 153: 'MD88_300863', 154: 'MD88_300873', 155: 'MD88_300874', 156: 'MD89_300863', 157: 'MD89_300873', 158: 'MD89_300874', 159: 'MD8B_300863', 160: 'MD8B_300873', 161: 'MD8B_300874', 162: 'MD8C_300863', 163: 'MD8C_300873', 164: 'MD8C_300874', 165: 'MD8D_300863', 166: 'MD8D_300873', 167: 'MD8D_300874', 168: 'MD8E_300863', 169: 'MD8E_300873', 170: 'MD8E_300874', 171: 'MD8F_300863', 172: 'MD8F_300873', 173: 'MD8F_300874', 174: 'MD8H_300863', 175: 'MD8H_300873', 176: 'MD8H_300874', 177: 'MD8J_300863', 178: 'MD8J_300873', 179: 'MD8J_300874', 180: 'MD8K_300863', 181: 'MD8K_300873', 182: 'MD8K_300874', 183: 'MD8L_300863', 184: 'MD8L_300873', 185: 'MD8L_300874', 186: 'MDC0_300863', 187: 'MDC0_300873', 188: 'MDC0_300874', 189: 'MDC2_300863', 190: 'MDC2_300873', 191: 'MDC2_300874', 192: 'MDC3_300863', 193: 'MDC3_300873', 194: 'MDC3_300874', 195: 'MDC4_300863', 196: 'MDC4_300873', 197: 'MDC4_300874', 198: 'MDC5_300863', 199: 'MDC5_300873', 200: 'MDC5_300874', 201: 'MDC6_300863', 202: 'MDC6_300873', 203: 'MDC6_300874', 204: 'MDC7_300863', 205: 'MDC7_300873', 206: 'MDC7_300874', 207: 'MDC8_300863', 208: 'MDC8_300873', 209: 'MDC8_300874', 210: 'MDCA_300863', 211: 'MDCA_300873', 212: 'MDCA_300874', 213: 'MDCB_300863', 214: 'MDCB_300873', 215: 'MDCB_300874', 216: 'MDCC_300863', 217: 'MDCC_300873', 218: 'MDCC_300874', 219: 'MDCE_300863', 220: 'MDCE_300873', 221: 'MDCE_300874', 222: 'MDCF_300863', 223: 'MDCF_300873', 224: 'MDCF_300874', 225: 'MDCG_300863', 226: 'MDCG_300873', 227: 'MDCG_300874', 228: 'MDCH_300863', 229: 'MDCH_300873', 230: 'MDCH_300874', 231: 'MDCI_300863', 232: 'MDCI_300873', 233: 'MDCI_300874', 234: 'MDCJ_300863', 235: 'MDCJ_300873', 236: 'MDCJ_300874', 237: 'MDCL_300863', 238: 'MDCL_300873', 239: 'MDCL_300874'}
# #{index: item_id for index, item_id in enumerate(unique_item_ids)}