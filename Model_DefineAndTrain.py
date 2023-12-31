import torch
import torch.nn as nn 
import math
from torch import nn, Tensor
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic
import logging
from joblib import load

# 第一句设置了日志的样式和什么级别的信息需要被记录。比如，只记录"INFO"级别及以上的信息。
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                    datefmt="[%Y-%m-%d %H:%M:%S]")
# 第二句创建了一个日志记录器，这样你就可以在程序的其他地方用这个记录器来保存日志信息
logger = logging.getLogger(__name__)

class PositionalEncoder(nn.Module):
    def __init__(self,dropout:float = 0.1,max_seq_len: int = 5000, d_model:int = 512,batch_first:bool = False):
        super().__init__()
        # dropout dropout率
        # max_seq_len 最大序列长度
        # d_model 模型维度
        # batch_first 批处理维度是否在第一个维度
        self.d_model = d_model 
        self.dropout = nn.Dropout(p=dropout) 
        self.batch_first = batch_first 
        self.x_dim = 1 if batch_first else 0
        # position是一个从0到max_seq_len-1的整数序列，用于表示序列中每个位置的索引
        position = torch.arange(max_seq_len).unsqueeze(1)
        # div_term是一个用于调整正弦和余弦函数频率的因子
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0)/d_model))
        
        # 形状为(max_seq_len, 1, d_model)的零张量，用于存储计算出的位置编码
        pe = torch.zeros(max_seq_len, 1, d_model)
        # 使用正弦函数（sin）编码偶数索引位置
        pe[:,0,0::2] = torch.sin(position*div_term)
        # 使用余弦函数（cos）编码奇数索引位置
        pe[:,0,1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)
    
    def forward(self,x:Tensor) -> Tensor:

        x = x + self.pe[:x.size(self.x_dim)] 
        # 添加位置编码
        # # 输入张量x与位置编码pe相加。这里，pe的前x.size(self.x_dim)行被用于与x相加，以确保它们的尺寸匹配

        return self.dropout(x) # 通过一个Dropout层进行正则化
    

class Time_Transformer(nn.Module):
    def __init__(self,input_size: int, n_heads: int =0, batch_first: bool = False, #out_seq_len : int =58,#dec_seq_len: int,
                 dim_val: int =512, n_encoder_layers: int=4, n_decoder_layers: int =4, 
                 dropout_encoder: float=0.2, dropout_decoder: float =0.2,dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int= 2048, dim_feedforward_decoder: int =2048, num_predicted_features: int =1): 
        super().__init__()

        #self.dec_seq_len = dec_seq_len

        #线性层：用于将输入数据映射到高维空间
        self.encoder_input_layer = nn.Linear(
            in_features = input_size,#模型输入的变量数量
            out_features= dim_val
        )
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
            #,max_seq_len=max_seq_len
            # max_seq_len用于定义位置编码的最大长度，以便能够处理不超过这个长度的任何输入序列。
        )
        #创建一个名为encoder_layer的对象，使用PyTorch的TransformerEncoderLayer类，包含自注意力机制和前馈网络
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder, # 前馈神经网络的隐藏层维度
            dropout=dropout_encoder,
            batch_first=batch_first
        )
        #这个encoder_layer对象被用作参数，传递给torch.nn.TransformerEncoder，以便堆叠四个相同的编码器层
        self.encoder = nn.TransformerEncoder(
            encoder_layer= encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )
        
        self.decoder_input_layer = nn.Linear(
            in_features = 1,#8,#num_predicted_features, #,
            out_features = dim_val
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder, # 前馈神经网络的隐藏层维度
            dropout=dropout_decoder,
            batch_first=batch_first
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

    # 定义模型的前向传播逻辑
    def forward(self,src:Tensor,tgt:Tensor,src_mask:Tensor=None,tgt_mask:Tensor=None) -> Tensor:
        # src: 编码器的输入序列。
        # tgt: 解码器的输入序列。
        # src_mask 和 tgt_mask: 这两个是用于遮盖序列的，通常用于处理不同长度的输入。
        
        src = self.encoder_input_layer(src)# src（源序列）通过编码器的输入层————把指标拆到高维
        src = self.positional_encoding_layer(src)# src通过位置编码层

        src = self.encoder(src=src)# src通过所有堆叠的编码器层————捕获"历史销量"和其他7个变量之间的关系

        decoder_output = self.decoder_input_layer(tgt)# tgt（目标序列）通过解码器的输入层————把指标拆到高维

        # 使用编码器的输出src（记忆）和目标序列tgt进行解码
        decoder_output = self.decoder(
            tgt = decoder_output,#tgt作为目标输入
            memory = src,#src作为记忆输入
            tgt_mask = tgt_mask,
            memory_mask = src_mask
        )
        # 对解码器的输出进行线性映射，以得到最终的预测结果
        decoder_output = self.linear_mapping(decoder_output)


        #ps：没有使用遮盖（masking），因为所有输入序列自然地具有相同的长度
        return decoder_output



class SensorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)#读出来变成dataframe
        self.root_dir = root_dir
        self.transform = MinMaxScaler()#归一化
        self.T = training_length
        self.S = forecast_window

    # 统计传感器的数量
    def __len__(self):
        return len(self.df.groupby(by=["reindexed_id"]))#同一个id的传感器聚合起来，统计有多少个传感器

    # 以“油站油品”为粒度的滑动窗口，窗口长度=30，且对"历史销量"做归一化处理
    def __getitem__(self, idx):
        self.df = self.df.sort_values(by='日期')
        if len(self.df[self.df["reindexed_id"] == idx]) - self.T - self.S<6:
            idx = idx + 1
        start=0
        # start = np.random.randint(0, len(self.df[self.df["reindexed_id"] == idx]) - self.T - self.S)
        sensor_number = str(self.df[self.df["reindexed_id"] == idx][["item_id"]][start:start + 1].values.item())
        # training data  随机生成的训练数据index
        index_in = torch.tensor([i for i in range(start, start + self.T)])
        # forecast data  随机生成的预测数据index
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        # torch.Size([30, 8])
        _input = torch.tensor(self.df[self.df["reindexed_id"] == idx][
                                  ["历史销量", "当天温度", "当天油价",'节假日_embed_0', '当时天气_embed_0', '节假日_embed_1','当时天气_embed_1', '节假日_embed_2', '当时天气_embed_2', 'item_id_embed_0',
       'item_id_embed_1', 'item_id_embed_2']][#, "sin_day", "cos_day", "sin_month", "cos_month","year"
                              start: start + self.T].values)
        # torch.Size([12, 8])
        target = torch.tensor(self.df[self.df["reindexed_id"] == idx][
                                  ["历史销量"]][#, "当天温度", "当天油价"#, "sin_day", "cos_day", "sin_month", "cos_month","year"
                              start + self.T-1: start + self.T + self.S].values)

        # # 对 _input 和 target 张量中的第一列（即 "历史销量"）进行归一化处理
        # scaler = self.transform
        # scaler.fit(_input[:, 0].unsqueeze(-1))
        # _input[:, 0] = torch.tensor(scaler.transform(_input[:, 0].unsqueeze(-1)).squeeze(-1))
        # target[:, 0] = torch.tensor(scaler.transform(target[:, 0].unsqueeze(-1)).squeeze(-1))

        # dump(scaler, 'scalar_item.joblib')
        idx = idx + 1
        return index_in, index_tar, _input, target, sensor_number

# # 生成掩码：函数生成上三角矩阵，其中对角线上方的元素是 -inf（负无穷），对角线上是 0
# # 这种类型的掩码用于 Transformer 的自注意力机制，以防止模型在预测某个元素时查看该元素之后的信息。

# dim1 和 dim2 是生成掩码的维度
def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.

    Args:
        dim1: int, for both src and tgt masking, this must be target sequence length
        dim2: int, for src masking this must be encoder sequence length (i.e. the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def flip_from_probability(p):
    return True if random.random() < p else False
# save train or validation loss
def log_loss(loss_val: float, path_to_save_loss: str, train: bool = True):
    if train:
        file_name = "train_loss.txt"
    else:
        file_name = "val_loss.txt"

    path_to_file = path_to_save_loss + file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "a") as f:
        f.write(str(loss_val) + "\n")
        f.close()


def plot_loss(path_to_save, train=True):
    plt.rcParams.update({'font.size': 10})
    with open(path_to_save + "/train_loss.txt", 'r') as f:
        loss_list = [float(line) for line in f.readlines()]
    if train:
        title = "Train"
    else:
        title = "Validation"
    EMA_loss = EMA(loss_list)
    plt.plot(loss_list, label="loss")
    plt.plot(EMA_loss, label="EMA loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title + "_loss")
    plt.savefig(path_to_save + f"/{title}.png")
    plt.close()


def plot_training_3(epoch, path_to_save, src, sampled_src, prediction, sensor_number, index_in, index_tar):
    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))

    idx_scr = [i for i in range(len(src))]
    # index shift 1 for compare prediction and real of next day
    idx_pred = [i for i in range(1, len(prediction) + 1)]
    idx_sampled_src = [i for i in range(len(sampled_src))]
    np.savetxt('src.txt', src, fmt='%f', delimiter='  ', newline='\n')
    np.savetxt('prediction.txt', prediction, fmt='%f', delimiter='  ', newline='\n')
    np.savetxt('sampled_src.txt', sampled_src, fmt='%f', delimiter='  ', newline='\n')

    plt.figure(figsize=(15, 6))
    plt.rcParams.update({"font.size": 18})
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', linestyle='--', alpha=0.5)
    plt.minorticks_on()

    # REMOVE DROPOUT FOR THIS PLOT TO APPEAR AS EXPECTED !!
    # DROPOUT INTERFERES WITH HOW THE SAMPLED SOURCES ARE PLOTTED
    plt.plot(idx_sampled_src, sampled_src, 'o-.', color='red', label='sampled source', linewidth=1, markersize=10)
    plt.plot(idx_scr, src, 'o-.', color='blue', label='input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color='limegreen', label='prediction sequence', linewidth=1)
    plt.title("Teaching Forcing from Sensor " + str(sensor_number[0]) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save + f"/Epoch_{str(epoch)}.png")
    plt.close()


def transformer(dataloader, EPOCH, k,  path_to_save_model, path_to_save_loss, path_to_save_predictions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device)
    print("---device---", device)

    # #tgt_mask：这是一个目标序列（通常是解码器的输入）的掩码。它的形状是 [output_sequence_length, output_sequence_length]。
    # # 这用于确保解码器在生成第 i 个输出元素时，只能查看到第 i 个及其之前的元素。
    # tgt_mask = generate_square_subsequent_mask(
    #     dim1=4,
    #     dim2=4
    # )
    # #src_mask：这是一个源序列（通常是编码器的输入）的掩码。它的形状是 [output_sequence_length, enc_seq_len]。
    # # 这用于可能的源和目标序列长度不匹配的情况
    # src_mask = generate_square_subsequent_mask(
    #     dim1=4,
    #     dim2=60
    #     )   
    # src_mask = src_mask.double()
    # tgt_mask = tgt_mask.double()

    model = Time_Transformer(
        input_size=12,#8 
        #dec_seq_len=dec_seq_len,
        # max_seq_len=max_seq_len,
        # out_seq_len=output_sequence_length, 
        # n_decoder_layers=n_decoder_layers,
        # n_encoder_layers=n_encoder_layers,
        n_heads=8)
    model = model.double()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0
        val_loss = 0

        model.train()
        for index_in, index_tar, _input, target, sensor_number in dataloader:

            # Shape of _input : [batch, input_length, feature]  torch.Size([1, 30, 8])
            # Desired input for model: [input_length, batch, feature]  torch.Size([30, 1, 8])

            optimizer.zero_grad()#对每个batch的数据做训练之间，会做一次梯度清零

            # # choose which value: real or prediction
              # t0 torch.Size([1, 1, 8])
            src = _input.permute(1, 0, 2).double().to(device) # torch.Size([29, 1, 8])——[有多少条记录, 每条记录的维度, 7]
            tgt = target.permute(1, 0, 2).double().to(device)  # src shifted by 1.
            # sampled_src = src[:1, :, :]
            prediction = model(src,tgt) 
            # prediction = model(src,tgt, src_mask, tgt_mask)
            
            loss = criterion(target.permute(1, 0, 2)[:, :, 0].unsqueeze(-1), prediction)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        if train_loss < min_train_loss:
            if epoch % 20 == 0:
                torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
                torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
                best_model = f"best_train_{epoch}.pth"
            min_train_loss = train_loss



        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        predictions = []
        targets = []

        model.eval()  # 切换模型为评估模式
        with torch.no_grad():
            for index_in, index_tar, _input, target, sensor_number in dataloader:
                src = _input.permute(1, 0, 2).double().to(device)
                tgt = target.permute(1, 0, 2).double().to(device)
                # prediction = model(src, tgt, src_mask, tgt_mask)
                prediction = model(src, tgt)
                predictions.extend(prediction.cpu().numpy())
                targets.extend(target.cpu().numpy())
        predictions_reshaped = np.squeeze(predictions)
        print(predictions_reshaped)
        targets_reshaped = np.concatenate(targets).reshape(-1)
        # 绘制预测值和真实值
        plt.figure()
        plt.plot(predictions_reshaped, label='Predictions')
        plt.plot(targets_reshaped, label='Targets')
        plt.legend()
        plt.title(f'Epoch {epoch}')

        # 确保保存路径存在
        if not os.path.exists(path_to_save_predictions):
            os.makedirs(path_to_save_predictions)

        plt.savefig(os.path.join(path_to_save_predictions, f'plot_epoch_{epoch}.png'))
        plt.close()

    # plot_loss(path_to_save_loss, train=True)
    return best_model

#从一个给定的时间序列中生成三个不同的序列：编码器输入（src）、解码器输入（trg）和目标序列（trg_y）

# sequence: 一个1D的PyTorch张量，其长度应等于编码器输入长度（enc_seq_len）和目标序列长度（target_seq_len）之和。
# enc_seq_len: 编码器输入的期望长度。
# target_seq_len: 目标序列的期望长度。
def get_src_trg(
        self,
        sequence: torch.Tensor, 
        enc_seq_len: int, 
        target_seq_len: int
        ) -> list[torch.tensor, torch.tensor, torch.tensor]:
    
    assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

    src = sequence[:enc_seq_len] 

    trg = sequence[enc_seq_len-1:len(sequence)-1]#src 的最后一个元素和 trg 的第一个元素是相同的，即 sequence[enc_seq_len-1]
    trg = trg[:, 0] #从二维张量（Tensor）trg中提取了第一列（索引从0开始）

    if len(trg.shape) == 1: #trg 如果是一维的，会在最后一个维度上增加一个新的维度
        trg = trg.unsqueeze(-1)

    trg_y = sequence[-target_seq_len:]#提取最后 target_seq_len 个元素

    trg_y = trg_y[:, 0]

    return src, trg, trg_y.squeeze(-1)

train_dataset = SensorDataset(csv_name="临阵磨枪.csv", root_dir="/Users/hanqiyu/Desktop/易万里/transformer/要发论文", 
                              training_length=20,forecast_window=3)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#batch_size=1 指的是每个批次（batch）从 train_dataset 中抽取的数据样本数为1。在深度学习中，批次是用于在每个优化步骤中更新网络权重的数据子集。


test_dataset = SensorDataset(csv_name="test.csv", root_dir="/Users/hanqiyu/Desktop/transformer/要发论文", 
                             training_length=30,forecast_window=12)
# test.csv里每个油站油品有66条数据，but这块是不是对test来讲不应该随机抽了，就应该要最后那42条？
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

best_model = transformer(train_dataloader, 5, 60,  "/Users/hanqiyu/Desktop/易万里/transformer/要发论文/模型保存", "/Users/hanqiyu/Desktop/易万里/transformer/要发论文/模型保存/loss",
                            "/Users/hanqiyu/Desktop/易万里/transformer/要发论文/模型保存/prediction")
    








 


