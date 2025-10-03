import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PositionalEncoding(nn.Module):

  def __init__(self, d_model , max_len):
    self.d_model = d_model
    self.max_len = max_len
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
     x = x + self.pe[:, :x.size(1),:]
     return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, ff_hidden, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.ffn(x)
        x = self.norm2(x + self.dropout(x2))
        return x


class StockTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=32, seq_len=18, num_heads=2, ff_hidden=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder1 = TransformerEncoderBlock(d_model, ff_hidden, num_heads, dropout)
        self.encoder2 = TransformerEncoderBlock(d_model, ff_hidden, num_heads, dropout)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = x[-1, :, :]
        out = self.regressor(x)
        return out.squeeze(-1)



def predict_stock_price_TF(input_seq,closing,company):

    if company == 'Amazon':
        with open(os.path.join(BASE_DIR,'x_scaler_amzn.pkl'), "rb") as f:
            X_scaler = pickle.load(f)
        with open(os.path.join(BASE_DIR,'y_scaler_amzn.pkl') , "rb") as f:
            y_scaler = pickle.load(f) 

        # torch.serialization.add_safe_globals([StockTransformer])
        # model = StockTransformer(input_dim=4, d_model=32, seq_len=18,
        #                  num_heads=4, ff_hidden=64, dropout=0.1)

        #model = torch.load(os.path.join(BASE_DIR,"stock_transformer_Amazon.pth"),weights_only=False)   
        model = StockTransformer(input_dim=4, d_model=32, seq_len=18,
                         num_heads=2, ff_hidden=64, dropout=0.1)

# Load the state_dict
        state_dict_path = os.path.join(BASE_DIR, "stock_transformer_Amazon_state.pth")
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
        model.eval()   
        #os.chdir("/home/piyush/StockTransformer/Amazon")   
    else:
        with open(os.path.join(BASE_DIR,'x_scaler_apple.pkl') , "rb") as f:
            X_scaler = pickle.load(f)
        with open(os.path.join(BASE_DIR,'y_scaler_apple.pkl') , "rb") as f:
            y_scaler = pickle.load(f) 

        # torch.serialization.add_safe_globals([StockTransformer]) 
        # model = StockTransformer(input_dim=4, d_model=32, seq_len=18,
                        #  num_heads=4, ff_hidden=64, dropout=0.1)

        model = StockTransformer(input_dim=4, d_model=32, seq_len=18,
                         num_heads=2, ff_hidden=64, dropout=0.1)

# Load the state_dict
        state_dict_path = os.path.join(BASE_DIR, "stock_transformer_Apple_state.pth")
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
        model.eval()
        #os.chdir("/home/piyush/StockTransformer/Apple")  

    
    # test = pd.read_csv(file_name)
    # test_data_with_returns = []
    # closing_prices = []
    # for i in range(len(test)-1):
    #     if i != 0:
    #         open_return = (test['Open'][i] - test['Open'][i-1]) / test['Open'][i-1]
    #         high_return = (test['High'][i] - test['High'][i-1]) / test['High'][i-1]
    #         low_return = (test['Low'][i] - test['Low'][i-1]) / test['Low'][i-1]
    #         close_return = (test['Close'][i] - test['Close'][i-1]) / test['Close'][i-1]
    #         closing_prices.append(test['Close'][i])
    #         test_data_with_returns.append([open_return,high_return,low_return,close_return])

    # seq_len = 18
    # X_test = []
    # y_test = []
    # closingfinal=[]

    # for i in range(len(test_data_with_returns)-seq_len):
    #     X_test.append(test_data_with_returns[i:i+seq_len])
    #     y_test.append(test_data_with_returns[i+seq_len][3])
    #     closingfinal.append(closing_prices[i+seq_len])

    X_test = np.array(input_seq)
    #y_test = np.array(y_test)

    
    X_scaled = X_scaler.transform(X_test)
    X_scaled = np.expand_dims(X_scaled,axis=0)
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_scaled.astype(np.float32)))

    pred_return = y_scaler.inverse_transform(pred_scaled.numpy().reshape(-1, 1))[0, 0]
    price = closing * ( 1 + pred_return)
    print(f"Predicted Price : {price}")

    return price

