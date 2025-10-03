import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def predict_stock_price_lstm(input_seq,closing,company):

    if company == 'Amazon':
        path = os.path.join(BASE_DIR,'lstm_scaler_amazon.pkl')
        with open(path, "rb") as f:
            scaler = pickle.load(f)

        loaded_model = tf.keras.models.load_model('lstm_returns_Amazon.h5', compile=False)
        loaded_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse') 
        #os.chdir("/home/piyush/StockTransformer/Amazon")  
   

    else:
        path = os.path.join(BASE_DIR,'lstm_scaler_apple.pkl')
        with open(path ,  "rb") as f:
            scaler = pickle.load(f)

        loaded_model = tf.keras.models.load_model(os.path.join(BASE_DIR,'lstm_returns_Apple.h5'), compile=False)
        loaded_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')
        #os.chdir("/home/piyush/StockTransformer/Apple")       

    
    # Stock_dataframe = pd.read_csv(file_name)

    # test_close = []
    # for i in range(len(Stock_dataframe)):
    #     test_close.append(Stock_dataframe['Close'][i])

    # test_close = np.array(test_close)
    # print(len(input_seq))
    # print(input_seq[6])
    input = []
    for i in range(len(input_seq)):
        input.append(input_seq[i][3])
    test_returns = np.array(input)
    test_returns_scaled = scaler.transform(test_returns.reshape(-1,1))

    window_size = 9 

    #for i in range(len(test_returns_scaled)-window_size):
    input_seq = test_returns_scaled.reshape(1, window_size,1)
    pred_scaled = loaded_model.predict(input_seq)

    pred_return = pred_scaled[0,0] * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

    prev_price = closing
    predicted_price = prev_price * (1 + pred_return)


    print(f"Predicted Close: {predicted_price:.2f}")

    return predicted_price    

