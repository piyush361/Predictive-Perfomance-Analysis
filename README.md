# Predictive Performance Analysis

This project demonstrates the predictive performance of two deep learning models based on their ability to forecast the closing prices of various stock symbols. Both models are widely used in time-series forecasting:

- **Stacked LSTM**
- **Transformer (Encoder-only architecture)**

The system includes full model pipelines, a FastAPI backend, a Kafka-based communication setup, and a simple HTML frontend for visualization.

---

##  Model Architecture

### **1. Stacked LSTM Model**
A recurrent model designed to learn short-term temporal dependencies in stock return sequences.

- **Input Layer:**  
  - Accepts **9 timestamps**, each representing 1 day of price returns

- **LSTM Layers:**  
  - **Two LSTM layers stacked**
  - Both LSTM layers use the **same number of neurons**
  - The first LSTM layer processes the input sequence  
  - The second LSTM layer further refines temporal patterns learned from the first

- **Dense Output Layer:**  
  - A fully connected layer converts the final LSTM output into the closing-price prediction

- **Purpose:**  
  - Captures sequential patterns and short-term dependencies found in financial time-series data

---

### **2. Transformer (Encoder-Only) Model**
A multi-head self-attention model capable of learning long-range dependencies.

- **Input Layer:**  
  - Accepts **18 timestamps**. Training data has all 4 open , high , low , close returns

- **Positional Encoding:**  
  - Adds time-step information that the attention mechanism alone cannot represent

- **Encoder Blocks:**  
  - **Two Transformer Encoder Blocks**
    - Each block contains:
      - **Multi-Head Self-Attention (2 heads)**
      - **Layer Normalization**
      - **Feed-Forward Network (FFN)**  
        - Two dense hidden layers, each with **64 neurons**  

- **Dropout Layer:**  
  - Added at the end of the encoder stack to reduce overfitting

- **Purpose:**  
  - Learns long-range relationships and global temporal structure across price sequences


## Project Architecture

- Amazon/ [Datasets]
- Apple/  [Datasets]
- LSTMs 
  - lstm_scaler.pkl (for two stock symbols)
  - LSTM_prediction.py (has prediction function)   
- Transformers
  - x_scaler and y_scaler.pkl (for training(OHLC) and test(C) data)
  - Architecture.py
  - TFM_Prediction.py (has prediction function)
- backend.py [FastApi backend that handles requests from frontend]
- consumer.py and producer.py [local Kafka test files]
- index.html [frontend]


---

## Installation 

> You need to have kafka installed on your local. Start a cluster in kraft mode (without zookeeper). Before starting , make sure you start a new cluster (as logs are written in tmp/kraft-combined-logs). For making new cluster to starting it , use following list of commands.


```bash
bin/kafka-storage.sh random-uuid
bin/kafka-storage.sh format --standalone -t pK9Yh3uFSxiKnC4M0ZsSDQ -c config/server.properties

bin/kafka-server-start.sh config/server.properties
```

> After starting the kafka , run below command to start the backend.

```bash [python]
uvicorn backend:app
```



