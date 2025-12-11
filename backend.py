from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from confluent_kafka import Producer
from kafka import KafkaConsumer
from LSTMs.LSTM_Prediction import predict_stock_price_lstm 
from Transformers.TFM_Prediction import predict_stock_price_TF , StockTransformer
from collections import deque
import pandas as pd
import asyncio
import os
import json

app = FastAPI()
clients = set()


conf = {'bootstrap.servers': "localhost:9092"}
producer = Producer(conf)


def delivery_report(err, msg):
    if err is not None:
        print("Delivery failed:", err)
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

@app.websocket("/ws/{set_id}")
async def websocket_endpoint(websocket: WebSocket, set_id: str):
    await websocket.accept()
    clients.add(websocket)
    producer_started = False
    try:
        while True:
            data = await websocket.receive_text()
            print("From client:", data)
            if data == "start_producer" and not producer_started:
                producer_started = True
                asyncio.create_task(produce_messages(set_id=set_id))
                #asyncio.create_task(produce_messages(set_id=set_id,model="Transformer"))
    except WebSocketDisconnect:
        clients.remove(websocket)



async def produce_messages(set_id):
    await asyncio.sleep(2)
    os.chdir("/home/piyush/Predictive_performance_analysis/Apple")       
    df = pd.read_csv(f"Apple_test_set_{set_id}.csv")

    price_returns = []
    for i in range(19):
        if i!=0:
            open_return = (df['Open'][i] - df['Open'][i-1]) / df['Open'][i-1]
            high_return = (df['High'][i] - df['High'][i-1]) / df['High'][i-1]
            low_return = (df['Low'][i] - df['Low'][i-1]) / df['Low'][i-1]
            close_return = (df['Close'][i] - df['Close'][i-1]) / df['Close'][i-1]
            price_returns.append([open_return,high_return,low_return,close_return])

    tf_input_seq = price_returns
    lstm_input_seq = price_returns[9:18]

    #closing = df['Close'][18]
    for i in range(len(df)-19):
        prev_closing = float(df['Close'][i+18])
        closing_price = float(df['Close'][i+19])
        
        next_day_price_tf = predict_stock_price_TF(input_seq=tf_input_seq,closing=prev_closing,company='AAPL')
        next_day_return_tf = (next_day_price_tf - prev_closing)/prev_closing
        tf_input_seq.pop(0)
        tf_input_seq.append([0,0,0,next_day_return_tf])
        
        next_day_price_lstm = predict_stock_price_lstm(input_seq=lstm_input_seq,closing=prev_closing,company="AAPL") 
        next_day_return_lstm = (next_day_price_lstm - prev_closing)/prev_closing
        lstm_input_seq.pop(0)
        lstm_input_seq.append([0,0,0,next_day_return_lstm])
 
        symbol = 'AAPL'
        msg_value = json.dumps({ "symbol": symbol, "actual": closing_price , "tf_predicted" : next_day_price_tf , "lstm_predicted" : next_day_price_lstm})

        producer.produce(
            "stock-topic",
            key=f"stocktest_{set_id}",
            value=msg_value,
            callback=delivery_report
        )
        producer.poll(0)

        await asyncio.sleep(1.5)

    producer.flush()


async def consume_kafka():
    consumer = KafkaConsumer(
        f'stock-topic',
        bootstrap_servers=['localhost:9092'],
        group_id='my_consumer_group',
        auto_offset_reset='earliest',
    )
    while True:
        msg_pack = consumer.poll(timeout_ms=50)
        for tp, msgs in msg_pack.items():
            for message in msgs:
                try:
                    msg_value = message.value.decode()
                    for client in clients.copy():
                        await client.send_text(msg_value)
                except Exception as e:
                    print(f"Error sending to client: {e}")
                    clients.remove(client)

        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(consume_kafka())
