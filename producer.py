from confluent_kafka import Producer
import time

conf = {'bootstrap.servers': "localhost:9092"}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print("Delivery failed:", err)
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")


time.sleep(1.5)


for i in range(7):
    producer.produce("test-topic", key="key1", value=f"Hello Kafka!{i}", callback=delivery_report)
    producer.poll(0)
    time.sleep(2)

producer.flush()
