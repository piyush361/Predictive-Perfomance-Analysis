from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'test-topic',  
    bootstrap_servers=['localhost:9092'],  
    group_id='my_consumer_group',  
    auto_offset_reset='earliest', 
)

try:
    while True:
        msg_pack = consumer.poll(timeout_ms=100)
        for tp,msgs in msg_pack.items():
            for message in msgs:
                print(f"message key : {message.key} , value : {message.value}")

except KeyboardInterrupt:
    print("Closed by keyboard")

finally:
    consumer.close()

