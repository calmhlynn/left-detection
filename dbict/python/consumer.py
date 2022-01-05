from kafka import KafkaProducer, KafkaConsumer
from json import loads
import sysv_ipc
import socket
import time
import os
import json
import sys

KAFKA_SERVERS  = 'localhost:9092'
kafka_consumer = KafkaConsumer(
            'control',
            bootstrap_servers = KAFKA_SERVERS,
            auto_offset_reset = 'latest',
            enable_auto_commit = True,
            value_deserializer = lambda x: loads(x.decode('utf-8'))
)
class RUN_Kafka :
    
    def __init__(self, kafka_consumer):
        self.kafka_consumer = kafka_consumer


    def get_msg(self):
        for msg in self.kafka_consumer :
            for status in msg.value.keys():
                print(status)
                if status == 'region':
                    for x_y_coordinate in msg.value.values():
                        print(x_y_coordinate)
                        kafka.region(x_y_coordinate)

                elif status == 'signal':
                    for sig in msg.value.values():
                        kafka.signal(sig)
                        
                elif status == 'stddev':
                    for std in msg.value.values():
                        kafka.stddev(std)


    def region(self, x_y_coordinate):
        os.mkdir("/home/user/jetson-inference/dbict/control/mod")
        region_file_path = "/home/user/jetson-inference/dbict/control/region.txt"
        file = open(region_file_path, "w")
        file.write(x_y_coordinate)
        file.close()

    def signal(self, sig):
        if sig == 'oo':
            os.mkdir("/home/user/jetson-inference/dbict/control/uarton")
            print("make on signal")
        elif sig == 'ox':
            os.rmdir("/home/user/jetson-inference/dbict/control/uarton")
            print("remove on signal")
        elif sig == 'xo':
            os.mkdir("/home/user/jetson-inference/dbict/control/uartoff")
            print("make off signal")
        elif sig == 'xx':
            os.rmdir("/home/user/jetson-inference/dbict/control/uartoff")
            print("remove off signal")

    def stddev(self, std):
        os.mkdir("/home/user/jetson-inference/dbict/control/std")
        region_file_path = "/home/user/jetson-inference/dbict/control/stddev.txt"
        file = open(region_file_path, "w")
        file.write(std)
        file.close()

kafka = RUN_Kafka(kafka_consumer)

while True:
    try:
        kafka.get_msg()


    except Exception as e:
        print("Except : ", e)
        time.sleep(10)
        sys.exit()