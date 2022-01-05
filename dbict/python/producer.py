from kafka  import KafkaProducer, KafkaConsumer
import sysv_ipc
import json
import time
import sys
import os

KAFKA_SERVERS  = 'localhost:9092'

kafka_producer = KafkaProducer(
            bootstrap_servers = KAFKA_SERVERS)


class RUN_Kafka:

    def __init__(self, Run, CAN, LMB, UART):

        self.Run = Run
        self.CAN = CAN
        self.LMB = LMB
        self.UART = UART

    def check_err(self):
        Run_check = os.path.isdir("/home/user/jetson-inference/dbict/control/run")
        can_err = os.path.isdir("/home/user/jetson-inference/dbict/control/canerr")
        lmb_err = os.path.isdir("/home/user/jetson-inference/dbict/control/lmberr")
        uarton = os.path.isdir("/home/user/jetson-inference/dbict/control/uarton")
        uartoff = os.path.isdir("/home/user/jetson-inference/dbict/control/uartoff")

        if Run_check :
            self.RUN = 1
        else :
            self.RUN = 0
        
        if can_err :
            self.CAN = 0  ## CAN Controller connect failed 
        else :
            self.CAN = 1
        
        if lmb_err :
            self.LMB = 0  ## LMB connect failed
        else :
            self.LMB = 1

        if uartoff :
            self.UART = 2
        elif uarton:
            self.UART = 1
        else :
            self.UART = 0

    def publish_message(self, producer, topic_name, key, value):
        try:
            data = { key : value }
            producer.send(topic_name, json.dumps(data).encode())
        except Exception as e:
            print('Exception in publishing message ->', e)

    def bytes_to_hex(self, one, two ,three, four):

        detect = [format(one,"02X"), format(two,"02X"), format(three,"02X"), format(four,"02X")]

        return detect

RUN = 0
CAN = 0
LMB = 0
UART = 0

kafka = RUN_Kafka(RUN, CAN, LMB, UART)
shm = sysv_ipc.SharedMemory(0x1000)


while True:
    try :
        memory = bytearray(shm.read())
        kafka.check_err()

        kafka.publish_message(kafka_producer, 'log', 'log', [kafka.RUN, kafka.CAN, kafka.LMB, kafka.UART] )
        kafka.publish_message(kafka_producer, 'detect', 'detect', kafka.bytes_to_hex(memory[128+60], memory[128+61], memory[128+62], memory[128+63]))
        time.sleep(0.5)

    except Exception as e:
        print("Except : ", e)
        time.sleep(10)
        sys.exit()