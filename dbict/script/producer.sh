#!/bin/bash

prev_producer=`ps -ef | grep 'python3 /home/user/jetson-inference/dbict/python/producer.py' | grep -v 'grep' | awk '{print $2}'`



if [[ -z $prev_producer ]]; then
    echo "start kafka producer program"
    echo "123" | sudo -S nohup python3 /home/user/jetson-inference/dbict/python/producer.py &
else
    exit 0
fi

while true
do
    producer=`ps -ef | grep 'python3 /home/user/jetson-inference/dbict/python/producer.py' | grep -v 'grep' | awk '{print $2}'`
    if [[ -z $producer ]]; then
        echo "restart kafka producer program"
        echo "123" | sudo -S nohup python3 /home/user/jetson-inference/dbict/python/producer.py &
    fi
    sleep 2
done
