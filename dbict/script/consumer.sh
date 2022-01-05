#!/bin/bash

prev_serial=`ps -ef | grep 'python3 /home/user/jetson-inference/dbict/python/consumer.py' | grep -v 'grep' | awk '{print $2}'`



if [[ -z $prev_serial ]]; then
    echo "start kafka consumer program"
    echo "123" | sudo -S nohup python3 /home/user/jetson-inference/dbict/python/consumer.py &
else
    exit 0
fi

while true
do
    serial=`ps -ef | grep 'python3 /home/user/jetson-inference/dbict/python/consumer.py' | grep -v 'grep' | awk '{print $2}'`
    if [[ -z $serial ]]; then
        echo "restart kafka consumer program"
        echo "123" | sudo -S nohup python3 /home/user/jetson-inference/dbict/python/consumer.py &
    fi
    sleep 2
done
