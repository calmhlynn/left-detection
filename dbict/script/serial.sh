#!/bin/bash

prev_serial=`ps -ef | grep 'python3 /home/user/jetson-inference/dbict/python/uart.py' | grep -v 'grep' | awk '{print $2}'`



if [[ -z $prev_serial ]]; then
    echo "start serial program"
    echo "123" | sudo -S nohup python3 /home/user/jetson-inference/dbict/python/uart.py &
else
    exit 0
fi

while true
do
    serial=`ps -ef | grep 'python3 /home/user/jetson-inference/dbict/python/uart.py' | grep -v 'grep' | awk '{print $2}'`
    if [[ -z $serial ]]; then
        echo "restart serial program"
        echo "123" | sudo -S nohup python3 /home/user/jetson-inference/dbict/python/uart.py &
    fi
    sleep 2
done
