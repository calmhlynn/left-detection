#!/usr/bin/env bash
echo "make recoding forder"
sudo dpkg-reconfigure dash
sudo apt update
sudo apt install openjdk-8-jdk -y
sudo pip3 install pyserial
sudo pip3 install sysv_ipc
sudo pip3 install kafka-python

wget https://archive.apache.org/dist/kafka/2.8.0/kafka_2.13-2.8.0.tgz
tar xvf kafka_2.13-2.8.0.tgz
rm kafka_2.13-2.8.0.tgz
mv kafka_2.13-2.8.0 ${HOME}
export KAFKA_HEAP_OPTS="-Xmx400m -Xms400m"



vi /home/user/kafka_2.13-2.8.0/config/server.properties


sudo chmod +755 ./script/*
sudo cp service/zookeeper.service /etc/systemd/system/
sudo cp service/kafka.service /etc/systemd/system/
sudo cp service/producer.service /etc/systemd/system/
sudo cp service/consumer.service /etc/systemd/system/
sudo cp service/uart.service /etc/systemd/system/

echo "complete service settings"


sudo systemctl daemon-reload
sudo systemctl start zookeeper
sudo systemctl start kafka
sudo systemctl start uart
sudo systemctl start consumer
sudo systemctl start producer
sudo systemctl enable zookeeper
sudo systemctl enable kafka
sudo systemctl enable uart
sudo systemctl enable consumer
sudo systemctl enable producer


/home/user/kafka_2.13-2.8.0/bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic control
/home/user/kafka_2.13-2.8.0/bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic detect
/home/user/kafka_2.13-2.8.0/bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic log


sudo systemctl stop kafka
sudo systemctl stop zookeeper
sudo systemctl start zookeeper
sudo systemctl start kafka
echo "create topic "