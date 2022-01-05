import time
import sysv_ipc
from socket import *
import pickle
import sys

# tcp/ip server
IP_ADDRESS     = '172.16.0.201'
PORT_NUM       = 7070
server_address = (IP_ADDRESS, PORT_NUM)
BUFFER_SIZE    = 1024

#    msg = stx  / stx / len / addr / op / data[n]) /lrc
sndmsg   = [0x7e, 0x7e, 4+65, 3, 0x15] + ([0] * 66)

def hex_dump(msg, buf):
    for value in buf: msg += " %02X" %value
    print(msg)

def genlrc (msg):
    lrc = 0
    for value in msg: lrc ^= value
    return lrc

def receive_packet(conn):

    rcvmsg = conn.recv(BUFFER_SIZE)
    if len(rcvmsg) < 3: return None
    if rcvmsg[0] != 0x7e or rcvmsg[1] != 0x7e: return None

    repeat = 0
    rxcnt = rcvmsg[2] + 2
    while len(rcvmsg) < rxcnt:
        time.sleep(0.2)
        rcvmsg += conn.recv(BUFFER_SIZE)
        repeat += 1
        print("repeat = ", repeat)
        if repeat > 10: return None

    if genlrc(rcvmsg[:rxcnt]) != 0: return None

    return rcvmsg

if __name__ == "__main__":

    frame_num     = 0
    clientConnect = 0
    comm_errcnt   = 0
    shm = sysv_ipc.SharedMemory(0x1000, flags = sysv_ipc.IPC_CREAT, size = 256)

    print("start serial/TCP process ~")

    while True:
        try:
            if clientConnect == 0:
                client = socket(AF_INET, SOCK_STREAM)
                client.settimeout(10)
                client.connect(server_address)        # 서버와의 연결을 시도
                clientConnect = 1
                print(time.strftime('%X', time.localtime()), "server connected =", server_address)

            client.settimeout(10)

            rcvmsg = bytearray(receive_packet(client))       ####### Receive From HOST
            if rcvmsg == None:
                comm_errcnt += 1
                if comm_errcnt > 6:
                    print("None host command")
                    client.close()
                    clientConnect = 0
                    comm_errcnt   = 0
                continue

            #hex_dump("RXD[%s] :" %time.strftime('%X', time.localtime()), rcvmsg)
            comm_errcnt = 0
            memory = bytearray(shm.read())

            if rcvmsg[4] != 0x12 and rcvmsg[4] != 0x42:               # special server_command
                length = rcvmsg[2] + 2
                if length > 40:
                    length = 40
                    rcvmsg[2]  = 38
                    rcvmsg[39] = genlrc(rcvmsg[0:38])
                memory[40:40+length] = rcvmsg[0:length]
             #   hex_dump("host cmd:", rcvmsg)

            if memory[82] != 0:                 # response server_command
                length = memory[82] + 2
                sndmsg = memory[80:80+length]
                memory[82] = 0
            else:
                sndmsg = [0x7e, 0x7e, 4+65, rcvmsg[3], 0x15, frame_num]
                sndmsg += memory[128:128+64]
                sndmsg.append(genlrc(sndmsg[0:70]))
                frame_num = (frame_num + 1) & 0x3f

            client.send(bytearray(sndmsg))
            shm.write(memory)
           # hex_dump("TXD[%s] :" %time.strftime('%X', time.localtime()), sndmsg[3:-1])

        except Exception as e:
            print("socket exception ->", e)
            client.close()
            clientConnect = 0
            time.sleep(5)
            sys.exit()