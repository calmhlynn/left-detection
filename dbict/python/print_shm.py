import time
import sysv_ipc

def hex_dump(msg, buf):
    for value in buf: msg += " %02X" %value
    print(msg)

def bytes_to_hex(one, two ,three, four):

    detect = [format(one,"02X"), format(two,"02X"), format(three,"02X"), format(four,"02X")]

    return detect

print("start serial process ~")

shm = sysv_ipc.SharedMemory(0x1000)

while True:
    memory = bytearray(shm.read())

    #hex_dump("CAN[%x-%x] :" %(memory[0], memory[1]), memory[128+60:128+64])
   # print(bytes_to_int(memory[128+60:128+64]))
    # print(memory[128+60].print(bytes_to_hex(memory[128+60], memory[128+61], memory[128+62], memory[128+63]))hex())
    
#    memory[1] = (memory[1] + 1) & 0xff
#    shm.write(memory)
    time.sleep(0.5)