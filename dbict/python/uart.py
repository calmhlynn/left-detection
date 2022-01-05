import serial
import time
import sysv_ipc
import sys
import datetime
import os


class CAN_Serial:

    def __init__(self, shm, ser, inactive, chk, ser_fail, err, lmb_active):

        self.shm = shm
        self.ser = ser
        self.inactive = inactive
        self.chk = chk
        self.ser_fail = ser_fail
        self.err = err
        self.lmb_active = lmb_active

    def hex_dump(self, msg, buf):
        for value in buf: msg += " %02X" % value
        print(msg)

    def genlrc(self, msg):
        lrc = 0
        for value in msg: lrc ^= value
        return lrc

    def receive_uart_msg(self):
        rxcnt = self.ser.inWaiting()
        if rxcnt < 3: return None
        rxmsg = self.ser.read(3)

        if rxmsg[0] != 0x7e or rxmsg[1] != 0x7e:
            time.sleep(0.01)
            rxcnt = self.ser.inWaiting()
            rxmsg = self.ser.read(rxcnt)
            return None

        if rxcnt < 71:
            time.sleep(0.01)
            rxcnt += self.ser.inWaiting()

        rxmsg += self.ser.read(rxcnt - 3)
        return rxmsg

    def program_chk(self, memory, next_cnt):

        code = (memory[1] & 0x0f) | (memory[16] << 1)  # detector state

        prev_cnt = int(memory[0])

        if memory[0] != 0:  # active-check detectnet
            self.inactive = 0
            self.err = code
        else:  # inactive detectnet
            self.inactive += 1
            if self.inactive >= 50:
                self.inactive = 99
                code = 0x02
                self.err = code
                return self.err

        if prev_cnt == next_cnt:  # error check when detectnet restart
            self.chk += 1
            if self.chk > 100:
                code = 0x02
                self.err = code
                return self.err
            else:
                self.err = code
        else:
            self.chk = 0
            self.err = code

        return self.err

    def write_shm(self, memory):
        length = memory[42]
        frame_num = 0
        if length > 0:  # server command
            length += 2
            sndmsg = memory[40:40 + length]

            memory[42] = 0  # clear length
            self.shm.write(memory)
        else:
            frame_num = (frame_num + 1) & 0x3f

            sndmsg = [0x7e, 0x7e, 6, 1, 0x00, frame_num, chkcode]
            sndmsg.append(uart.genlrc(sndmsg))

        # send to UART (CAN_Controller)
        self.ser.write(sndmsg)
        # uart.hex_dump("TXD : ", sndmsg)

    def chk_can_err(self, rcvmsg):

        can_err_dir = os.path.isdir("/home/user/jetson-inference/dbict/control/canerr")
        lmb_err_dir = os.path.isdir("/home/user/jetson-inference/dbict/control/lmberr")

        if rcvmsg:

            self.ser_fail = 0
            rxcnt = len(rcvmsg)

            # uart.hex_dump("RXD : ", rcvmsg)

            if rcvmsg[64] != 0:

                # print("ON")

                self.lmb_active = True
                if lmb_err_dir:
                    os.rmdir("/home/user/jetson-inference/dbict/control/lmberr")
                else:
                    pass

            else:

                # print("OFF")

                self.lmb_active = False
                if lmb_err_dir == False:
                    os.mkdir("/home/user/jetson-inference/dbict/control/lmberr")
                else:
                    pass

            if can_err_dir:
                os.rmdir("/home/user/jetson-inference/dbict/control/canerr")
            else:
                pass

            if rxcnt >= 71 and rcvmsg[6] == 1 and rcvmsg[6 + 8] == 2:
                memory[128:128 + 64] = rcvmsg[6:6 + 64]
            else:
                memory[80:80 + rxcnt] = rcvmsg

            shm.write(memory)
        else:
            self.ser_fail += 1
            # print("fail : " , self.ser_fail)
            if self.ser_fail > 100:
                self.errcnt = 99
                self.ser_fail = 999
                if can_err_dir == False:
                    os.mkdir("/home/user/jetson-inference/dbict/control/canerr")
                else:
                    pass
                if lmb_err_dir == False:
                    os.mkdir("/home/user/jetson-inference/dbict/control/lmberr")
                else:
                    pass


inactive = 0
next_cnt = 0
chk = 0
ser_fail = 0
err = False
lmb_active = False
ser = serial.Serial("/dev/ttyTHS1", baudrate=115200, timeout=1)
if ser.isOpen() == False:
    ser.open()

shm = sysv_ipc.SharedMemory(0x1000, flags=sysv_ipc.IPC_CREAT, size=256)

uart = CAN_Serial(shm, ser, inactive, chk, ser_fail, err, lmb_active)

while True:
    try:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

        memory = bytearray(shm.read())

        chkcode = uart.program_chk(memory, next_cnt)  ## Error check
        next_cnt = int(memory[0])  ## Error check

        uart.write_shm(memory)  ## Write Shared Memory

        rcvmsg = uart.receive_uart_msg()
        uart.chk_can_err(rcvmsg)

        time.sleep(0.1)


    except Exception as e:
        print("Except : ", e)
        time.sleep(10)
        sys.exit()
