import os
import sys
import time
import traceback

from typing import List

from serial import Serial
from sysv_ipc import SharedMemory, IPC_CREAT

can_dir = "/home/user/jetson-inference/dbict/control/canerr"
lmb_dir = "/home/user/jetson-inference/dbict/control/lmberr"
can_err_dir = os.path.isdir(can_dir)
lmb_err_dir = os.path.isdir(lmb_dir)

class ShdMemory:
    def __init__(self, shared_memory: SharedMemory):
        self.shm = shared_memory
        self.shm_memory = bytearray(self.shm.read())
        self.next_count = 0
        self.detection_code = 0
        self.detection_start = False
        self.exiting = 0

        self.shm.write(bytes(64))
        """초기 공유 메모리 값 초기화"""

    def read(self):
        self.shm_memory = bytearray(self.shm.read())

    def write(self, shm_memory):
        self.shm.write(shm_memory)

    def get_detect_code(self):
        self.detection_code = (self.shm_memory[1] & 0x0f) | (self.shm_memory[16] << 1)  # detector state

    def detection_check(self):
        if self.shm_memory[0] != self.next_count:
            """
            공유 메모리 0번지의 영상처리 프레임을 체크 해 영상처리 프로그램 실행여부 체크
            0번지(shm_memory[0])와 next_count의 값이 다르면 '프로그램 실행 중'.
            """
            self.detection_start = True
            self.exiting = 0
        else:
            if not self.detection_start:
                """시스템 부팅시 이 if문으로 진입"""
                self.detection_code = 0x02

            self.exiting += 1
            if self.exiting > 100 and self.detection_start:
                """
                예기치 않게 프로그램이 종료되었을 시 이 if문으로 진입.
                공유 메모리 값 초기화, 그리고 detection_start를 False로 초기화
                """
                self.shm.write(bytes(64))
                self.detection_start = False


class UartSerial:

    def __init__(self, serial: Serial):

        self.ser = serial
        if not self.ser.isOpen():
            self.ser.open()

        self.detection_code = 0
        self.detection_start = False

    def receive_uart_msg_check(self):
        """
        1. rx data의 길이를 체크
        2. rx data의 0, 1번지의 값이 '0x7E'로 시작하는 지 체크
        3. rx data의 길이가 71 이하인지 체크
        """

        rx_length = self.ser.inWaiting()

        if rx_length < 3:
            return None
        rx_message = self.ser.read(3)

        if rx_message[0] != 0x7e or rx_message[1] != 0x7e:
            time.sleep(0.01)
            rx_length = self.ser.inWaiting()
            rx_message = self.ser.read(rx_length)
            return None

        if rx_length < 71:
            time.sleep(0.01)
            rx_length += self.ser.inWaiting()

        rx_message += self.ser.read(rx_length - 3)
        return rx_message


def hex_dump(message: str, buf: bytearray):
    """
    bytearray 타입의 통신 데이터를 hex로 변환 후 결과를 출력한다.
    ============================================================
    :param message: 출력할 hex타입의 데이터
    :param buf: 송수신되고 있는 bytearray 타입의 데이터
    :return None
    ============================================================
    """

    for value in buf:
        message += " %02X" % value
    print(message)


def gen_lrc(msg: List):
    """
    Longitudinal Redundancy Check(LRC); 세로 중복 검사
    참고: https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=deepb1ue&logNo=221189953315
    ============================================================
    :param msg: can 컨트롤러에 write하기 전 오류 검출 코드를 검사할 데이터
    :return: LRC Code
    ============================================================
    """

    lrc = 0
    for value in msg:
        lrc ^= value
    return lrc


def rx_data_check(shm: ShdMemory, uart: UartSerial, can_conn_fail_count=0):
    rx_msg = uart.receive_uart_msg_check()

    if rx_msg:
        can_conn_fail_count = 0
        os.rmdir(lmb_dir) if rx_msg[64] != 0 else os.mkdir(lmb_dir)
        """ rx_msg의 64번지의 값을 체크하여 lmb의 데이터를 수신하는지 체크한다."""

        if can_err_dir: os.rmdir(can_dir)

        finally_shm_write(shm, rx_msg)
    else:
        can_conn_fail_count += 1
        if can_conn_fail_count == 100:
            can_conn_fail_count = 999
            if not can_err_dir: os.mkdir(can_dir)
            if not lmb_err_dir: os.mkdir(lmb_dir)


def writing(shm: ShdMemory, uart: UartSerial):
    length = shm.shm_memory[42]
    frame_num = 0
    if length > 0:
        length += 2
        serial_msg = shm.shm_memory[40:40 + length]
        shm.shm_memory[42] = 0
        shm.shm.write(shm.shm_memory)
    else:
        frame_num = (frame_num + 1) & 0x3f
        serial_msg = [0x7e, 0x7e, 6, 1, 0x00, frame_num, shm.detection_code]
        serial_msg.append(gen_lrc(serial_msg))

    uart.ser.write(serial_msg)
    # hex_dump("TXD : ", serial_msg)


def finally_shm_write(shm, rx_msg):
    rx_length = len(rx_msg)
    if rx_length >= 71 and rx_msg[6] == 1 and rx_msg[6 + 8] == 2:
        shm.shm_memory[128:128 + 64] = rx_msg[6:6 + 64]
    else:
        shm.shm_memory[80:80 + rx_length] = rx_msg

    shm.write(shm.shm_memory)


def main():
    shm = ShdMemory(SharedMemory(0x1000, flags=IPC_CREAT, size=256))
    uart = UartSerial(Serial("/dev/ttyTHS1", baudrate=115200, timeout=1))
    while True:
        try:
            """
            1. shm.read: 공유 메모리 읽기
            2. shm.get_detect_code: 공유 메모리에 있는 프레임과 검지코드 확인
            3. shm.detection_check: 공유 메모리의 프레임을 확인해 프로그램이 실행이 되었는지 확인
            4. shm.next_count: 영상처리 실행여부 체크를 위한 프레임 저장
            5. rx_data_check: uart의 수신 데이터를 잘 받는지 체크
            6. writing: uart의 수신 데이터와 공유 메모리의 결과를 최종적으로 제어기에 write
            """
            shm.read()
            shm.get_detect_code()
            shm.detection_check()
            shm.next_count = shm.shm_memory[0]
            rx_data_check(shm, uart)
            writing(shm, uart)

            time.sleep(0.1)

        except Exception as e:
            """error line debug."""
            print(traceback.format_exc())
            time.sleep(10)
            sys.exit()


if __name__ == '__main__':
    main()
