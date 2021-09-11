from pyfirmata import Arduino
import time


class sucker_control():
    def __init__(self):
        self.board = Arduino('/dev/ttyUSB1')

    def suck(self):
        self.board.digital[8].write(0)
        self.board.digital[9].write(1)

    def inflate(self):
        self.board.digital[8].write(1)
        self.board.digital[9].write(0)
        time.sleep(1)
        self.board.digital[8].write(0)
        self.board.digital[9].write(0)

    def loose(self):
        self.board.digital[8].write(0)
        self.board.digital[9].write(0)

        self.board.digital[8].write(1)
        self.board.digital[9].write(0)
        time.sleep(2)
        self.board.digital[8].write(0)
        self.board.digital[9].write(0)


if __name__ == '__main__':
    sucker = sucker_control()
    # sucker.inflate()
    # time.sleep(1)
    # sucker.suck()
    # print('*****')
    # time.sleep(5)
    sucker.loose()
