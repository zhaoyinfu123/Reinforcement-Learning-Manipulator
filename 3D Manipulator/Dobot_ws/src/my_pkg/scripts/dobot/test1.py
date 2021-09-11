from pyfirmata import Arduino

class arduino_control():
    def __init__(self):
        self.board = Arduino('/dev/ttyUSB1')

    def suck(self):
        self.board.servo_config(9, 0, 180, 180)
        self.board.servo_config(8, 0, 180, 0)
        time.sleep(2)

    def hold(self):
        self.board.servo_config(9, 0, 180, 0)
        self.board.servo_config(8, 0, 180, 0)

    def loose(self):
        self.board.servo_config(9, 0, 180, 0)
        self.board.servo_config(8, 0, 180, 180)

