#!/usr/bin/env python3
# -- coding:UTF-8 --

from pyfirmata import Arduino
import time

board = Arduino('/dev/ttyUSB1')


board.servo_config(9, 0, 180, 180)
board.servo_config(8, 0, 180, 0)
time.sleep(3)


board.servo_config(9, 0, 180, 0)
board.servo_config(8, 0, 180, 0)
time.sleep(5)
print('loose')

board.servo_config(9, 0, 180, 0)
board.servo_config(8, 0, 180, 180)
time.sleep(1)


board.servo_config(9, 0, 180, 0)
board.servo_config(8, 0, 180, 0)
time.sleep(1)
