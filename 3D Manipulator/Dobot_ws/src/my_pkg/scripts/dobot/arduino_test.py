#!/usr/bin/env python3
# -- coding:UTF-8 --

from pyfirmata import Arduino
import time

board = Arduino('/dev/ttyUSB0')


while True:
    board.digital[13].write(0)
    board.pass_time(1)
    board.digital[13].write(1)
    board.pass_time(1)
