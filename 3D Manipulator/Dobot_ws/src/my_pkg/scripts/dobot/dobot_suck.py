#!/usr/bin/env python
#-- coding:UTF-8 --
# dobot机械臂专用模块pydobot


from serial.tools import list_ports
from pydobot import Dobot
import time

 
dobot = Dobot(port='/dev/ttyUSB0', verbose=True) # 和与port链接的dobot相连，verbose是否在终端打印命令

dobot.suck(True)
time.sleep(2)

dobot.suck(False)

dobot.close()


