#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file generateTest.py
# @brief Generate golden test results for test cases
# @author Jijoong Moon <jijoong.moon@samsung.com>

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from struct import pack
import random

import numpy as np

def saveTestData(filename, width, height, channel, batch):
    string = b''
    data = []

    for b in range(0,batch):
        for h in range(0,height):
            for w in range(0,width):
                for c in range(0,channel):
                    n = random.uniform(0.0, 10.0)
                    string += pack('f', n)
                    data.append(n)

    with open(filename,'wb') as file:
        file.write(string)
    file.close()

    return data

#merge with channel direction
ch = [3, 2, 4]
width = 100
height= 50
batch= 1

buf=[]

buf.append(saveTestData("channel_00.dat", width, height, 3, batch))
buf.append(saveTestData("channel_01.dat", width, height, 2, batch))
buf.append(saveTestData("channel_02.dat", width, height, 4, batch))

out = b''
for b in range(0, batch):
    for h in range(0,height):
        for w in range(0,width):
            for n in range(0,3):
                for c in range(0,ch[n]):
                    out += pack('f',buf[n][b*height*width*ch[n]+h*width*ch[n] + w * ch[n] + c])

with open("channel.golden", 'wb') as file:
    file.write(out)

#merge with width direction
width = [100, 200, 300]
ch = 3
height= 50
batch= 1

buf=[]

buf.append(saveTestData("width_100.dat", width[0], height, ch, batch))
buf.append(saveTestData("width_200.dat", width[1], height, ch, batch))
buf.append(saveTestData("width_300.dat", width[2], height, ch, batch))

out = b''

for b in range(0, batch):
    for h in range(0,height):
        for n in range(0,3):
            for w in range(0,width[n]):
                for c in range(0,ch):
                    out += pack('f',buf[n][b*height*width[n]*ch + h*width[n]*ch + w * ch + c])

with open("width.golden", 'wb') as file:
    file.write(out)

#merge with width direction
batch = [1, 2, 3]
ch = 3
height= 50
width= 100

buf=[]

buf.append(saveTestData("batch_1.dat", width, height, ch, batch[0]))
buf.append(saveTestData("batch_2.dat", width, height, ch, batch[1]))
buf.append(saveTestData("batch_3.dat", width, height, ch, batch[2]))

out = b''
for n in range(0,3):
    for b in range(0, batch[n]):
        for h in range(0,height):
            for w in range(0,width):
                for c in range(0,ch):
                    out += pack('f',buf[n][b*height*width*ch + h*width*ch + w * ch + c])

with open("batch.golden", 'wb') as file:
    file.write(out)
