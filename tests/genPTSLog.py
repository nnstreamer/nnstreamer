#!/usr/bin/env python

##
# @brief Write the generated data
#
def write(filename, data):
    with open(filename, 'wb') as file:
        file.write(data)

##
# @brief Save PTS log to "filename"
# @param[in] filename The filename to be saves as a PTS log file.
# @param[in] numBuffers number of buffer to log
# @param[in] framerate Framerate of video
# @param[in] offset PTS offset(ns) 1000000000 = 1sec
def genVideoPTS(filename, numBuffers, framerate, offset):
    offset = float(offset)/1000000000
    duration_sec = float(1) / framerate
    duration_min = 0
    duration_hour = 0

    if duration_sec >= 60:
        carry = int(duration_sec / 60)
        duration_min = duration_min + carry
        duration_sec = int(duration_sec % 60)

    if duration_min >= 60:
        carry = int(duration_min / 60)
        duration_hour = duration_hour + carry
        duration_min = int(duration_min % 60)

    pts_sec=offset
    pts_min=0
    pts_hour=0

    str=""

    for i in range(1, numBuffers+1):
        if pts_sec >= 60:
            carry = int(pts_sec / 60)
            pts_min = pts_min + carry
            pts_sec = int(pts_sec % 60)

        if pts_min >= 60:
            carry = int(pts_min / 60)
            pts_hour = pts_hour + carry
            pts_min = int(pts_min % 60)

        if pts_sec >= 10:
            str+="%d:%02d:%.9f\n"%(pts_hour, pts_min, pts_sec)
        else:
            str+="%d:%02d:0%.9f\n"%(pts_hour, pts_min, pts_sec)
        pts_sec = pts_sec + duration_sec
        pts_min = pts_min + duration_min
        pts_hour = pts_hour + duration_hour

    write(filename, str)
