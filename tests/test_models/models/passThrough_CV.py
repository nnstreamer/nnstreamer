#!/bin/env python3
##
# SPDX-License-Identifier: LGPL-2.1-only
#
# @file         passThrough_CV.py
# @brief        A trivial python custom filter to test the usage of CV2 in python script, which tests multithreading in a script as well.
# @author       beto-gulliver @ github

import numpy as np
import nnstreamer_python as nns

D1 = 3
D2 = 280
D3 = 40
D4 = 1

USE_CV2 = True # [NG] XXX: 'cv2' halts!!! -->  github issue (threads' race condition/deadlock? release GIL? ???)
#USE_CV2 = False # [OK]
if USE_CV2 : import cv2

class CustomFilter(object):
    def __init__(self, *args):
        self.input_dims  = [nns.TensorShape([D1, D2, D3, D4], np.uint8)]
        self.output_dims = [nns.TensorShape([D1, D2, D3, D4], np.uint8)]

    def getInputDim(self):
        return self.input_dims

    def getOutputDim(self):
        return self.output_dims

    def invoke(self, input_array):
        # passthrough, just return.
        print(f"--- USE_CV2:{USE_CV2}", __file__, input_array[0].shape)
        if USE_CV2 : cv2.imwrite("/tmp/x.png", np.zeros((240, 320, 3), dtype=np.uint8)) # [NG] XXX: 'cv2' halts!!! -->  github issue (threads' race condition/deadlock? release GIL? ???)
        return input_array

# ----------------------------------------------------------------------
def main():
    import sys
    cf = CustomFilter()
    print(cf)
    shape, dtype = [D1, D2, D3, D4], "uint8"
    if 1 :
        input_dims = [nns.TensorShape(shape, dtype)]
#        cf.setInputDim(input_dims) # callback
    for idx in range(10) :
        in_ = np.ones(shape).astype(np.uint8)
        in_ = (np.random.random(shape) * 255).astype(np.uint8)
        in_ = [np.ravel(in_)] # as buffer (memory)
        print(idx, f"USE_CV2:{USE_CV2} in_ {in_}", __file__)
        out = cf.invoke(in_)
        print(idx, f"USE_CV2:{USE_CV2} out {out}", __file__)

# voila!
if __name__ == '__main__':
    main()
# ----------------------------------------------------------------------
