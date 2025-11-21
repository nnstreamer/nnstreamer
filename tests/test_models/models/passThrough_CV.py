#!/bin/env python3
##
# SPDX-License-Identifier: LGPL-2.1-only
#
# @file         passThrough_CV.py
# @brief        A trivial python custom filter to test the usage of CV2 in python script, which tests multithreading in a script as well.
# @author       beto-gulliver @ github

import numpy as np
import nnstreamer_python as nns
import threading
import time

D1 = 3
D2 = 280
D3 = 40
D4 = 1

USE_CV2 = True # Fixed: Thread-safe cv2 usage with proper error handling
#USE_CV2 = False # [OK]
if USE_CV2 : 
    import cv2
    # Thread-safe cv2 initialization
    cv2_lock = threading.Lock()

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
        if USE_CV2 :
            try:
                # Thread-safe cv2 operation with proper error handling
                with cv2_lock:
                    # Create a simple test image and save it
                    test_image = np.zeros((240, 320, 3), dtype=np.uint8)
                    # Use a unique filename to avoid conflicts
                    filename = f"/tmp/x_{threading.get_ident()}_{int(time.time() * 1000)}.png"
                    success = cv2.imwrite(filename, test_image)
                    if not success:
                        print(f"Warning: Failed to write image to {filename}")
            except Exception as e:
                print(f"Warning: cv2 operation failed: {e}")
                # Continue without cv2 operation to avoid breaking the pipeline
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
