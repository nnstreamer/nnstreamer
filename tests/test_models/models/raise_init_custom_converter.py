##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file    raise_init_custom_converter.py
# @brief   Python custom converter whose constructor raises
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>


##
# @brief  User-defined custom converter that cannot be created
class CustomConverter:
    ##
    # @brief  The constructor, which raises
    def __init__(self):
        raise RuntimeError('the converter refuses to be created')

    ##
    # @brief  Python callback: convert, which is never reached
    # @param  input_array  The input data
    # @return nothing
    def convert(self, input_array):
        return None
