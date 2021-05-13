/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Pipeline from/to PBTxt Converter Parser
 * Copyright (C) 2021 MyungJoo Ham <myungjoo.ham@samsung.com>
 * Copyright (C) 2021 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    convert.h
 * @date    13 May 2021
 * @brief   GStreamer pipeline from/to pbtxt converter
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __GST_PARSE_CONVERT_H__
#define __GST_PARSE_CONVERT_H__

#include "types.h"

void convert_to_pbtxt (_Element *pipeline);

#endif
