/*
 * NNStreamer Common Header
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * @file	tensor_common.h
 * @date	23 May 2018
 * @brief	Common header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 */

#ifndef __GST_TENSOR_COMMON_H__
#define __GST_TENSOR_COMMON_H__

#include <glib.h>
#include <stdint.h>
#include "tensor_typedef.h"

G_BEGIN_DECLS

/* @TODO I'm not sure if the range is to be 1, 65535 or larger */
#define GST_TENSOR_CAP_DEFAULT \
    "other/tensor, " \
    "rank = (int) [ 1, 4 ], " \
    "dim1 = (int) [ 1, 65535 ], " \
    "dim2 = (int) [ 1, 65535 ], " \
    "dim3 = (int) [ 1, 65535 ], " \
    "dim4 = (int) [ 1, 65535 ], " \
    "type = (string) { float32, float64, int32, uint32, int16, uint16, int8, uint8 }, " \
    "framerate = (fraction) [ 0/1, 2147483647/1 ]"

/**
 * @brief Possible input stream types for other/tensor.
 *
 * This is realted with media input stream to other/tensor.
 * There is no restrictions for the outputs.
 */
typedef enum _nns_media_type {
  _NNS_VIDEO = 0,
  _NNS_AUDIO, /* Not Supported Yet */
  _NNS_STRING, /* Not Supported Yet */

  _NNS_MEDIA_END,
} media_type;

/**
 * @brief Byte-per-element of each tensor element type.
 */
static const unsigned int tensor_element_size[] = {
        [_NNS_INT32] = 4,
        [_NNS_UINT32] = 4,
        [_NNS_INT16] = 2,
        [_NNS_UINT16] = 2,
        [_NNS_INT8] = 1,
        [_NNS_UINT8] = 1,
        [_NNS_FLOAT64] = 8,
        [_NNS_FLOAT32] = 4,
};

/**
 * @brief String representations for each tensor element type.
 */
extern const gchar* tensor_element_typename[];

/**
 * @brief Get tensor_type from string tensor_type input
 * @return Corresponding tensor_type. _NNS_END if unrecognized value is there.
 * @param typestr The string type name, supposed to be one of tensor_element_typename[]
 */
extern tensor_type get_tensor_type(const gchar* typestr);

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
extern int find_key_strv(const gchar **strv, const gchar *key);

/**
 * @brief Parse tensor dimension parameter string
 * @return The Rank.
 * @param param The parameter string in the format of d1:d2:d3:d4, d1:d2:d3, d1:d2, or d1, where dN is a positive integer and d1 is the innermost dimension; i.e., dim[d4][d3][d2][d1];
 */
extern int get_tensor_dimension(const gchar* param, uint32_t dim[NNS_TENSOR_RANK_LIMIT]);

/**
 * @brief Count the number of elemnts of a tensor
 * @return The number of elements. 0 if error.
 * @param dim The tensor dimension
 */
extern size_t get_tensor_element_count(uint32_t dim[NNS_TENSOR_RANK_LIMIT]);

#define str(s) xstr(s)
#define xstr(s) #s

#include <glib/gprintf.h>
#ifdef TIZEN
#include <dlog.h>
#else
#define dlog_print(loglevel, component, ...) \
  do { \
    g_message(__VA_ARGS__); \
  } while (0)
#endif
#define debug_print(cond, ...)	\
  do { \
    if ((cond) == TRUE) { \
      dlog_print(DLOG_DEBUG, "nnstreamer", __FILE__ ":" str(__LINE__) " "  __VA_ARGS__); \
    } \
  } while (0)

#define err_print(...) dlog_print(DLOG_ERROR, "nnstreamer", __VA_ARGS__)

G_END_DECLS

#endif /* __GST_TENSOR_COMMON_H__ */
