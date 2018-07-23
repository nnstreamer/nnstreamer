/**
 * NNStreamer Common Header
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 */
/**
 * @file	tensor_common.h
 * @date	23 May 2018
 * @brief	Common header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __GST_TENSOR_COMMON_H__
#define __GST_TENSOR_COMMON_H__

#include <glib.h>
#include <stdint.h>
#include "tensor_typedef.h"
#include <gst/gst.h>

G_BEGIN_DECLS

/** @todo I'm not sure if the range is to be 1, 65535 or larger */
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
 * @brief Default static capibility for other/tensors
 *
 * This type uses GstMetaTensor to describe tensor. So there is no need to ask information
 * to identify each tensor.
 *
 */
#define GST_TENSORS_CAP_DEFAULT \
    "other/tensors, " \
    "rank = (int) [ 1, 4 ], " \
    "num_tensors = (int) [1, 65535], "\
    "framerate = (fraction) [ 0/1, 2147483647/1 ]"
    /**
     * type should be one of
     * { float32, float64, int32, uint32, int16, uint16, int8, uint8 }
     * "types = (string) uint8, uint8, uint8"
     * Dimensions of Tensors for negotiation. It's comment out here,
       but when we call gst_structure_get_string, it actually is working well
     * "dimensions = (string) dim1:dim2:dim3:dim4, dim1:dim2:dim3:dim4"
     */

/**
 * @brief Possible input stream types for other/tensor.
 *
 * This is realted with media input stream to other/tensor.
 * There is no restrictions for the outputs.
 */
typedef enum _nns_media_type {
  _NNS_VIDEO = 0, /**< supposedly video/x-raw */
  _NNS_AUDIO, /**< Not Supported Yet */
  _NNS_STRING, /**< Not Supported Yet */

  _NNS_MEDIA_END, /**< End Marker */
} media_type;

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
extern int get_tensor_dimension(const gchar* param, tensor_dim dim);

/**
 * @brief Count the number of elemnts of a tensor
 * @return The number of elements. 0 if error.
 * @param dim The tensor dimension
 */
extern size_t get_tensor_element_count(const tensor_dim dim);

/**
 * @brief Read pad-cap, return corresponding tensor-dim/type.
 * @return Notifies which part of dim/type is determined.
 * @param[in] caps the pad-cap to be interpreted.
 * @param[out] dim the corresponging dimension info from the cap.
 * @param[out] type the corresponding element type from the cap.
 * @param[out] framerate_num Numerator of framerate. Set null to not use this.
 * @param[out] framerate_denum Denumerator of framerate. Set null to not use this.
 */
extern GstTensor_Filter_CheckStatus
get_tensor_from_padcap(const GstCaps * caps, tensor_dim dim, tensor_type *type,
    int *framerate_num, int *framerate_denum);

/**
 * @brief Make str(xyz) ==> "xyz" with macro expansion
 */
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

/**
 * @brief Debug message print. In Tizen, it uses dlog; otherwise,m it uses g_message().
 */
#define debug_print(cond, ...)	\
  do { \
    if ((cond) == TRUE) { \
      dlog_print(DLOG_DEBUG, "nnstreamer", __FILE__ ":" str(__LINE__) " "  __VA_ARGS__); \
    } \
  } while (0)

/**
 * @brief Error message print. In Tizen, it uses dlog; otherwise,m it uses g_message().
 */
#define err_print(...) dlog_print(DLOG_ERROR, "nnstreamer", __VA_ARGS__)

/**
 * @brief A callback for typefind, trying to find whether a file is other/tensors or not.
 * For the concrete definition of headers, please look at the wiki page of nnstreamer:
 * https://github.com/nnsuite/nnstreamer/wiki/Design-External-Save-Format-for-other-tensor-and-other-tensors-Stream-for-TypeFind
 */
extern void gst_tensors_typefind_function (GstTypeFind *tf, gpointer pdata);

#define GST_TENSOR_TYPEFIND_REGISTER(plugin)  do { \
    gst_type_find_register (plugin, "other/tensorsave", \
        GST_RANK_PRIMARY, gst_tensors_typefind_function, "tnsr", \
        gst_caps_new_simple ("other/tensorsave", NULL, NULL), NULL, NULL)); \
    } while (0)

G_END_DECLS

#endif /* __GST_TENSOR_COMMON_H__ */
