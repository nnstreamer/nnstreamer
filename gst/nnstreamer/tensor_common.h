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
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __GST_TENSOR_COMMON_H__
#define __GST_TENSOR_COMMON_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <glib.h>
#include <stdint.h>
#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>

#include "tensor_typedef.h"
#include "nnstreamer_plugin_api.h"

#ifdef HAVE_ORC
#include <orc/orcfunctions.h>

#define nns_memcpy(d,s,n) do { \
    if ((n) > 100) orc_memcpy ((d), (s), (n)); \
    else memcpy ((d), (s), (n)); \
  } while (0)

#define nns_memset orc_memset
#else
#define nns_memcpy memcpy
#define nns_memset memset
#endif

G_BEGIN_DECLS

/**
 * @brief time synchronization options
 */
typedef enum
{
  SYNC_NOSYNC = 0,
  SYNC_SLOWEST = 1,
  SYNC_BASEPAD = 2,
  SYNC_END,
} tensor_time_sync_mode;

/**
 * @brief Tensor Merge/Mux sync data for baspad mode
 */
typedef struct _tensor_sync_basepad_data{
  guint sink_id;
  GstClockTime duration;
} tensor_sync_basepad_data;

/**
 * @brief Tensor Merge/Mux time sync data
 */
typedef struct _tensor_time_sync_data {
  tensor_time_sync_mode mode;
  gchar *option;
  union {
    tensor_sync_basepad_data data_basepad;
  };
} tensor_time_sync_data;

/**
 * @brief Internal data structure for Collect Pad in mux / merge
 */
typedef struct
{
  GstCollectData collect;
  GstClockTime pts_timestamp;
  GstClockTime dts_timestamp;
  GstBuffer *buffer;
  GstPad *pad;
} GstTensorCollectPadData;

/**
 * @brief A callback for typefind, trying to find whether a file is other/tensors or not.
 * For the concrete definition of headers, please look at the wiki page of nnstreamer:
 * https://github.com/nnsuite/nnstreamer/wiki/Design-External-Save-Format-for-other-tensor-and-other-tensors-Stream-for-TypeFind
 */
extern void gst_tensors_typefind_function (GstTypeFind * tf, gpointer pdata);

#define GST_TENSOR_TYPEFIND_REGISTER(plugin)  do { \
    gst_type_find_register (plugin, "other/tensorsave", \
        GST_RANK_PRIMARY, gst_tensors_typefind_function, "tnsr", \
        gst_caps_new_simple ("other/tensorsave", NULL, NULL), NULL, NULL); \
    } while (0)

#define NNSTREAMER_PLUGIN_INIT(name)	\
  gboolean G_PASTE(nnstreamer_export_, name) (GstPlugin * plugin)


/**
 * @brief A function call to decide current timestamp among collected pads based on PTS.
 * It will decide current timestamp according to sync option.
 * @return True / False if EOS, it return TRUE.
 * @param collect Collect pad.
 * @param current_time Current time
 * @param sync Synchronization Option (NOSYNC, SLOWEST, BASEPAD, END)
 */
extern gboolean gst_tensor_set_current_time(GstCollectPads *collect, GstClockTime *current_time, tensor_time_sync_data sync);

/**
 * @brief  A function call to make tensors from collected pads
 * It decide which buffer is going to be used according to sync option.
 * @return True / False if EOS, it return TRUE.
 * @param collect Collect pad.
 * @param sync Synchronization Option (NOSYNC, SLOWEST, BASEPAD, END)
 * @param current_time Current Timestamp
 * @param need_buffer Boolean for Update Collect Pads
 * @param tensors_buf Generated GstBuffer for Collected Buffer
 * @param configs Configuration Info for Collected Buffer
 */
extern gboolean gst_gen_tensors_from_collectpad (GstCollectPads * collect, tensor_time_sync_data sync, GstClockTime current_time, gboolean *need_buffer, GstBuffer *tensors_buf, GstTensorsConfig *configs);

G_END_DECLS
#endif /* __GST_TENSOR_COMMON_H__ */
