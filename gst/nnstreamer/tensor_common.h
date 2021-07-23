/**
 * NNStreamer Common Header
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
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
 * @see		https://github.com/nnstreamer/nnstreamer
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
#include "nnstreamer_log.h"
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
 * @see https://github.com/nnstreamer/nnstreamer/wiki/Synchronization-Policies-at-Mux-and-Merge
 */
typedef enum
{
  SYNC_NOSYNC = 0,
  SYNC_SLOWEST = 1,
  SYNC_BASEPAD = 2,
  SYNC_REFRESH = 3,
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
  GstBuffer *buffer;
  GstPad *pad;
} GstTensorCollectPadData;

/**
 * @brief Internal data structure for pad in demux / split
 */
typedef struct
{
  GstPad *pad;
  GstClockTime last_ts;
  GstFlowReturn last_ret;
  gint nth;
} GstTensorPad;

/**
 * @brief Get the corresponding mode from the string value.
 * @param[in] str The string value for the mode.
 * @return Corresponding mode for the string. SYNC_END for errors.
 */
extern tensor_time_sync_mode
gst_tensor_time_sync_get_mode (const gchar * str);

/**
 * @brief Get the time-sync mode string.
 * @return Corresponding mode string.
 */
extern const gchar *
gst_tensor_time_sync_get_mode_string (tensor_time_sync_mode mode);

/**
 * @brief Setup time sync option.
 * @param[in/out] filter "this" pointer. Sync mode & option MUST BE set already.
 * @return True if successfully set the option.
 */
extern gboolean
gst_tensor_time_sync_set_option_data (tensor_time_sync_data * sync);

/**
 * @brief A function call to decide current timestamp among collected pads based on PTS.
 * It will decide current timestamp according to sync option.
 * @return True / False. If EOS, it return TRUE.
 * @param collect Collect pad.
 * @param sync Synchronization Option (NOSYNC, SLOWEST, BASEPAD, END)
 * @param current_time Current time
 */
extern gboolean
gst_tensor_time_sync_get_current_time (GstCollectPads * collect, tensor_time_sync_data * sync, GstClockTime * current_time);

/**
 * @brief A function to be called while processing a flushing event.
 * It should clear old buffer and reset pad data.
 * @param collect Collect pad.
 */
extern void
gst_tensor_time_sync_flush (GstCollectPads * collect);

/**
 * @brief  A function call to make tensors from collected pads
 * It decide which buffer is going to be used according to sync option.
 * @return True to push buffer.
 * @param collect Collect pad.
 * @param sync Synchronization Option (NOSYNC, SLOWEST, BASEPAD, END)
 * @param current_time Current Timestamp
 * @param tensors_buf Generated GstBuffer for Collected Buffer
 * @param configs Configuration Info for Collected Buffer
 * @param is_eos True when EOS (end-of-stream)
 */
extern gboolean
gst_tensor_time_sync_buffer_from_collectpad (GstCollectPads * collect, tensor_time_sync_data * sync, GstClockTime current_time, GstBuffer * tensors_buf, GstTensorsConfig * configs, gboolean * is_eos);

/**
 * @brief Get pad caps from tensors config and caps of the peer connected to the pad.
 * @param pad GstPad to get possible caps
 * @param config tensors config structure
 * @return caps for given config. Caller is responsible for unreffing the returned caps.
 * @note This function is included in nnstreamer internal header for native APIs.
 *       When changing the declaration, you should update the internal header (nnstreamer_internal.h).
 */
extern GstCaps *
gst_tensor_pad_caps_from_config (GstPad * pad, const GstTensorsConfig * config);

/**
 * @brief Get all possible caps from tensors config. Unlike gst_tensor_pad_caps_from_config(), this function does not check peer caps.
 * @param pad GstPad to get possible caps
 * @param config tensors config structure
 * @return caps for given config. Caller is responsible for unreffing the returned caps.
 * @note This function is included in nnstreamer internal header for native APIs.
 *       When changing the declaration, you should update the internal header (nnstreamer_internal.h).
 */
extern GstCaps *
gst_tensor_pad_possible_caps_from_config (GstPad * pad, const GstTensorsConfig * config);

/**
 * @brief Check current pad caps is flexible tensor.
 * @param pad GstPad to check current caps
 * @return TRUE if pad has flexible tensor caps.
 * @note This function is included in nnstreamer internal header for native APIs.
 *       When changing the declaration, you should update the internal header (nnstreamer_internal.h).
 */
extern gboolean
gst_tensor_pad_caps_is_flexible (GstPad * pad);

/******************************************************
 ************ Commonly used debugging macros **********
 ******************************************************
 */
/**
 * @brief Macro for debug message.
 */
#define silent_debug(self, ...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

/**
 * @brief Macro for capability debug message.
 */
#define silent_debug_caps(self, caps, msg) do { \
  if (DBG) { \
    if (caps) { \
      GstStructure *caps_s; \
      gchar *caps_s_string; \
      guint caps_size, caps_idx; \
      caps_size = gst_caps_get_size (caps);\
      for (caps_idx = 0; caps_idx < caps_size; caps_idx++) { \
        caps_s = gst_caps_get_structure (caps, caps_idx); \
        caps_s_string = gst_structure_to_string (caps_s); \
        GST_DEBUG_OBJECT (self, msg " = %s\n", caps_s_string); \
        g_free (caps_s_string); \
      } \
    } \
  } \
} while (0)

G_END_DECLS
#endif /* __GST_TENSOR_COMMON_H__ */
