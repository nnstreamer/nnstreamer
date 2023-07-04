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
#include <gst/base/gstadapter.h>
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
  guint nth;
} GstTensorPad;

/**
 * @brief Parses a configuration file and sets the corresponding properties on a GObject.
 *
 * This function reads the contents of the configuration file located at the given path
 * and sets the properties of the specified GObject based on the configuration data.
 *
 * @param config_path The path to the configuration file.
 * @param object      The GObject on which to set the properties.
 *
 * @note The responsibility of managing the memory of the GObject passed as a parameter
 *       lies outside this function.
 */
extern void 
gst_tensor_parse_config_file (const gchar *config_path, const GObject *object);

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
 * GstMeta is also copied with same sync mode.
 * @return True / False. If EOS, it return TRUE.
 * @param collect Collect pad.
 * @param sync Synchronization Option (NOSYNC, SLOWEST, BASEPAD, END)
 * @param current_time Current time
 * @param tensors_buf Generated GstBuffer for Collected Buffer
 */
extern gboolean
gst_tensor_time_sync_get_current_time (GstCollectPads * collect, tensor_time_sync_data * sync, GstClockTime * current_time, GstBuffer * tensors_buf);

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
 * @brief Configure gst-buffer with tensors information.
 * NNStreamer handles single memory chunk as single tensor.
 * If incoming buffer has invalid memories, separate it and generate new gst-buffer using tensors information.
 * Note that this function always takes the ownership of input buffer.
 * @param in input buffer
 * @param config tensors config structure
 * @return Newly allocated buffer. Null if failed. Caller should unref the buffer using gst_buffer_unref().
 */
extern GstBuffer *
gst_tensor_buffer_from_config (GstBuffer * in, GstTensorsConfig * config);

/**
 * @brief Get pad caps from tensors config and caps of the peer connected to the pad.
 * @param pad GstPad to get possible caps
 * @param config tensors config structure
 * @return caps for given config. Caller is responsible for unreffing the returned caps.
 */
extern GstCaps *
gst_tensor_pad_caps_from_config (GstPad * pad, const GstTensorsConfig * config);

/**
 * @brief Get all possible caps from tensors config. Unlike gst_tensor_pad_caps_from_config(), this function does not check peer caps.
 * @param pad GstPad to get possible caps
 * @param config tensors config structure
 * @return caps for given config. Caller is responsible for unreffing the returned caps.
 */
extern GstCaps *
gst_tensor_pad_possible_caps_from_config (GstPad * pad, const GstTensorsConfig * config);

/**
  * @brief Get tensor format of current pad caps.
  * @param pad GstPad to check current caps.
  * @return The tensor_format of current pad caps.
  *
  * If pad does not have tensor caps return _NNS_TENSOR_FORMAT_END
  */
extern tensor_format
gst_tensor_pad_get_format (GstPad *pad);

/**
 * @brief Macro to check current pad caps is static tensor.
 */
#define gst_tensor_pad_caps_is_static(p) (gst_tensor_pad_get_format (p) == _NNS_TENSOR_FORMAT_STATIC)

/**
 * @brief Macro to check current pad caps is flexible tensor.
 */
#define gst_tensor_pad_caps_is_flexible(p) (gst_tensor_pad_get_format (p) == _NNS_TENSOR_FORMAT_FLEXIBLE)

/**
 * @brief Macro to check current pad caps is sparse tensor.
 */
#define gst_tensor_pad_caps_is_sparse(p) (gst_tensor_pad_get_format (p) == _NNS_TENSOR_FORMAT_SPARSE)

/**
 * @brief Gets new hash table for tensor aggregation.
 * @return Newly allocated hash table, caller should release this using g_hash_table_destroy().
 */
extern GHashTable *
gst_tensor_aggregation_init (void);

/**
 * @brief Clears buffers from adapter.
 * @param table a hash table instance initialized with gst_tensor_aggregation_init()
 * @param key the key to look up (set null to get default adapter)
 */
extern void
gst_tensor_aggregation_clear (GHashTable * table, const guint32 key);

/**
 * @brief Clears buffers from all adapters in hash table.
 * @param table a hash table instance initialized with gst_tensor_aggregation_init()
 */
extern void
gst_tensor_aggregation_clear_all (GHashTable * table);

/**
 * @brief Gets adapter from hash table.
 * @param table a hash table instance initialized with gst_tensor_aggregation_init()
 * @param key the key to look up (set null to get default adapter)
 * @return gst-adapter instance. DO NOT release this instance.
 */
extern GstAdapter *
gst_tensor_aggregation_get_adapter (GHashTable * table, const guint32 key);

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
      gchar *caps_s_string = gst_caps_to_string (caps); \
      GST_DEBUG_OBJECT (self, msg " = %s\n", caps_s_string); \
      g_free (caps_s_string); \
    } \
  } \
} while (0)

G_END_DECLS
#endif /* __GST_TENSOR_COMMON_H__ */
