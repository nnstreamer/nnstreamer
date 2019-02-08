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
#include <gst/video/video-format.h>
#include <gst/audio/audio-format.h>
#include <gst/base/gstcollectpads.h>
#include <gst/gstplugin.h>

#include "tensor_typedef.h"
#include "nnstreamer_plugin_api.h"

#ifdef HAVE_ORC
#include <orc/orc.h>

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

#define GST_TENSOR_VIDEO_CAPS_STR \
    GST_VIDEO_CAPS_MAKE ("{ RGB, BGR, RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR, GRAY8 }") \
    ", views = (int) 1, interlace-mode = (string) progressive"

#define GST_TENSOR_AUDIO_CAPS_STR \
    GST_AUDIO_CAPS_MAKE ("{ S8, U8, S16LE, S16BE, U16LE, U16BE, S32LE, S32BE, U32LE, U32BE, F32LE, F32BE, F64LE, F64BE }") \
    ", layout = (string) interleaved"

#define GST_TENSOR_TEXT_CAPS_STR \
    "text/x-raw, format = (string) utf8"

#define GST_TENSOR_OCTET_CAPS_STR \
    "application/octet-stream"

/**
 * @brief Caps string for supported types
 * @todo Support other types
 */
#define GST_TENSOR_MEDIA_CAPS_STR \
    GST_TENSOR_VIDEO_CAPS_STR "; " \
    GST_TENSOR_AUDIO_CAPS_STR "; " \
    GST_TENSOR_TEXT_CAPS_STR "; " \
    GST_TENSOR_OCTET_CAPS_STR

#define GST_TENSOR_TYPE_ALL "{ float32, float64, int64, uint64, int32, uint32, int16, uint16, int8, uint8 }"

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
 * @brief Compare tensor info
 * @param TRUE if equal
 */
extern gboolean
gst_tensor_info_is_equal (const GstTensorInfo * i1, const GstTensorInfo * i2);

/**
 * @brief Copy tensor info
 * @note GstTensorInfo::name should be freed with g_free()
 */
extern void
gst_tensor_info_copy (GstTensorInfo * dest, const GstTensorInfo * src);

/**
 * @brief Compare tensors info
 * @param TRUE if equal
 */
extern gboolean
gst_tensors_info_is_equal (const GstTensorsInfo * i1, const GstTensorsInfo * i2);

/**
 * @brief Copy tensor info
 * @note GstTensorInfo::name should be freed with g_free()
 */
extern void
gst_tensors_info_copy (GstTensorsInfo * dest, const GstTensorsInfo * src);

/**
 * @brief Parse the string of names
 * @param info tensors info structure
 * @param name_string string of names
 * @return number of parsed names
 */
extern guint
gst_tensors_info_parse_names_string (GstTensorsInfo * info,
    const gchar * name_string);

/**
 * @brief Get the string of dimensions in tensors info
 * @param info tensors info structure
 * @return string of dimensions in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
extern gchar *
gst_tensors_info_get_dimensions_string (const GstTensorsInfo * info);

/**
 * @brief Get the string of types in tensors info
 * @param info tensors info structure
 * @return string of types in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
extern gchar *
gst_tensors_info_get_types_string (const GstTensorsInfo * info);

/**
 * @brief Get the string of tensor names in tensors info
 * @param info tensors info structure
 * @return string of names in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
extern gchar *
gst_tensors_info_get_names_string (const GstTensorsInfo * info);

/**
 * @brief Compare tensor config info
 * @param TRUE if equal
 */
extern gboolean
gst_tensor_config_is_equal (const GstTensorConfig * c1,
    const GstTensorConfig * c2);

/**
 * @brief Get tensor caps from tensor config (for other/tensor)
 * @param config tensor config info
 * @return caps for given config
 */
extern GstCaps *
gst_tensor_caps_from_config (const GstTensorConfig * config);

/**
 * @brief Compare tensor config info (for other/tensors)
 * @param TRUE if equal
 */
extern gboolean
gst_tensors_config_is_equal (const GstTensorsConfig * c1,
    const GstTensorsConfig * c2);

/**
 * @brief Get caps from tensors config (for other/tensors)
 * @param config tensors config info
 * @return caps for given config
 */
extern GstCaps *
gst_tensors_caps_from_config (const GstTensorsConfig * config);

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
extern int find_key_strv (const gchar ** strv, const gchar * key);

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
