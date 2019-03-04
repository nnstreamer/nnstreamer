/**
 * NNStreamer Common API Header for Plug-Ins
 * Copyright (C) 2019 Jijoong Moon <jijoong.moon@samsung.com>
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
 * @file  no_audio_define.h
 * @date  07 Mar 2019
 * @brief Define collection of audio variables and functions for no audio support
 * @see https://github.com/nnsuite/nnstreamer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 *
 */
#ifndef __NO_AUDIO_DEFINE_H__
#define __NO_AUDIO_DEFINE_H__

/**
 * @brief Macro for gst_tensor_config_from_audio_info
 */
#define gst_tensor_config_from_audio_info(...) do { \
    GST_ERROR ("\n This binary does not support audio type. Please build NNStreamer with disable-audio-support : false\n"); \
    return FALSE; \
  } while(0)

/**
 * @brief Macro for gst_audio_format_from_string
 */
#define gst_audio_format_from_string(...) GST_AUDIO_FORMAT_UNKNOWN

/**
 * @brief Macro for gst_audio_format_to_string
 */
#define gst_audio_format_to_string(...) "Unknown"

/**
 * @brief Macro for gst_audio_info_init
 */
#define gst_audio_info_init(...) do { \
    GST_ERROR ("\n This binary does not support audio type. Please build NNStreamer with disable-audio-support : false\n"); \
  } while(0)

/**
 * @brief Macro for gst_audio_info_from_caps
 */
#define gst_audio_info_from_caps(...) FALSE

/**
 * @brief Macro for GstAudioInfo structure
 */
#define GstAudioInfo gsize

/**
 * @brief Macro for GST_AUDIO_INFO_BPF
 */
#define GST_AUDIO_INFO_BPF(...) 1

/**
 * @brief GstAudioFormat ( There are more variables in audio-format.h )
 */
typedef enum _GstAudioFormat
{
  GST_AUDIO_FORMAT_UNKNOWN,
  GST_AUDIO_FORMAT_S8,
  GST_AUDIO_FORMAT_U8,
  GST_AUDIO_FORMAT_S16,
  GST_AUDIO_FORMAT_U16,
  GST_AUDIO_FORMAT_S32,
  GST_AUDIO_FORMAT_U32,
  GST_AUDIO_FORMAT_F32,
  GST_AUDIO_FORMAT_F64
} GstAudioFormat;

/**
 * @brief Macro for GST_AUDIO_CAPS_MAKE
 */
#define GST_AUDIO_CAPS_MAKE(format) \
      "audio/x-raw, " \
          "format = (string) " format
#endif
