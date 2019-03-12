/**
 * NNStreamer media type definition for tensor-converter
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
 */

/**
 * @file  converter-media-info.h
 * @date  26 Mar 2019
 * @brief Define collection of media type and functions to parse media info for the case of no audio/video support
 * @see https://github.com/nnsuite/nnstreamer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __CONVERTER_MEDIA_INFO_H__
#define __CONVERTER_MEDIA_INFO_H__

#ifndef NO_VIDEO
#include <gst/video/video-info.h>

/**
 * @brief Caps string for supported video format
 */
#define VIDEO_CAPS_STR \
    GST_VIDEO_CAPS_MAKE ("{ RGB, BGR, RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR, GRAY8 }") \
    ", views = (int) 1, interlace-mode = (string) progressive"

#define append_video_caps_template(caps) \
    gst_caps_append (caps, gst_caps_from_string (VIDEO_CAPS_STR))

#define is_video_supported(...) TRUE
#else
#define append_video_caps_template(caps)
#define is_video_supported(...) FALSE

#define GstVideoInfo gsize

typedef enum {
  GST_VIDEO_FORMAT_UNKNOWN,
  GST_VIDEO_FORMAT_GRAY8,
  GST_VIDEO_FORMAT_RGB,
  GST_VIDEO_FORMAT_BGR,
  GST_VIDEO_FORMAT_RGBx,
  GST_VIDEO_FORMAT_BGRx,
  GST_VIDEO_FORMAT_xRGB,
  GST_VIDEO_FORMAT_xBGR,
  GST_VIDEO_FORMAT_RGBA,
  GST_VIDEO_FORMAT_BGRA,
  GST_VIDEO_FORMAT_ARGB,
  GST_VIDEO_FORMAT_ABGR,
  GST_VIDEO_FORMAT_I420
} GstVideoFormat;

#define gst_video_info_init(i) memset (i, 0, sizeof (GstVideoInfo))
#define gst_video_info_from_caps(...) FALSE
#define gst_video_format_to_string(...) "Unknown"

#define GST_VIDEO_INFO_FORMAT(...) GST_VIDEO_FORMAT_UNKNOWN
#define GST_VIDEO_INFO_WIDTH(...) 0
#define GST_VIDEO_INFO_HEIGHT(...) 0
#define GST_VIDEO_INFO_SIZE(...) 0
#define GST_VIDEO_INFO_FPS_N(...) 0
#define GST_VIDEO_INFO_FPS_D(...) 1
#endif /* NO_VIDEO */

#ifndef NO_AUDIO
#include <gst/audio/audio-info.h>

/**
 * @brief Caps string for supported audio format
 */
#define AUDIO_CAPS_STR \
    GST_AUDIO_CAPS_MAKE ("{ S8, U8, S16LE, S16BE, U16LE, U16BE, S32LE, S32BE, U32LE, U32BE, F32LE, F32BE, F64LE, F64BE }") \
    ", layout = (string) interleaved"

#define append_audio_caps_template(caps) \
    gst_caps_append (caps, gst_caps_from_string (AUDIO_CAPS_STR))

#define is_audio_supported(...) TRUE
#else
#define append_audio_caps_template(caps)
#define is_audio_supported(...) FALSE

#define GstAudioInfo gsize

typedef enum {
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

#define gst_audio_info_init(i) memset (i, 0, sizeof (GstAudioInfo))
#define gst_audio_info_from_caps(...) FALSE
#define gst_audio_format_to_string(...) "Unknown"

#define GST_AUDIO_INFO_FORMAT(...) GST_AUDIO_FORMAT_UNKNOWN
#define GST_AUDIO_INFO_CHANNELS(...) 0
#define GST_AUDIO_INFO_RATE(...) 0
#define GST_AUDIO_INFO_BPF(...) 0
#endif /* NO_AUDIO */

/**
 * @brief Caps string for text input
 */
#define TEXT_CAPS_STR "text/x-raw, format = (string) utf8"

#define append_text_caps_template(caps) \
    gst_caps_append (caps, gst_caps_from_string (TEXT_CAPS_STR))

/**
 * @brief Caps string for binary stream
 */
#define OCTET_CAPS_STR "application/octet-stream"

#define append_octet_caps_template(caps) \
    gst_caps_append (caps, gst_caps_from_string (OCTET_CAPS_STR))

#endif /* __CONVERTER_MEDIA_INFO_H__ */
