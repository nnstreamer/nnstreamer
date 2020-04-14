/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer media type definition for tensor-converter
 * Copyright (C) 2019 Jijoong Moon <jijoong.moon@samsung.com>
 */

/**
 * @file  converter-media-info-no-audio.h
 * @date  26 Mar 2019
 * @brief Define collection of media type and functions to parse media info for audio if there is no audio support
 * @see https://github.com/nnstreamer/nnstreamer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __CONVERTER_MEDIA_INFO_NO_AUDIO_H__
#define __CONVERTER_MEDIA_INFO_NO_AUDIO_H__

#ifndef NO_AUDIO
#error This header is not supported if NO_AUDIO is not defined
#endif

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


#endif /* __CONVERTER_MEDIA_INFO_NO_AUDIO_H__ */
