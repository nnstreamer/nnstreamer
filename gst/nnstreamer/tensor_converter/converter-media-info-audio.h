/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer media type definition for tensor-converter
 * Copyright (C) 2019 Jijoong Moon <jijoong.moon@samsung.com>
 */

/**
 * @file  converter-media-info-audio.h
 * @date  26 Mar 2019
 * @brief Define collection of media type and functions to parse media info for audio support
 * @see https://github.com/nnstreamer/nnstreamer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __CONVERTER_MEDIA_INFO_AUDIO_H__
#define __CONVERTER_MEDIA_INFO_AUDIO_H__

#ifdef NO_AUDIO
#error This header is not supported if NO_AUDIO is defined
#endif

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
#endif /* __CONVERTER_MEDIA_INFO_AUDIO_H__ */
