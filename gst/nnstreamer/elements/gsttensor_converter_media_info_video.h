/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer media type definition for tensor-converter
 * Copyright (C) 2019 Jijoong Moon <jijoong.moon@samsung.com>
 */

/**
 * @file  gsttensor_converter_media_info_video.h
 * @date  26 Mar 2019
 * @brief Define collection of media type and functions to parse media info for video support
 * @see https://github.com/nnstreamer/nnstreamer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_CONVERTER_MEDIA_INFO_VIDEO_H__
#define __GST_TENSOR_CONVERTER_MEDIA_INFO_VIDEO_H__

#ifdef NO_VIDEO
#error This header is not supported if NO_VIDEO is defined
#endif

#include <gst/video/video-info.h>

/**
 * @brief Caps string for supported video format
 */
#define VIDEO_CAPS_STR \
    GST_VIDEO_CAPS_MAKE ("{ RGB, BGR, RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR, GRAY8, GRAY16_BE, GRAY16_LE }") \
    ", interlace-mode = (string) progressive"

#define append_video_caps_template(caps) \
    gst_caps_append (caps, gst_caps_from_string (VIDEO_CAPS_STR))

#define is_video_supported(...) TRUE
#endif /* __GST_TENSOR_CONVERTER_MEDIA_INFO_VIDEO_H__ */
