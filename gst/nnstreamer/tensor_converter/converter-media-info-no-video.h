/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer media type definition for tensor-converter
 * Copyright (C) 2019 Jijoong Moon <jijoong.moon@samsung.com>
 */

/**
 * @file  converter-media-info-video.h
 * @date  26 Mar 2019
 * @brief Define collection of media type and functions to parse media info for video if there is no video support
 * @see https://github.com/nnstreamer/nnstreamer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __CONVERTER_MEDIA_INFO_NO_VIDEO_H__
#define __CONVERTER_MEDIA_INFO_NO_VIDEO_H__

#ifndef NO_VIDEO
#error This header is not supported if NO_VIDEO is not defined
#endif

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

#endif /* __CONVERTER_MEDIA_INFO_NO_VIDEO_H__ */
