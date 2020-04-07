/**
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Library General Public License for more details.
 */

/**
 * @file	nnstreamer_log.h
 * @date	12 Mar 2020
 * @brief	Internal log util for NNStreamer plugins and native APIs.
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __NNSTREAMER_LOG_H__
#define __NNSTREAMER_LOG_H__

#define TAG_NAME "nnstreamer"

#if defined(__TIZEN__)
#include <dlog.h>

#define ml_logi(...) \
    dlog_print (DLOG_INFO, TAG_NAME, __VA_ARGS__)

#define ml_logw(...) \
    dlog_print (DLOG_WARN, TAG_NAME, __VA_ARGS__)

#define ml_loge(...) \
    dlog_print (DLOG_ERROR, TAG_NAME, __VA_ARGS__)

#define ml_logd(...) \
    dlog_print (DLOG_DEBUG, TAG_NAME, __VA_ARGS__)

#elif defined(__ANDROID__)
#include <android/log.h>

#define ml_logi(...) \
    __android_log_print (ANDROID_LOG_INFO, TAG_NAME, __VA_ARGS__)

#define ml_logw(...) \
    __android_log_print (ANDROID_LOG_WARN, TAG_NAME, __VA_ARGS__)

#define ml_loge(...) \
    __android_log_print (ANDROID_LOG_ERROR, TAG_NAME, __VA_ARGS__)

#define ml_logd(...) \
    __android_log_print (ANDROID_LOG_DEBUG, TAG_NAME, __VA_ARGS__)

#else /* Linux distro */
#include <glib.h>

#define ml_logi g_info
#define ml_logw g_warning
#define ml_loge g_critical
#define ml_logd g_debug
#endif

#define nns_logi ml_logi
#define nns_logw ml_logw
#define nns_loge ml_loge
#define nns_logd ml_logd

#endif /* __NNSTREAMER_LOG_H__ */
