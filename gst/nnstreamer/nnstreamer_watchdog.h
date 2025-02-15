/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer watchdog header
 * Copyright (C) 2024 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	nnstreamer_watchdog.h
 * @date	30 Oct 2024
 * @brief	NNStreamer watchdog header to manage the schedule in the element.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs except for NYI items
 */


#ifndef __NNSTREAMER_WATCHDOG_H__
#define __NNSTREAMER_WATCHDOG_H__

#include <glib.h>
#include "nnstreamer_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *nns_watchdog_h;

/**
 * @brief Create nnstreamer watchdog. Recommended using watchdog handle with proper lock (e.g., GST_OBJECT_LOCK())
 */
gboolean NNS_API nnstreamer_watchdog_create (nns_watchdog_h *watchdog_h);

/**
 * @brief Destroy nnstreamer watchdog. Recommended using watchdog handle with proper lock (e.g., GST_OBJECT_LOCK())
 */
void NNS_API nnstreamer_watchdog_destroy (nns_watchdog_h watchdog_h);

/**
 * @brief Release watchdog source. Recommended using watchdog handle with proper lock (e.g., GST_OBJECT_LOCK())
 */
void NNS_API nnstreamer_watchdog_release (nns_watchdog_h watchdog_h);

/**
 * @brief Set watchdog timeout. Recommended using watchdog handle with proper lock (e.g., GST_OBJECT_LOCK())
 */
gboolean NNS_API nnstreamer_watchdog_feed (nns_watchdog_h watchdog_h, GSourceFunc func, guint interval, void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* __NNSTREAMER_WATCHDOG_H__ */
