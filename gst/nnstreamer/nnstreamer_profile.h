/**
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file	nnstreamer_profile.h
 * @date	14 April 2020
 * @brief	Internal util for profile log.
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __NNSTREAMER_PROFILE_H__
#define __NNSTREAMER_PROFILE_H__

#ifdef NNS_PROFILE_LOG
#include <glib.h>

G_BEGIN_DECLS

extern void nns_profile_start (const gchar * name);
extern void nns_profile_end (const gchar * name);

#define PROFILE_START 0
#define PROFILE_END 1

#define profile_log(task,state) do { \
    if (state == PROFILE_START) \
      nns_profile_start (task); \
    else \
      nns_profile_end (task); \
  } while (0);

G_END_DECLS
#else
#define profile_log(task,state)
#endif /* NNS_PROFILE_LOG */

#endif /* __NNSTREAMER_PROFILE_H__ */
