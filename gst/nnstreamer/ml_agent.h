/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2023 Wook Song <wook16.song@samsung.com>
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 *
 * @file    ml_agent.h
 * @date    23 Jun 2023
 * @brief   Internal header to make a bridge between NNS filters and the ML Agent service
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  wook16.song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __ML_AGENT_H__
#define __ML_AGENT_H__

#include <glib-object.h>

#ifdef ENABLE_ML_AGENT
/**
 * @brief Parse the given URI into the valid file path string
 */
const gchar *mlagent_parse_uri_string (const GValue * val);
#else
#define mlagent_parse_uri_string g_value_get_string
#endif

#endif /* __ML_AGENT_H__ */
