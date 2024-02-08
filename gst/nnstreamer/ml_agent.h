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
 * @param[in] val A pointer to a GValue holding a G_TYPE_STRING value
 * @return A newly allocated c-string containing the model file path in the case that the valid URI is given.
 * Otherwise, it simply returns c-string that the val contains.
 * @note The caller should free the return c-string after using it.
 */
gchar *mlagent_get_model_path_from (const GValue * val);
#else
#define mlagent_get_model_path_from(v) g_value_dup_string (v)
#endif

#endif /* __ML_AGENT_H__ */
