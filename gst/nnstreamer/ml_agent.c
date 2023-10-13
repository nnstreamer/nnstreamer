/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2023 Wook Song <wook16.song@samsung.com>
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 *
 * @file    ml_agent.c
 * @date    23 Jun 2023
 * @brief   Internal helpers to make a bridge between NNS filters and the ML Agent service
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  wook16.song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <ml-agent-interface.h>

#include "ml_agent.h"

const gchar URI_SCHEME[] = "mlagent";
const gchar URI_KEYWORD_MODEL[] = "model";

/**
 * @brief Parse the given URI into the valid file path string
 */
const gchar *
mlagent_parse_uri_string (const GValue * val)
{
  const g_autofree gchar *ret = g_value_get_string (val);
  const g_autofree gchar *scheme;
  gchar *uri_hier_part;
  gchar **parts;

  if (!ret)
    return NULL;

  /** Common checker for the given URI */
  scheme = g_uri_parse_scheme (ret);
  if (!scheme || g_strcmp0 (URI_SCHEME, scheme)) {
    return g_steal_pointer (&ret);
  }

  uri_hier_part = g_strstr_len (ret, -1, ":");
  while (*uri_hier_part == ':' || *uri_hier_part == '/') {
    uri_hier_part++;
  }

  /**
   * @note Only for the following URI formats to get the file path of
   *       the matching models are currently supported.
   *       mlagent://model/name/version or mlagent://model/name/version
   *
   *       It is required to be revised to support more scenarios
   *       that exploit the ML Agent.
   */
  parts = g_strsplit_set (uri_hier_part, "/", 0);
  {
    enum MODEL_PART_CONSTANTS
    {
      MODEL_PART_IDX_NAME = 1,
      MODEL_PART_IDX_VERSION = 2,
      MODEL_VERSION_MIN = 1,
      MODEL_VERSION_MAX = 255,
    };

    const size_t NUM_PARTS_MODEL = 3;
    size_t num_parts;
    size_t i = 0;

    while (parts[++i]);

    num_parts = i;
    if (num_parts == 0) {
      goto fallback;
    }

    if (!g_strcmp0 (parts[0], URI_KEYWORD_MODEL)
        && num_parts >= NUM_PARTS_MODEL) {
      /** Convert the given URI for a model to the file path */
      g_autofree gchar *name = g_strdup (parts[MODEL_PART_IDX_NAME]);
      guint version = strtoul (parts[MODEL_PART_IDX_VERSION], NULL, 10);
      g_auto (GStrv) stringfied_json;
      GError *err;
      gint rcode;

      /**
       * @todo The specification of the data layout filled in the third
       *       argument (i.e., stringfied_json) by the callee is not fully decided.
       */
      rcode = ml_agent_model_get (name, version, stringfied_json, &err);
      g_clear_error (&err);

      if (rcode != 0)
        goto fallback;

      /** @todo Parse stringfied_json to get the model's path */
    } else {
      /** TBU */
      goto fallback;
    }
  }

fallback:
  g_strfreev (parts);

  return g_steal_pointer (&ret);
}
