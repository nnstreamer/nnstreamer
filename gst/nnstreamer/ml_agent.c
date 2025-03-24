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

#include <json-glib/json-glib.h>
#include <nnstreamer_log.h>
#include <mlops-agent-interface.h>

#include "ml_agent.h"

const gchar URI_SCHEME[] = "mlagent";
const gchar URI_KEYWORD_MODEL[] = "model";
const gchar JSON_KEY_MODEL_PATH[] = "path";

/**
 * @brief Get a path of the model file from a given GValue
 * @param[in] val A pointer to a GValue holding a G_TYPE_STRING value
 * @return A newly allocated c-string representing the model file path, if the given GValue contains a valid URI
 * Otherwise, it simply returns a duplicated (strdup'ed) c-string that the val contains.
 * @note The caller should free the return c-string after using it.
 */
gchar *
mlagent_get_model_path_from (const GValue * val)
{
  g_autofree gchar *scheme = NULL;
  g_autofree gchar *uri = g_value_dup_string (val);
  GError *err = NULL;
  gchar *uri_hier_part;
  gchar **parts;

  if (!uri)
    return NULL;

  /** Common checker for the given URI */
  scheme = g_uri_parse_scheme (uri);
  if (!scheme || g_strcmp0 (URI_SCHEME, scheme)) {
    return g_steal_pointer (&uri);
  }

  uri_hier_part = g_strstr_len (uri, -1, ":");
  if (!uri_hier_part)
    return NULL;

  while (*uri_hier_part == ':' || *uri_hier_part == '/') {
    uri_hier_part++;
  }

  /**
   * @note Only for the following URI formats to get the file path of
   *       the matching models are currently supported.
   *       mlagent://model/name or mlagent://model/name/version
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

    num_parts = g_strv_length (parts);
    if (num_parts == 0) {
      goto fallback;
    }

    if (!g_strcmp0 (parts[0], URI_KEYWORD_MODEL)) {
      /** Convert the given URI for a model to the file path */
      g_autofree gchar *name = g_strdup (parts[MODEL_PART_IDX_NAME]);
      g_autofree gchar *stringified_json = NULL;
      g_autoptr (JsonParser) json_parser = NULL;
      gint rcode;

      if (num_parts < NUM_PARTS_MODEL - 1) {
        goto fallback;
      }

      /**
       * @todo The specification of the data layout filled in the third
       *       argument (i.e., stringified_json) by the callee is not fully decided.
       */
      if (num_parts == NUM_PARTS_MODEL - 1) {
        rcode = ml_agent_model_get_activated (name, &stringified_json);
      } else {
        guint version = strtoul (parts[MODEL_PART_IDX_VERSION], NULL, 10);
        rcode = ml_agent_model_get (name, version, &stringified_json);
      }

      if (rcode != 0) {
        nns_loge
            ("Failed to get the stringified JSON using the given URI(%s)", uri);
        goto fallback;
      }

      json_parser = json_parser_new ();
      /** @todo Parse stringified_json to get the model's path */
      if (!json_parser_load_from_data (json_parser, stringified_json, -1, &err)) {
        nns_loge ("Failed to parse the stringified JSON while "
            "get the model's path: %s",
            (err ? err->message : "unknown reason"));
        goto fallback;
      }
      g_clear_error (&err);

      {
        const gchar *path = NULL;
        JsonNode *jroot;
        JsonObject *jobj;

        jroot = json_parser_get_root (json_parser);
        if (jroot == NULL) {
          nns_loge ("Failed to get JSON root node while get the model's path");
          goto fallback;
        }

        jobj = json_node_get_object (jroot);
        if (jobj == NULL) {
          nns_loge
              ("Failed to get JSON object from the root node while get the model's path");
          goto fallback;
        }

        if (!json_object_has_member (jobj, JSON_KEY_MODEL_PATH)) {
          nns_loge
              ("Failed to get the model's path from the given URI: "
              "There is no key named, %s, in the JSON object",
              JSON_KEY_MODEL_PATH);
          goto fallback;
        }

        path = json_object_get_string_member (jobj, JSON_KEY_MODEL_PATH);
        if (path == NULL || !g_strcmp0 (path, "")) {
          nns_loge
              ("Failed to get the model's path from the given URI: "
              "Invalid value for the key, %s", JSON_KEY_MODEL_PATH);
          goto fallback;
        }

        g_strfreev (parts);
        return g_strdup (path);
      }
    }
  }

fallback:
  g_clear_error (&err);
  g_strfreev (parts);

  return g_strdup (uri);
}
