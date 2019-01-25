/**
 * NNStreamer Common Header's Contents
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	tensor_common.c
 * @date	29 May 2018
 * @brief	Common data for NNStreamer, the GStreamer plugin for neural networks
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <tensor_common.h>
#include <string.h>
#include <glib.h>

/**
 * @brief Get media type from caps
 * @param caps caps to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_END for unsupported type)
 */
media_type
gst_tensor_media_type_from_caps (const GstCaps * caps)
{
  GstStructure *structure;

  structure = gst_caps_get_structure (caps, 0);

  return gst_tensor_media_type_from_structure (structure);
}

/**
 * @brief Compare tensor info
 * @param TRUE if equal
 */
gboolean
gst_tensor_info_is_equal (const GstTensorInfo * i1, const GstTensorInfo * i2)
{
  guint i;

  g_return_val_if_fail (i1 != NULL, FALSE);
  g_return_val_if_fail (i2 != NULL, FALSE);

  if (i1->type != i2->type) {
    return FALSE;
  }

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (i1->dimension[i] != i2->dimension[i]) {
      return FALSE;
    }
  }

  /** matched all */
  return TRUE;
}

/**
 * @brief Copy tensor info
 * @note GstTensorInfo::name should be freed with g_free()
 */
void
gst_tensor_info_copy (GstTensorInfo * dest, const GstTensorInfo * src)
{
  guint i;

  g_return_if_fail (dest != NULL);
  g_return_if_fail (src != NULL);

  dest->name = (src->name) ? g_strdup (src->name) : NULL;
  dest->type = src->type;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    dest->dimension[i] = src->dimension[i];
  }
}

/**
 * @brief Compare tensors info
 * @param TRUE if equal
 */
gboolean
gst_tensors_info_is_equal (const GstTensorsInfo * i1, const GstTensorsInfo * i2)
{
  guint i;

  g_return_val_if_fail (i1 != NULL, FALSE);
  g_return_val_if_fail (i2 != NULL, FALSE);

  if (i1->num_tensors != i2->num_tensors) {
    return FALSE;
  }

  for (i = 0; i < i1->num_tensors; i++) {
    if (!gst_tensor_info_is_equal (&i1->info[i], &i2->info[i])) {
      return FALSE;
    }
  }

  /** matched all */
  return TRUE;
}

/**
 * @brief Copy tensor info
 * @note GstTensorInfo::name should be freed with g_free()
 */
void
gst_tensors_info_copy (GstTensorsInfo * dest, const GstTensorsInfo * src)
{
  guint i;

  g_return_if_fail (dest != NULL);
  g_return_if_fail (src != NULL);

  dest->num_tensors = src->num_tensors;

  for (i = 0; i < src->num_tensors; i++) {
    gst_tensor_info_copy (&dest->info[i], &src->info[i]);
  }
}

/**
 * @brief Parse the string of names
 * @param info tensors info structure
 * @param name_string string of names
 * @return number of parsed names
 */
guint
gst_tensors_info_parse_names_string (GstTensorsInfo * info,
    const gchar * name_string)
{
  gint num_names = 0;

  g_return_val_if_fail (info != NULL, 0);

  if (name_string) {
    guint i;
    gchar *str_name;
    gchar **str_names;

    str_names = g_strsplit (name_string, ",", -1);
    num_names = g_strv_length (str_names);

    if (num_names > NNS_TENSOR_SIZE_LIMIT) {
      GST_WARNING ("Invalid param, names (%d) max (%d)\n",
          num_names, NNS_TENSOR_SIZE_LIMIT);

      num_names = NNS_TENSOR_SIZE_LIMIT;
    }

    for (i = 0; i < num_names; i++) {
      str_name = g_strdup (str_names[i]);
      g_strstrip (str_name);

      g_free (info->info[i].name);
      info->info[i].name = str_name;
    }

    g_strfreev (str_names);
  }

  return num_names;
}

/**
 * @brief Get the string of dimensions in tensors info
 * @param info tensors info structure
 * @return string of dimensions in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
gchar *
gst_tensors_info_get_dimensions_string (const GstTensorsInfo * info)
{
  gchar *dim_str = NULL;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *dimensions = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      dim_str = get_tensor_dimension_string (info->info[i].dimension);

      g_string_append (dimensions, dim_str);

      if (i < info->num_tensors - 1) {
        g_string_append (dimensions, ",");
      }

      g_free (dim_str);
    }

    dim_str = g_string_free (dimensions, FALSE);
  }

  return dim_str;
}

/**
 * @brief Get the string of types in tensors info
 * @param info tensors info structure
 * @return string of types in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
gchar *
gst_tensors_info_get_types_string (const GstTensorsInfo * info)
{
  gchar *type_str = NULL;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *types = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      g_string_append (types, tensor_element_typename[info->info[i].type]);

      if (i < info->num_tensors - 1) {
        g_string_append (types, ",");
      }
    }

    type_str = g_string_free (types, FALSE);
  }

  return type_str;
}

/**
 * @brief Get the string of tensor names in tensors info
 * @param info tensors info structure
 * @return string of names in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
gchar *
gst_tensors_info_get_names_string (const GstTensorsInfo * info)
{
  gchar *name_str = NULL;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *names = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      g_string_append (names, info->info[i].name);

      if (i < info->num_tensors - 1) {
        g_string_append (names, ",");
      }
    }

    name_str = g_string_free (names, FALSE);
  }

  return name_str;
}

/**
 * @brief Compare tensor config info
 * @param TRUE if equal
 */
gboolean
gst_tensor_config_is_equal (const GstTensorConfig * c1,
    const GstTensorConfig * c2)
{
  g_return_val_if_fail (c1 != NULL, FALSE);
  g_return_val_if_fail (c2 != NULL, FALSE);

  /** @todo 1/2 == 2/4 Don't say they are different! */
  if (c1->rate_n != c2->rate_n || c1->rate_d != c2->rate_d) {
    return FALSE;
  }

  return gst_tensor_info_is_equal (&c1->info, &c2->info);
}

/**
 * @brief Get tensor caps from tensor config
 * @param config tensor config info
 * @return caps for given config
 */
GstCaps *
gst_tensor_caps_from_config (const GstTensorConfig * config)
{
  GstCaps *caps;

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT);

  if (gst_tensor_dimension_is_valid (config->info.dimension)) {
    gchar *dim_str = get_tensor_dimension_string (config->info.dimension);

    gst_caps_set_simple (caps, "dimension", G_TYPE_STRING, dim_str, NULL);
    g_free (dim_str);
  }

  if (config->info.type != _NNS_END) {
    gst_caps_set_simple (caps, "type", G_TYPE_STRING,
        tensor_element_typename[config->info.type], NULL);
  }

  if (config->rate_n >= 0 && config->rate_d > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
  }

  return gst_caps_simplify (caps);
}

/**
 * @brief Compare tensor config info
 * @param TRUE if equal
 */
gboolean
gst_tensors_config_is_equal (const GstTensorsConfig * c1,
    const GstTensorsConfig * c2)
{
  g_return_val_if_fail (c1 != NULL, FALSE);
  g_return_val_if_fail (c2 != NULL, FALSE);

  if (c1->rate_n != c2->rate_n || c1->rate_d != c2->rate_d) {
    return FALSE;
  }

  return gst_tensors_info_is_equal (&c1->info, &c2->info);
}

/**
 * @brief Get caps from tensors config (for other/tensors)
 * @param config tensors config info
 * @return caps for given config
 */
GstCaps *
gst_tensors_caps_from_config (const GstTensorsConfig * config)
{
  GstCaps *caps;

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_caps_from_string (GST_TENSORS_CAP_DEFAULT);

  if (config->info.num_tensors > 0) {
    gchar *dim_str, *type_str;

    dim_str = gst_tensors_info_get_dimensions_string (&config->info);
    type_str = gst_tensors_info_get_types_string (&config->info);

    gst_caps_set_simple (caps, "num_tensors", G_TYPE_INT,
        config->info.num_tensors, NULL);
    gst_caps_set_simple (caps, "dimensions", G_TYPE_STRING, dim_str, NULL);
    gst_caps_set_simple (caps, "types", G_TYPE_STRING, type_str, NULL);

    g_free (dim_str);
    g_free (type_str);
  }

  if (config->rate_n >= 0 && config->rate_d > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
  }

  return gst_caps_simplify (caps);
}

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index. Returns -1 if not found.
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
int
find_key_strv (const gchar ** strv, const gchar * key)
{
  int cursor = 0;

  g_assert (strv != NULL);
  while (strv[cursor]) {
    if (!g_ascii_strcasecmp (strv[cursor], key))
      return cursor;
    cursor++;
  }

  return -1;                    /* Not Found */
}

/**
 * @brief A callback for typefind, trying to find whether a file is other/tensors or not.
 * For the concrete definition of headers, please look at the wiki page of nnstreamer:
 * https://github.com/nnsuite/nnstreamer/wiki/Design-External-Save-Format-for-other-tensor-and-other-tensors-Stream-for-TypeFind
 */
void
gst_tensors_typefind_function (GstTypeFind * tf, gpointer pdata)
{
  const guint8 *data = gst_type_find_peek (tf, 0, 40);  /* The first 40 bytes are header-0 in v.1 protocol */
  const char formatstr[] = "TENSORST";
  const unsigned int *supported_version = (const unsigned int *) (&data[8]);
  const unsigned int *num_tensors = (const unsigned int *) (&data[12]);
  if (data &&
      memcmp (data, formatstr, 8) == 0 &&
      *supported_version == 1 && *num_tensors <= 16 && *num_tensors >= 1) {
    gst_type_find_suggest (tf, GST_TYPE_FIND_MAXIMUM,
        gst_caps_new_simple ("other/tensorsave", NULL, NULL));
  }
}

/**
 * @brief A function call to decide current timestamp among collected pads based on PTS.
 * It will decide current timestamp according to sync option.
 */
gboolean
gst_tensor_set_current_time (GstCollectPads * collect,
    GstClockTime * current_time, tensor_time_sync_data sync)
{
  GSList *walk = NULL;
  walk = collect->data;
  gboolean isEOS = FALSE;
  guint count = 0;

  while (walk) {
    GstCollectData *data = (GstCollectData *) walk->data;
    walk = g_slist_next (walk);
    GstBuffer *buf = NULL;
    buf = gst_collect_pads_peek (collect, data);

    if (buf == NULL) {
      isEOS = TRUE;
      return isEOS;
    }

    switch (sync.mode) {
      case SYNC_SLOWEST:
        if (*current_time < GST_BUFFER_PTS (buf))
          *current_time = GST_BUFFER_PTS (buf);
        break;
      case SYNC_BASEPAD:
        if (count == sync.data_basepad.sink_id)
          *current_time = GST_BUFFER_PTS (buf);
        break;
      default:
        break;
    }
    count++;
    gst_buffer_unref (buf);
  }

  return isEOS;
}

/**
 * @brief A function call to make tensors from collected pads.
 * It decide which buffer is going to be used according to sync option.
 */
gboolean
gst_gen_tensors_from_collectpad (GstCollectPads * collect,
    tensor_time_sync_data sync, GstClockTime current_time,
    gboolean * need_buffer, GstBuffer * tensors_buf, GstTensorsConfig * configs)
{
  GSList *walk = NULL;
  GstMemory *mem;
  gboolean isEOS = FALSE;
  gint old_numerator = G_MAXINT;
  gint old_denominator = G_MAXINT;
  gint counting = 0;
  GstTensorsConfig in_configs;
  GstClockTime base = 0;
  guint i = 0;

  walk = collect->data;

  if (sync.mode == SYNC_BASEPAD) {
    walk = g_slist_nth (walk, sync.data_basepad.sink_id);
    GstCollectData *data = (GstCollectData *) walk->data;
    GstTensorCollectPadData *pad = (GstTensorCollectPadData *) data;
    GstBuffer *buf = NULL;
    buf = gst_collect_pads_peek (collect, data);
    if (buf != NULL) {
      if (pad->buffer != NULL)
        base =
            MIN (sync.data_basepad.duration,
            ABS (GST_CLOCK_DIFF (GST_BUFFER_PTS (buf),
                    GST_BUFFER_PTS (pad->buffer))) - 1);
      gst_buffer_unref (buf);
    }
  }

  walk = collect->data;

  while (walk) {
    GstCollectData *data = (GstCollectData *) walk->data;
    GstTensorCollectPadData *pad = (GstTensorCollectPadData *) data;
    GstCaps *caps = gst_pad_get_current_caps (pad->pad);
    GstStructure *s = gst_caps_get_structure (caps, 0);

    gst_tensors_config_from_structure (&in_configs, s);
    g_assert (gst_tensors_config_validate (&in_configs));

    if (in_configs.rate_d < old_denominator)
      old_denominator = in_configs.rate_d;
    if (in_configs.rate_n < old_numerator)
      old_numerator = in_configs.rate_n;

    gst_caps_unref (caps);

    walk = g_slist_next (walk);

    GstBuffer *buf = NULL;

    buf = gst_collect_pads_peek (collect, data);

    switch (sync.mode) {
      case SYNC_SLOWEST:
        if (buf != NULL) {
          if (GST_BUFFER_PTS (buf) < current_time) {
            gst_buffer_unref (buf);
            if (pad->buffer != NULL)
              gst_buffer_unref (pad->buffer);
            pad->buffer = gst_collect_pads_pop (collect, data);
            *need_buffer = TRUE;
            return FALSE;
          }

          if (pad->buffer != NULL &&
              (ABS (GST_CLOCK_DIFF (current_time,
                          GST_BUFFER_PTS (pad->buffer))) <
                  ABS (GST_CLOCK_DIFF (current_time, GST_BUFFER_PTS (buf))))) {
            gst_buffer_unref (buf);
            buf = pad->buffer;
          } else {
            gst_buffer_unref (buf);
            buf = gst_collect_pads_pop (collect, data);
            if (pad->buffer != NULL)
              gst_buffer_unref (pad->buffer);
            pad->buffer = buf;
          }
        } else {
          buf = pad->buffer;
        }
        break;
      case SYNC_NOSYNC:
        if (buf != NULL) {
          gst_buffer_unref (buf);
          buf = gst_collect_pads_pop (collect, data);
        }
        break;
      case SYNC_BASEPAD:
        if (buf != NULL) {
          if (GST_BUFFER_PTS (buf) < current_time) {
            gst_buffer_unref (buf);
            if (pad->buffer != NULL)
              gst_buffer_unref (pad->buffer);
            pad->buffer = gst_collect_pads_pop (collect, data);
            *need_buffer = TRUE;
            return FALSE;
          }

          if ((pad->buffer != NULL &&
                  (ABS (GST_CLOCK_DIFF (current_time,
                              GST_BUFFER_PTS (buf))) > base))) {
            gst_buffer_unref (buf);
            buf = pad->buffer;
          } else {
            gst_buffer_unref (buf);
            buf = gst_collect_pads_pop (collect, data);
            if (pad->buffer != NULL)
              gst_buffer_unref (pad->buffer);
            pad->buffer = buf;
          }
        } else {
          buf = pad->buffer;
        }
        break;
      default:
        break;
    }

    if (GST_IS_BUFFER (buf)) {
      guint n_mem = gst_buffer_n_memory (buf);

      g_assert (n_mem == in_configs.info.num_tensors);
      g_assert ((counting + n_mem) < NNS_TENSOR_SIZE_LIMIT);

      for (i = 0; i < n_mem; ++i) {
        mem = gst_buffer_get_memory (buf, i);
        gst_buffer_append_memory (tensors_buf, mem);
        configs->info.info[counting] = in_configs.info.info[i];
        counting++;
      }
      if (sync.mode == SYNC_NOSYNC) {
        gst_buffer_unref (buf);
      }

    } else {
      isEOS = TRUE;
      return isEOS;
    }
  }

  configs->rate_d = old_denominator;
  configs->rate_n = old_numerator;

  GST_BUFFER_PTS (tensors_buf) = current_time;
  return isEOS;
}
