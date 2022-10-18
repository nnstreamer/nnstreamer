/* GStreamer
 *
 * Copyright (C) 2018 Samsung Electronics Co., Ltd.
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
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

/**
 * @file	gsttensor_train.c
 * @date	11 October 2022
 * @brief	Function to train tensor data using NN Frameworks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	nnfw <nnfw@samsung.com>
 * @bug		No known bugs except for NYI items
 * 
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "tensor_train.h"

GST_DEBUG_CATEGORY_EXTERN (gst_tensor_trainsink_debug);
#define GST_CAT_DEFAULT gst_tensor_trainsink_debug

/**
 * @brief Find sub-plugin trainer given the name list
 * @param[in] names comma, whitespace separated list of the sub-plugin name
 * @return the best-fit sub-plugin object or NULL if not found.
 */
static const GstTensorFilterFramework *
gst_tensor_trainsink_find_best_framework (const char *names)
{
  const GstTensorFilterFramework *fw = NULL;    /* need to change to GstTensorTrainerFramework* */
  gchar **subplugins;
  guint i, len;

  if (names == NULL || names[0] == '\0')
    return NULL;

  subplugins = g_strsplit_set (names, " ,;", -1);

  len = g_strv_length (subplugins);

  for (i = 0; i < len; i++) {
    if (strlen (g_strstrip (subplugins[i])) == 0)
      continue;

    fw = get_subplugin (NNS_SUBPLUGIN_FILTER, subplugins[i]);   /* need to add trainer type to subpluginType */
    if (fw) {
      GST_INFO ("i = %d found %s", i, subplugins[i]);
      break;
    }
  }
  g_strfreev (subplugins);

  return fw;
}

/**
 * @brief Find Trainer sub-plugin with the name.
 * @param[in] sink GstTensorTrainSink.
 * @param[in] name The name of trainer sub-plugin.
 */
void
gst_tensor_trainsink_find_framework (GstTensorTrainSink * sink,
    const char *name)
{
  const GstTensorFilterFramework *fw = NULL;
  gchar *_str;
  g_return_if_fail (name != NULL);

  GST_INFO ("find framework: %s", name);

  fw = get_subplugin (NNS_SUBPLUGIN_FILTER, name);      /* need to add trainer type to subpluginType */

  if (fw == NULL) {
    /* get sub-plugin priority from ini file and find sub-plugin */
    _str = nnsconf_get_custom_value_string (name, "subplugin_priority");
    fw = gst_tensor_trainsink_find_best_framework (_str);
    g_free (_str);
  }

  if (fw == NULL) {
    /* Check the filter-alias from ini file */
    _str = nnsconf_get_custom_value_string ("filter-aliases", name);
    fw = gst_tensor_trainsink_find_best_framework (_str);
    g_free (_str);
  }

  if (fw) {
    GST_INFO_OBJECT (sink, "find framework %s:%p", sink->fw_name, fw);
    sink->fw = fw;
  } else {
    GST_ERROR_OBJECT (sink, "Can not find framework(%s)", sink->fw_name);
  }
}

/**
 * @brief Open NN framework.
 */
void
gst_tensor_trainsink_create_framework (GstTensorTrainSink * sink)
{
  if (!sink->fw || sink->fw_opened) {
    GST_ERROR_OBJECT (sink, "fw is not opened(%d) or fw is null(%p)",
        sink->fw_opened, sink->fw);
    return;
  }

  /* for test */
  if (!sink->fw->open) {        /* fw->create, create model with configuration file */
    GST_ERROR_OBJECT (sink, "Could not find fw->create");
    return;
  }
  /* test code, need to create with load ini file */
  if (sink->fw->open (&sink->prop, &sink->privateData) >= 0)
    sink->fw_created = TRUE;
}

/**
 * @brief Calculate tensor buffer size.
 * @param self "this" pointer
 * @param index index of tensors
 * @return tensor buffer size
 */
gsize
gst_tensor_trainsink_get_tensor_size (GstTensorTrainSink * sink, guint index,
    gboolean is_input)
{
  GstTensorsInfo *info;

  if (is_input)
    info = &sink->prop.input_meta;
  else
    info = &sink->prop.output_meta;

  /* Internal Logic Error: out of bound */
  if (index >= info->num_tensors) {
    GST_ERROR_OBJECT (sink, "has inconsistent data");
    return 0;
  }

  return gst_tensor_info_get_size (&info->info[index]);
}
