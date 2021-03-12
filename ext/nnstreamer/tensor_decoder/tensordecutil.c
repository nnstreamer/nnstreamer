/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-IF
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	tensordecutil.c
 * @date	14 April 2020
 * @brief	Common utility functions for tensordec subplugins
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <glib.h>
#include <string.h>
#include <nnstreamer_log.h>
#include "tensordecutil.h"
#include <gst/gstvalue.h>

/**
 * @brief Load label file into the internal data
 * @param[in/out] l The given ImageLabelData struct.
 */
void
loadImageLabels (const char *label_path, imglabel_t * l)
{
  GError *err = NULL;
  gchar **_labels;
  gchar *contents = NULL;
  gsize len;
  guint i;

  _free_labels (l);

  /* Read file contents */
  if (!g_file_get_contents (label_path, &contents, &len, &err)) {
    ml_loge ("Unable to read file %s with error %s.", label_path, err->message);
    g_clear_error (&err);
    return;
  }

  if (contents[len - 1] == '\n')
    contents[len - 1] = '\0';

  _labels = g_strsplit (contents, "\n", -1);
  l->total_labels = g_strv_length (_labels);
  l->labels = g_new0 (char *, l->total_labels);
  if (l->labels == NULL) {
    ml_loge ("Failed to allocate memory for label data.");
    l->total_labels = 0;
    goto error;
  }

  for (i = 0; i < l->total_labels; i++) {
    l->labels[i] = g_strdup (_labels[i]);

    len = strlen (_labels[i]);
    if (len > l->max_word_length) {
      l->max_word_length = len;
    }
  }

error:
  g_strfreev (_labels);
  g_free (contents);

  if (l->labels != NULL) {
    ml_logi ("Loaded image label file successfully. %u labels loaded.",
        l->total_labels);
  }
  return;
}

/**
 * @brief Initialize ASCII font sprite.
 */
void
initSingleLineSprite (singleLineSprite_t v, rasters_t r, uint32_t pv)
{
  int i, j, k;
  /** @todo Constify singleLineSprite and remove this loop */

  for (i = 0; i < 256; i++) {
    int ch = i;
    uint8_t val;

    if (ch < 32 || ch >= 127) {
      /* It's not ASCII */
      ch = '*';
    }
    ch -= 32;

    for (j = 0; j < 13; j++) {
      val = r[ch][j];
      for (k = 0; k < 8; k++) {
        if (val & 0x80)
          v[i][12 - j][k] = pv;
        else
          v[i][12 - j][k] = 0;
        val <<= 1;
      }
    }
  }
}

/**
 * @brief Free image labels
 */
void
_free_labels (imglabel_t * data)
{
  guint i;

  if (data->labels) {
    for (i = 0; i < data->total_labels; i++)
      g_free (data->labels[i]);
    g_free (data->labels);
  }
  data->labels = NULL;
  data->total_labels = 0;
  data->max_word_length = 0;
}

/**
 * @brief Copy framerate caps from tensor config
 *
 * @param[out]  caps to configure
 * @param[in]   config to copy from
 */
void
setFramerateFromConfig (GstCaps * caps, const GstTensorsConfig * config)
{
  gint fn, fd;

  /** @todo Verify if this rate is ok */
  fn = config->rate_n;
  fd = config->rate_d;
  if (fn >= 0 && fd > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, fn, fd, NULL);
  }
  return;
}
