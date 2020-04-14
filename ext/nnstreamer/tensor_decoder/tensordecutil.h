/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-IF
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	tensordecutil.h
 * @date	14 April 2020
 * @brief	Common utility functions for tensordec subplugins
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef _TENSORDECUTIL_H__
#define _TENSORDECUTIL_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <glib.h>

typedef uint32_t singleLineSprite_t[256][13][8];
typedef uint8_t rasters_t[][13];

typedef struct {
  char **labels; /**< The list of loaded labels. Null if not loaded */
  guint total_labels; /**< The number of loaded labels */
  guint max_word_length; /**< The max size of labels */
} imglabel_t;

extern void
loadImageLabels (const char * label_path, imglabel_t *l);

extern void
initSingleLineSprite (singleLineSprite_t v, rasters_t r, uint32_t pv);

extern void _free_labels (imglabel_t *data);

#ifdef __cplusplus
}
#endif
#endif /* _TENSORDECUTIL_H__ */
