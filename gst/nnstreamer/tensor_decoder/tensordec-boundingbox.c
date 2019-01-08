/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "image labeling"
 * Copyright (C) 2018 Samsung Electronics Co. Ltd.
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
 * @file        tensordec-boundingbox.c
 * @date        15 Nov 2018
 * @brief       NNStreamer tensor-decoder subplugin, "bounding boxes",
 *              which converts tensors to video stream w/ boxes on
 *              transparent background.
 *              This code is NYI/WIP and not compilable.
 *
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 * option1: Decoder mode of bounding box.
 *          Available: ssd (single shot multibox detector with priors.)
 *          Default (unsupplied): ssd
 * option2: Location of label file
 *          This is independent from option1
 * option3: Location of box prior file (ssd) or any option1-dependent values
 *          !!This depends on option1 values!!
 * option4: Video Output Dimension (WIDTH:HEIGHT)
 *          This is independent from option1
 * option5: Input Dimension (WIDTH:HEIGHT)
 *          This is independent from option1
 * option6: Box Style (NYI)
 *
 * MAJOR TODO: Support other colorspaces natively from _decode for performance gain
 * (e.g., BGRA, ARGB, ...)
 *
 */

/** @todo getline requires _GNU_SOURCE. remove this later. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <glib.h>
#include <gst/gst.h>
#include <math.h>               /* expf */
#include <tensor_common.h>
#include "tensordec.h"
#include "font.h"

#define BOX_SIZE        4
#define DETECTION_MAX   1917
#define PIXEL_VALUE (0xFF0000FF)        /* RED 100% in RGBA */

/**
 * @todo Fill in the value at build time or hardcode this. It's const value
 * @brief The bitmap of characters
 * [Character (ASCII)][Height][Width]
 */
static uint32_t singleLineSprite[256][13][8];

/**
 * @brief There can be different schemes for bounding boxes.
 */
typedef enum
{
  BOUNDING_BOX_DEFAULT = 0,     /* SSD is the Default */
  BOUNDING_BOX_SSD = 0,
  BOUNDING_BOX_UNKNOWN,
} bounding_box_modes;

/**
 * @brief List of bounding-box decoding schemes in string
 */
static const gchar *bb_modes[] = {
  [BOUNDING_BOX_SSD] = "ssd",
  NULL,
};

/**
 * @brief Data structure for SSD boundig box info.
 */
typedef struct
{
  /* From option3, box prior data */
  gchar *box_prior_path; /**< Box Prior file path */
  gfloat box_priors[BOX_SIZE][DETECTION_MAX + 1]; /** loaded box prior */
} properties_SSD;

/**
 * @brief Data structure for boundig box info.
 */
typedef struct
{
  bounding_box_modes mode; /**< The bounding box decoding mode */

  union
  {
    properties_SSD ssd; /**< Properties for ssd configured by option 1 + 3 */
  };

  /* From option2 */
  gchar *label_path; /**< Label file path. */
  gchar **labels; /**< The list of loaded labels. Null if not loaded */
  guint total_labels; /**< The number of loaded labels */
  guint max_word_length; /**< The max size of labels */

  /* From option4 */
  guint width; /**< Output Video Width */
  guint height; /**< Output Video Height */

  /* From option5 */
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */

  guint max_detection;
} bounding_boxes;

/** @brief Initialize bounding_boxes per mode */
static gboolean
_init_modes (bounding_boxes * bdata)
{
  if (bdata->mode == BOUNDING_BOX_SSD) {
    /* properties_SSD *data = &bdata->ssd; */
    return TRUE;
  }
  return TRUE;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static gboolean
bb_init (GstTensorDec * self)
{
  /** @todo check if we need to ensure plugin_data is not yet allocated */
  bounding_boxes *bdata;
  int i, j, k;
  self->plugin_data = g_new0 (bounding_boxes, 1);

  bdata = self->plugin_data;
  bdata->mode = BOUNDING_BOX_DEFAULT;
  bdata->width = 640;
  bdata->height = 480;
  bdata->i_width = 300;
  bdata->i_height = 300;

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
      val = rasters[ch][j];
      for (k = 0; k < 8; k++) {
        if (val & 0x80)
          singleLineSprite[i][12 - j][k] = PIXEL_VALUE;
        else
          singleLineSprite[i][12 - j][k] = 0;
        val <<= 1;
      }
    }
  }

  /* The default values when the user didn't specify */
  return _init_modes (bdata);
}

/** @brief Free bounding_boxes per mode */
static void
_exit_modes (bounding_boxes * bdata)
{
  if (bdata->mode == BOUNDING_BOX_SSD) {
    /* properties_SSD *data = &bdata->ssd; */
  }
}

/** @brief tensordec-plugin's TensorDecDef callback */
static void
bb_exit (GstTensorDec * self)
{
  bounding_boxes *bdata = self->plugin_data;

  if (bdata->labels) {
    int i;
    for (i = 0; i < bdata->total_labels; i++)
      g_free (bdata->labels[i]);
    g_free (bdata->labels);
  }
  if (bdata->label_path)
    g_free (bdata->label_path);
  _exit_modes (bdata);

  g_free (self->plugin_data);
  self->plugin_data = NULL;
}

/**
 * @brief Load label file into the internal data
 * @param[in/out] data The given ImageLabelData struct.
 */
static void
loadImageLabels (bounding_boxes * data)
{
  FILE *fp;
  int i;

  /* Clean up previously configured data first */
  if (data->labels) {
    for (i = 0; i < data->total_labels; i++)
      g_free (data->labels[i]);
    g_free (data->labels);
  }
  data->labels = NULL;
  data->total_labels = 0;
  data->max_word_length = 0;

  if ((fp = fopen (data->label_path, "r")) != NULL) {
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    gchar *label;

    GList *labels = NULL, *cursor;

    while ((read = getline (&line, &len, fp)) != -1) {
      if (line) {
        if (line[strlen (line) - 1] == '\n') {
          line[strlen (line) - 1] = '\0';
        }
        label = g_strdup ((gchar *) line);
        labels = g_list_append (labels, label);
        free (line);
        if (strlen (label) > data->max_word_length)
          data->max_word_length = strlen (label);
      }
      line = NULL;
      len = 0;
    }

    if (line) {
      free (line);
    }

    /* Flatten labels (GList) into data->labels (array gchar **) */
    data->total_labels = g_list_length (labels);
    data->labels = g_new (gchar *, data->total_labels);
    i = 0;
    cursor = g_list_first (labels);
    for (cursor = labels; cursor != NULL; cursor = cursor->next) {
      data->labels[i] = cursor->data;
      i++;
      g_assert (i <= data->total_labels);
    }
    g_list_free (labels);       /* Do not free its elements */

    fclose (fp);
  } else {
    GST_ERROR ("Cannot load label file %s", data->label_path);
    return;
  }

  GST_INFO ("Loaded image label file successfully. %u labels loaded.",
      data->total_labels);
  return;
}

static gboolean _ssd_loadBoxPrior (bounding_boxes * bdata);

/** @brief configure per-mode option (option3) */
static gboolean
_setOption_mode (bounding_boxes * bdata, const gchar * param)
{
  if (bdata->mode == BOUNDING_BOX_SSD) {
    /* Load prior boxes with the path from option3 */
    properties_SSD *ssd = &bdata->ssd;

    if (ssd->box_prior_path)
      g_free (ssd->box_prior_path);
    ssd->box_prior_path = g_strdup (param);

    if (NULL != ssd->box_prior_path)
      return _ssd_loadBoxPrior (bdata);

  }
  return TRUE;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static gboolean
bb_setOption (GstTensorDec * self, int opNum, const gchar * param)
{
  bounding_boxes *bdata = self->plugin_data;

  if (opNum == 0) {
    /* option1 = Bounding Box Decoding mode */
    bounding_box_modes previous = bdata->mode;
    bdata->mode = find_key_strv (bb_modes, param);

    if (NULL == param || *param == '\0' || bdata->mode < 0)
      bdata->mode = BOUNDING_BOX_DEFAULT;

    if (bdata->mode != previous && bdata->mode != BOUNDING_BOX_UNKNOWN) {
      return _init_modes (bdata);
    }
    return TRUE;

  } else if (opNum == 1) {
    /* option2 = label text file location */
    if (NULL != bdata->label_path)
      g_free (bdata->label_path);
    bdata->label_path = g_strdup (param);

    if (NULL != bdata->label_path)
      loadImageLabels (bdata);

    if (bdata->total_labels > 0)
      return TRUE;
    else
      return FALSE;
      /** @todo Do not die for this */
  } else if (opNum == 2) {
    /* option3 = per-decoding-mode option */
    return _setOption_mode (bdata, param);
  } else if (opNum == 3) {
    /* option4 = output video size (width:height) */
    tensor_dim dim;
    int rank = get_tensor_dimension (param, dim);

    bdata->width = 0;
    bdata->height = 0;
    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR
          ("mode-option-2 of boundingbox is video output dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE;              /* Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING
          ("mode-option-2 of boundingbox is video output dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    bdata->width = dim[0];
    bdata->height = dim[1];
    return TRUE;
  } else if (opNum == 4) {
    /* option5 = input model size (width:height) */
    tensor_dim dim;
    int rank = get_tensor_dimension (param, dim);

    bdata->i_width = 0;
    bdata->i_height = 0;
    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR
          ("mode-option-3 of boundingbox is input video dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE;              /* Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING
          ("mode-option-3 of boundingbox is input video dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    bdata->i_width = dim[0];
    bdata->i_height = dim[1];
    return TRUE;
  }
  /**
   * @todo Accept color / border-width / ... with option-2
   */
  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

/**
 * @brief tensordec-plugin's TensorDecDef callback
 *
 * The first tensor is boxes. BOX_SIZE : 1 : #MaxDetection, ANY-TYPE
 * The second tensor is labels. #MaxLabel : #MaxDetection, ANY-TYPE
 * Both tensors are MANDATORY!
 *
 * If there are third or more tensors, such tensors will be ignored.
 */
static GstCaps *
bb_getOutCaps (GstTensorDec * self, const GstTensorsConfig * config)
{
  /** @todo this is compatible with "SSD" only. expand the capability! */
  bounding_boxes *data = self->plugin_data;
  GstCaps *caps;
  const uint32_t *dim1, *dim2;
  int i;
  gchar *str;
  guint max_detection, max_label;

  g_return_val_if_fail (config != NULL, NULL);
  g_return_val_if_fail (config->info.num_tensors >= 2, NULL);
  if (config->info.num_tensors > 2) {
    GST_WARNING ("tensor-decoder:boundingbox accepts 1 or 2 tensors. "
        "You are wasting the bandwidth by supplying %d tensors.",
        config->info.num_tensors);
  }

  /* Check if the first tensor is compatible */
  dim1 = config->info.info[0].dimension;
  g_return_val_if_fail (dim1[0] == BOX_SIZE, NULL);
  g_return_val_if_fail (dim1[1] == 1, NULL);
  max_detection = dim1[2];
  g_return_val_if_fail (max_detection > 0, NULL);
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim1[i] == 1, NULL);

  /* Check if the second tensor is compatible */
  dim2 = config->info.info[1].dimension;
  max_label = dim2[0];
  g_return_val_if_fail (max_label <= data->total_labels, NULL);
  if (max_label < data->total_labels)
    GST_WARNING
        ("The given tensor (2nd) has max_label (first dimension: %u) smaller than the number of labels in labels file (%s: %u).",
        max_label, data->label_path, data->total_labels);
  g_return_val_if_fail (max_detection == dim2[1], NULL);
  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim2[i] == 1, NULL);

  /* tensor-type of the two tensors shoule be the same */
  g_return_val_if_fail (config->info.info[0].type == config->info.info[1].type,
      NULL);

  /* Check consistency with max_detection */
  if (data->max_detection == 0)
    data->max_detection = max_detection;
  else
    g_return_val_if_fail (max_detection == data->max_detection, NULL);
  if (data->max_detection > DETECTION_MAX) {
    GST_ERROR
        ("Incoming tensor has too large detection-max (3rd rank in tensor_0 and 2nd rank in tensor_1) : %u",
        max_detection);
    return NULL;
  }

  str = g_strdup_printf ("video/x-raw, format = RGBA, " /* Use alpha channel to make the background transparent */
      "width = %u, height = %u"
      /** @todo Configure framerate! */
      , data->width, data->height);
  caps = gst_caps_from_string (str);
  g_free (str);

  return caps;
}

/**
 * @brief Load box-prior data from a file
 * @param[in/out] bdata The internal data.
 * @return TRUE if loaded and configured. FALSE if failed to do so.
 */
static gboolean
_ssd_loadBoxPrior (bounding_boxes * bdata)
{
  FILE *fp;
  properties_SSD *ssd = &bdata->ssd;

  if ((fp = fopen (ssd->box_prior_path, "r")) != NULL) {
    int row;
    int prev_reg = -1;

    for (row = 0; row < BOX_SIZE; row++) {
      int column = 0, registered = 0;
      size_t len = 0;
      char *line = NULL;
      ssize_t read;

      read = getline (&line, &len, fp);

      if (read == -1) {
        GST_ERROR
            ("Decoder/Bound-Box/SSD's box prior data file has rows too short");
        return FALSE;
      }
      if (line) {
        gchar **list = g_strsplit_set (line, " \t,", -1);
        gchar *word;

        while ((word = list[column]) != NULL) {
          column++;

          if (word && *word) {
            if (registered > DETECTION_MAX) {
              GST_WARNING
                  ("Decoder/Bound-Box/SSD's box prior data file has too many priors. %d >= %d",
                  registered, DETECTION_MAX);
              break;
            }
            ssd->box_priors[row][registered] = atof (word);
            registered++;
          }
        }
        g_strfreev (list);
        free (line);
      }
      if (prev_reg != -1 && prev_reg != registered) {
        GST_ERROR
            ("Decoder/Bound-Box/SSD's box prior data file is not consistent.");
        return FALSE;
      }
      prev_reg = registered;
    }

    fclose (fp);
  } else {
    GST_ERROR ("Decoder/Bound-Box/SSD's box prior file cannot be read: %s",
        ssd->box_prior_path);
    return FALSE;
  }
  return TRUE;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static gsize
bb_getTransformSize (GstTensorDec * self, GstCaps * caps,
    gsize size, GstCaps * othercaps, GstPadDirection direction)
{
  return 0;
  /** @todo Use appropriate values */
}

/** @brief Represents a detect object */
typedef struct
{
  gboolean valid;
  int class_id;
  int x;
  int y;
  int width;
  int height;
  gfloat prob;
} detectedObject;

#define DETECTION_THRESHOLD (.5f)
#define Y_SCALE (10.0f)
#define X_SCALE (10.0f)
#define H_SCALE (5.0f)
#define W_SCALE (5.0f)

#define _expit(x) \
    (1.f / (1.f + expf (- ((float)x))))

/**
 * @brief C++-Template-like box location calculation for box-priors
 * @bug This is not macro-argument safe. Use paranthesis!
 * @param[in] bb The configuration, "bounding_boxes"
 * @param[in] index The index (3rd dimension of BOX_SIZE:1:DETECTION_MAX:1)
 * @param[in] boxprior The box prior data from the box file of SSD.
 * @param[in] boxinputptr Cursor pointer of input + byte-per-index * index (box)
 * @param[in] detinputptr Cursor pointer of input + byte-per-index * index (detection)
 * @param[in] result The object returned. (pointer to object)
 */
#define _get_object_i(bb, index, boxprior, boxinputptr, detinputptr, result) \
  do { \
    int c; \
    for (c = 1; c < bb->total_labels; c++) { \
      gfloat score = _expit (detinputptr[c]); \
      if (score >= DETECTION_THRESHOLD) { \
        float ycenter = boxinputptr[0] / Y_SCALE * boxprior[2][index] + boxprior[0][index]; \
        float xcenter = boxinputptr[1] / X_SCALE * boxprior[3][index] + boxprior[1][index]; \
        float h = (float) expf (boxinputptr[2] / H_SCALE) * boxprior[2][index]; \
        float w = (float) expf (boxinputptr[3] / W_SCALE) * boxprior[3][index]; \
        float ymin = ycenter - h / 2.f; \
        float xmin = xcenter - w / 2.f; \
        int x = xmin * bb->i_width; \
        int y = ymin * bb->i_height; \
        int width = w * bb->i_width; \
        int height = h * bb->i_height; \
        result->class_id = c; \
        result->x = MAX (0, x); \
        result->y = MAX (0, y); \
        result->width = width; \
        result->height = height; \
        result->prob = score; \
        result->valid = TRUE; \
        break; \
      } \
    } \
   } while (0);

/**
 * @brief C++-Template-like box location calculation for box-priors
 * @param[in] bb The configuration, "bounding_boxes"
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] index The index (3rd dimension of BOX_SIZE:1:DETECTION_MAX:1)
 * @param[in] boxprior The box prior data from the box file of SSD.
 * @param[in] boxinput Input Tensor Data (Boxes)
 * @param[in] detinput Input Tensor Data (Detection). Null if not available. (numtensor ==1)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects(bb, _type, typename, boxprior, boxinput, detinput, config, results) \
  case typename: \
  { \
    int d; \
    _type * boxinput_ = (_type *) boxinput; \
    gsize boxbpi = config->info.info[0].dimension[0]; \
    _type * detinput_ = (_type *) detinput; \
    gsize detbpi = config->info.info[1].dimension[0]; \
    int num = (DETECTION_MAX > bb->max_detection) ? bb->max_detection : DETECTION_MAX; \
    detectedObject object = { .valid = FALSE, .class_id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .prob = 0 }; \
    for (d = 0; d < num; d++) { \
      _get_object_i (bb, d, boxprior, (boxinput_ + (d * boxbpi)), (detinput_ + (d * detbpi)), (&object)); \
      if (object.valid == TRUE) { \
        g_array_append_val (results, object); \
      } \
    } \
  } \
  break

/** @brief Macro to simplify calling _get_objects */
#define _get_objects_(type, typename) \
  _get_objects (bdata, type, typename, (bdata->ssd.box_priors), (boxes->data), (detections->data), config, results)


/**
 * @brief Compare Function for g_array_sort with detectedObject.
 */
static gint
compare_detection (gconstpointer _a, gconstpointer _b)
{
  const detectedObject *a = _a;
  const detectedObject *b = _b;

  /* Larger comes first */
  return (a->prob > b->prob) ? -1 : ((a->prob == b->prob) ? 0 : 1);
}

/**
 * @brief Calculate the intersected surface
 */
static gfloat
iou (detectedObject * a, detectedObject * b)
{
  int x1 = MAX (a->x, b->x);
  int y1 = MAX (a->y, b->y);
  int x2 = MIN (a->x + a->width, b->x + b->width);
  int y2 = MIN (a->y + a->height, b->y + b->height);
  int w = MAX (0, (x2 - x1 + 1));
  int h = MAX (0, (y2 - y1 + 1));
  float inter = w * h;
  float areaA = a->width * a->height;
  float areaB = b->width * b->height;
  float o = inter / (areaA + areaB - inter);
  return (o >= 0) ? o : 0;
}

#define THRESHOLD_IOU (.5f)
/**
 * @brief Apply NMS to the given results (obejcts[DETECTION_MAX])
 * @param[in/out] results The results to be filtered with nms
 */
static void
nms (GArray * results)
{
  guint boxes_size;
  guint i, j;

  g_array_sort (results, compare_detection);
  boxes_size = results->len;

  for (i = 0; i < boxes_size; i++) {
    detectedObject *a = &g_array_index (results, detectedObject, i);
    if (a->valid == TRUE) {
      for (j = i + 1; j < boxes_size; j++) {
        detectedObject *b = &g_array_index (results, detectedObject, j);
        if (b->valid == TRUE) {
          if (iou (a, b) > THRESHOLD_IOU) {
            b->valid = FALSE;
          }
        }
      }
    }
  }

  i = 0;
  do {
    detectedObject *a = &g_array_index (results, detectedObject, i);
    if (a->valid == FALSE)
      g_array_remove_index (results, i);
    else
      i++;
  } while (i < results->len);

}

/**
 * @brief Draw with the given results (obejcts[DETECTION_MAX]) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bouding-box internal data.
 * @param[in] results The final results to be drawn.
 */
static void
draw (GstMapInfo * out_info, bounding_boxes * bdata, GArray * results)
{
  uint32_t *frame = (uint32_t *) out_info->data;        /* Let's draw per pixel (4bytes) */
  int i;

  for (i = 0; i < results->len; i++) {
    int x1, x2, y1, y2;         /* Box positions on the output surface */
    int j;
    uint32_t *pos1, *pos2;
    const gchar *label;
    int label_len;
    detectedObject *a = &g_array_index (results, detectedObject, i);

    /* 1. Draw Boxes */
    x1 = (bdata->width * a->x) / bdata->i_width;
    x2 = MIN (bdata->width - 1,
        (bdata->width * (a->x + a->width)) / bdata->i_width);

    y1 = (bdata->height * a->y) / bdata->i_height;
    y2 = MIN (bdata->height - 1,
        (bdata->height * (a->y + a->height)) / bdata->i_height);

    /* 1-1. Horizontal */
    pos1 = &frame[y1 * bdata->width + x1];
    pos2 = &frame[y2 * bdata->width + x1];
    for (j = x1; j <= x2; j++) {
      *pos1 = PIXEL_VALUE;
      *pos2 = PIXEL_VALUE;
      pos1++;
      pos2++;
    }

    /* 1-2. Vertical */
    pos1 = &frame[(y1 + 1) * bdata->width + x1];
    pos2 = &frame[(y1 + 1) * bdata->width + x2];
    for (j = y1 + 1; j < y2; j++) {
      *pos1 = PIXEL_VALUE;
      *pos2 = PIXEL_VALUE;
      pos1 += bdata->width;
      pos2 += bdata->width;
    }

    /* 2. Write Labels */
    g_assert (a->class_id > 0 && a->class_id < bdata->total_labels);
    label = bdata->labels[a->class_id];
    label_len = strlen (label);
    /* x1 is the same: x1 = MAX (0, (bdata->width * a->x) / bdata->i_width); */
    y1 = MAX (0, (y1 - 14));
    pos1 = &frame[y1 * bdata->width + x1];
    for (j = 0; j < label_len; j++) {
      unsigned int char_index = label[j];
      if ((x1 + 8) > bdata->width)
        break;                  /* Stop drawing if it may overfill */
      pos2 = pos1;
      for (y2 = 0; y2 < 13; y2++) {
        /* 13 : character height */
        for (x2 = 0; x2 < 8; x2++) {
          /* 8: character width */
          *(pos2 + x2) = singleLineSprite[char_index][y2][x2];
        }
        pos2 += bdata->width;
      }
      x1 += 9;
      pos1 += 9;                /* charater width + 1px */
    }
  }
}

/** @brief tensordec-plugin's TensorDecDef callback */
static GstFlowReturn
bb_decode (GstTensorDec * self, const GstTensorMemory * input,
    GstBuffer * outbuf)
{
  bounding_boxes *bdata = self->plugin_data;
  const gsize size = bdata->width * bdata->height * 4;  /* RGBA */
  GstMapInfo out_info;
  GstMemory *out_mem;
  const GstTensorsConfig *config = &self->tensor_config;

  const int num_tensors = config->info.num_tensors;
  const GstTensorMemory *boxes, *detections = NULL;
  GArray *results =
      g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), 100);
  /**
   * @todo 100 is a heuristic number of objects in a picture frame
   *       We may have better "heuristics" than this.
   *       For the sake of performance, don't make it too small.
   */

  g_assert (outbuf);
  g_assert (num_tensors >= 2);

  boxes = &input[0];
  if (num_tensors >= 2)
    detections = &input[1];

  /* Ensure we have outbuf properly allocated */
  if (gst_buffer_get_size (outbuf) == 0) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  g_assert (gst_memory_map (out_mem, &out_info, GST_MAP_WRITE));

  /** reset the buffer with alpha 0 / black */
  memset (out_info.data, 0, size);

  switch (config->info.info[0].type) {
      _get_objects_ (uint8_t, _NNS_UINT8);
      _get_objects_ (int8_t, _NNS_INT8);
      _get_objects_ (uint16_t, _NNS_UINT16);
      _get_objects_ (int16_t, _NNS_INT16);
      _get_objects_ (uint32_t, _NNS_UINT32);
      _get_objects_ (int32_t, _NNS_INT32);
      _get_objects_ (uint64_t, _NNS_UINT64);
      _get_objects_ (int64_t, _NNS_INT64);
      _get_objects_ (float, _NNS_FLOAT32);
      _get_objects_ (double, _NNS_FLOAT64);
    default:
      g_assert (0);
  }

  nms (results);
  draw (&out_info, bdata, results);
  g_array_free (results, TRUE);

  gst_memory_unmap (out_mem, &out_info);

  if (gst_buffer_get_size (outbuf) == 0)
    gst_buffer_append_memory (outbuf, out_mem);

  return GST_FLOW_OK;
}

/** @brief Image Labeling tensordec-plugin TensorDecDef instance */
static TensorDecDef boundingBox = {
  .modename = "bounding_boxes",
  .type = OUTPUT_VIDEO,
  .init = bb_init,
  .exit = bb_exit,
  .setOption = bb_setOption,
  .getOutCaps = bb_getOutCaps,
  .getTransformSize = bb_getTransformSize,
  .decode = bb_decode,
};

/** @brief Initialize this object for tensordec-plugin */
__attribute__ ((constructor))
     void init_bb (void)
{
  tensordec_probe (&boundingBox);
}

/** @brief Destruct this object for tensordec-plugin */
__attribute__ ((destructor))
     void fini_bb (void)
{
  tensordec_exit (boundingBox.modename);
}
