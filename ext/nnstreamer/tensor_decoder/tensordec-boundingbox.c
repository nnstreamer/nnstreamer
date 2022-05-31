/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "bounding boxes"
 * Copyright (C) 2018 Samsung Electronics Co. Ltd.
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 * Copyright 2021 NXP
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
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
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 * option1: Decoder mode of bounding box.
 *          Available: yolov5
 *                     mobilenet-ssd (single shot multibox detector with priors.)
 *                     mobilenet-ssd-postprocess
 *                     ov-person-detection
 *                     tf-ssd (deprecated, recommend to use mobilenet-ssd-postprocess)
 *                     tflite-ssd (deprecated, recommend to use mobilenet-ssd)
 * option2: Location of label file
 *          This is independent from option1
 * option3: Any option1-dependent values
 *          !!This depends on option1 values!!
 *          for mobilenet-ssd mode:
 *            The option3 definition scheme is, in order, the following:
 *                - box priors location file (mandatory)
 *                - Detection threshold (optional, default set to 0.5)
 *                - Y box scale (optional, default set to 10.0)
 *                - X box scale (optional, default set to 10.0)
 *                - h box scale (optional, default set to 5.0)
 *                - w box scale (optional, default set to 5.0)
 *                - IOU box valid threshold (optional, default set to 0.5)
 *            The default parameters value could be set in the following ways:
 *            option3=box-priors.txt:0.5:10.0:10.0:5.0:5.0:0.5
 *            option3=box-priors.txt
 *            option3=box-priors.txt::::::
 *
 *            It's possible to set only few values, using the default values for
 *            those not specified through the command line.
 *            You could specify respectively the detection and IOU thresholds to 0.65
 *            and 0.6 with the option3 parameter as follow:
 *            option3=box-priors.txt:0.65:::::0.6
 *          for mobilenet-ssd-postprocess mode:
 *            The option3 is required to have 5 integer numbers, which tell
 *            the tensor-dec how to interpret the given tensor inputs.
 *            The first 4 numbers separated by colon, ':', designate which
 *            are location:class:score:number of the tensors.
 *            The last number separated by comma, ',' from the first 4 numbers
 *            designate the threshold in percent.
 *            In other words, "option3=%i:%i:%i:%i,%i".
 *          for mp-palm-detection mode:
 *            The option3 is required to have 5 float numbers, as following
 *                - box score threshold (mandatory)
 *                - number of layers for anchor generation (optional, default set to 4)
 *                - minimum scale factor for anchor generation (optional, default set to 1.0)
 *                - maximum scale factor for anchor generation (optional, default set to 1.0)
 *                - X offset (optional, default set to 0.5)
 *                - Y offset (optional, default set to 0.5)
 *                - strides for each layer for anchor generation (optional, default set to 8:16:16:16)
 *            The default parameter value could be set in the following ways:
 *            option3=0.5
 *            option3=0.5:4:0.2:0.8
 *            option3=0.5:4:1.0:1.0:0.5:0.5:8:16:16:16
 *
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

/** @todo _GNU_SOURCE fix build warning expf (nested-externs). remove this later. */
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
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "tensordecutil.h"

void init_bb (void) __attribute__ ((constructor));
void fini_bb (void) __attribute__ ((destructor));

/* font.c */
extern uint8_t rasters[][13];

#define BOX_SIZE                                (4)
#define MOBILENET_SSD_DETECTION_MAX             (1917)
#define MOBILENET_SSD_MAX_TENSORS               (2U)
#define MOBILENET_SSD_PP_DETECTION_MAX          (100)
#define MOBILENET_SSD_PP_MAX_TENSORS            (4U)
#define OV_PERSON_DETECTION_MAX                 (200U)
#define OV_PERSON_DETECTION_MAX_TENSORS         (1U)
#define OV_PERSON_DETECTION_SIZE_DETECTION_DESC (7)
#define OV_PERSON_DETECTION_CONF_THRESHOLD      (0.8)
#define YOLOV5_DETECTION_NUM_INFO               (5)
#define YOLOV5_DETECTION_CONF_THRESHOLD         (0.3)
#define YOLOV5_DETECTION_IOU_THRESHOLD          (0.6)
#define PIXEL_VALUE                             (0xFF0000FF)    /* RED 100% in RGBA */
#define MP_PALM_DETECTION_INFO_SIZE             (18)
#define MP_PALM_DETECTION_MAX_TENSORS           (2U)
#define MP_PALM_DETECTION_DETECTION_MAX         (2016)

/**
 * @todo Fill in the value at build time or hardcode this. It's const value
 * @brief The bitmap of characters
 * [Character (ASCII)][Height][Width]
 */
static singleLineSprite_t singleLineSprite;

/**
 * @brief There can be different schemes for bounding boxes.
 */
typedef enum
{
  MOBILENET_SSD_BOUNDING_BOX = 0,
  MOBILENET_SSD_PP_BOUNDING_BOX = 1,
  OV_PERSON_DETECTION_BOUNDING_BOX = 2,
  OV_FACE_DETECTION_BOUNDING_BOX = 3,

  /* the modes started with 'OLDNAME_' is for backward compatibility. */
  OLDNAME_MOBILENET_SSD_BOUNDING_BOX = 4,
  OLDNAME_MOBILENET_SSD_PP_BOUNDING_BOX = 5,

  YOLOV5_BOUNDING_BOX = 6,

  MP_PALM_DETECTION_BOUNDING_BOX = 7,

  BOUNDING_BOX_UNKNOWN,
} bounding_box_modes;

/**
 * @brief MOBILENET SSD PostProcess Output tensor feature mapping.
 */
typedef enum
{
  MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS = 0,
  MOBILENET_SSD_PP_BBOX_IDX_CLASSES = 1,
  MOBILENET_SSD_PP_BBOX_IDX_SCORES = 2,
  MOBILENET_SSD_PP_BBOX_IDX_NUM = 3,
  MOBILENET_SSD_PP_BBOX_IDX_UNKNOWN
} mobilenet_ssd_pp_bbox_idx_t;

/**
 * @brief List of bounding-box decoding schemes in string
 */
static const char *bb_modes[] = {
  [MOBILENET_SSD_BOUNDING_BOX] = "mobilenet-ssd",
  [MOBILENET_SSD_PP_BOUNDING_BOX] = "mobilenet-ssd-postprocess",
  [OV_PERSON_DETECTION_BOUNDING_BOX] = "ov-person-detection",
  [OV_FACE_DETECTION_BOUNDING_BOX] = "ov-face-detection",
  [OLDNAME_MOBILENET_SSD_BOUNDING_BOX] = "tflite-ssd",
  [OLDNAME_MOBILENET_SSD_PP_BOUNDING_BOX] = "tf-ssd",
  [YOLOV5_BOUNDING_BOX] = "yolov5",
  [MP_PALM_DETECTION_BOUNDING_BOX] = "mp-palm-detection",
  NULL,
};

/**
 * @brief Data structure for SSD bounding box info for mobilenet ssd model.
 */
typedef struct
{
  /* From option3, box prior data */
  char *box_prior_path; /**< Box Prior file path */
  gfloat box_priors[BOX_SIZE][MOBILENET_SSD_DETECTION_MAX + 1]; /** loaded box prior */
#define MOBILENET_SSD_PARAMS_THRESHOLD_IDX 0
#define MOBILENET_SSD_PARAMS_Y_SCALE_IDX 1
#define MOBILENET_SSD_PARAMS_X_SCALE_IDX 2
#define MOBILENET_SSD_PARAMS_H_SCALE_IDX 3
#define MOBILENET_SSD_PARAMS_W_SCALE_IDX 4
#define MOBILENET_SSD_PARAMS_IOU_THRESHOLD_IDX 5
#define MOBILENET_SSD_PARAMS_MAX 6
  gfloat params[MOBILENET_SSD_PARAMS_MAX]; /** Post Processing parameters */
  gfloat sigmoid_threshold; /** Inverse value of valid detection threshold in sigmoid domain */
} properties_MOBILENET_SSD;

/**
 * @brief Data structure for SSD bounding box info for mobilenet ssd postprocess model.
 */
typedef struct
{
  /* From option3, output tensor mapping */
  gint tensor_mapping[MOBILENET_SSD_PP_MAX_TENSORS];    /* Output tensor index mapping */
  gfloat threshold;             /* Detection threshold */
} properties_MOBILENET_SSD_PP;

/**
 * @brief anchor data
 */
typedef struct {
  float x_center;
  float y_center;
  float w;
  float h;
} anchor;

/**
 * @brief Data structure for bounding box info for mediapipe palm detection model.
 */
typedef struct
{
  /* From option3, anchor data */
#define MP_PALM_DETECTION_PARAMS_STRIDE_SIZE 8
#define MP_PALM_DETECTION_PARAMS_MAX 13

  gint num_layers; /** Number of stride layers */
  gfloat min_scale; /** Minimum scale */
  gfloat max_scale; /** Maximum scale */
  gfloat offset_x; /** anchor X offset */
  gfloat offset_y; /** anchor Y offset */
  gint strides[MP_PALM_DETECTION_PARAMS_STRIDE_SIZE]; /** Stride data for each layers */
  gfloat min_score_threshold; /** minimum threshold of score */

  GArray *anchors;

} properties_MP_PALM_DETECTION;

/**
 * @brief Data structure for bounding box info.
 */
typedef struct
{
  bounding_box_modes mode; /**< The bounding box decoding mode */

  union
  {
    properties_MOBILENET_SSD mobilenet_ssd; /**< Properties for mobilenet_ssd configured by option 1 + 3 */
    properties_MOBILENET_SSD_PP mobilenet_ssd_pp; /**< mobilenet_ssd_pp mode properties configuration settings */
  };

  properties_MP_PALM_DETECTION mp_palm_detection; /**< mp_palm_detection mode properties configuration settings */

  /* From option2 */
  imglabel_t labeldata;
  char *label_path;

  /* From option4 */
  guint width; /**< Output Video Width */
  guint height; /**< Output Video Height */

  /* From option5 */
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */

  guint max_detection;
  gboolean flag_use_label;
} bounding_boxes;

/** @brief check the mode is mobilenet-ssd */
static inline gboolean
_check_mode_is_mobilenet_ssd (bounding_box_modes mode)
{
  gboolean ret = FALSE;
  if (mode == MOBILENET_SSD_BOUNDING_BOX
      || mode == OLDNAME_MOBILENET_SSD_BOUNDING_BOX) {
    ret = TRUE;
  }
  return ret;
}

/** @brief check the mode is mobilenet-ssd-post-processing */
static inline gboolean
_check_mode_is_mobilenet_ssd_pp (bounding_box_modes mode)
{
  gboolean ret = FALSE;
  if (mode == MOBILENET_SSD_PP_BOUNDING_BOX
      || mode == OLDNAME_MOBILENET_SSD_PP_BOUNDING_BOX) {
    ret = TRUE;
  }
  return ret;
}

/** @brief check the mode is mp-palm-detection */
static inline gboolean
_check_mode_is_mp_palm_detection (bounding_box_modes mode)
{
  gboolean ret = FALSE;
  if (mode == MP_PALM_DETECTION_BOUNDING_BOX) {
    ret = TRUE;
  }
  return ret;
}

/** @brief Helper to retrieve tensor index by feature */
static inline int
_get_mobilenet_ssd_pp_tensor_idx (bounding_boxes * bdata,
    mobilenet_ssd_pp_bbox_idx_t idx)
{
  return bdata->mobilenet_ssd_pp.tensor_mapping[idx];
}

/** @brief Helper to retrieve object detection confidence threshold */
static inline float
_get_mobilenet_ssd_pp_threshold (bounding_boxes * bdata)
{
  return bdata->mobilenet_ssd_pp.threshold;
}

/** @brief Mathematic inverse of sigmoid function, aka logit */
static float
logit (float x)
{
  if (x <= 0.0f)
    return -INFINITY;

  if (x >= 1.0f)
    return INFINITY;

  return log (x / (1.0 - x));
}

/** @brief Initialize bounding_boxes per mode */
static int
_init_modes (bounding_boxes * bdata)
{
  if (_check_mode_is_mobilenet_ssd (bdata->mode)) {
    properties_MOBILENET_SSD *data = &bdata->mobilenet_ssd;
#define DETECTION_THRESHOLD (.5f)
#define THRESHOLD_IOU (.5f)
#define Y_SCALE (10.0f)
#define X_SCALE (10.0f)
#define H_SCALE (5.0f)
#define W_SCALE (5.0f)

    data->params[MOBILENET_SSD_PARAMS_THRESHOLD_IDX] = DETECTION_THRESHOLD;
    data->params[MOBILENET_SSD_PARAMS_Y_SCALE_IDX] = Y_SCALE;
    data->params[MOBILENET_SSD_PARAMS_X_SCALE_IDX] = X_SCALE;
    data->params[MOBILENET_SSD_PARAMS_H_SCALE_IDX] = H_SCALE;
    data->params[MOBILENET_SSD_PARAMS_W_SCALE_IDX] = W_SCALE;
    data->params[MOBILENET_SSD_PARAMS_IOU_THRESHOLD_IDX] = THRESHOLD_IOU;
    data->sigmoid_threshold = logit (DETECTION_THRESHOLD);

    return TRUE;
  } else if (_check_mode_is_mobilenet_ssd_pp (bdata->mode)) {
    properties_MOBILENET_SSD_PP *data = &bdata->mobilenet_ssd_pp;

#define MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS_DEFAULT 3
#define MOBILENET_SSD_PP_BBOX_IDX_CLASSES_DEFAULT 1
#define MOBILENET_SSD_PP_BBOX_IDX_SCORES_DEFAULT 2
#define MOBILENET_SSD_PP_BBOX_IDX_NUM_DEFAULT 0
#define MOBILENET_SSD_PP_BBOX_THRESHOLD_DEFAULT G_MINFLOAT

    data->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS] =
        MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS_DEFAULT;
    data->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_CLASSES] =
        MOBILENET_SSD_PP_BBOX_IDX_CLASSES_DEFAULT;
    data->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_SCORES] =
        MOBILENET_SSD_PP_BBOX_IDX_SCORES_DEFAULT;
    data->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_NUM] =
        MOBILENET_SSD_PP_BBOX_IDX_NUM_DEFAULT;
    data->threshold = MOBILENET_SSD_PP_BBOX_THRESHOLD_DEFAULT;

    return TRUE;
  } else if (_check_mode_is_mp_palm_detection (bdata->mode)) {
    properties_MP_PALM_DETECTION *data = &bdata->mp_palm_detection;

#define MP_PALM_DETECTION_NUM_LAYERS_DEFAULT (4)
#define MP_PALM_DETECTION_MIN_SCALE_DEFAULT (1.0)
#define MP_PALM_DETECTION_MAX_SCALE_DEFAULT (1.0)
#define MP_PALM_DETECTION_OFFSET_X_DEFAULT (0.5)
#define MP_PALM_DETECTION_OFFSET_Y_DEFAULT (0.5)
#define MP_PALM_DETECTION_STRIDE_0_DEFAULT (8)
#define MP_PALM_DETECTION_STRIDE_1_DEFAULT (16)
#define MP_PALM_DETECTION_STRIDE_2_DEFAULT (16)
#define MP_PALM_DETECTION_STRIDE_3_DEFAULT (16)
#define MP_PALM_DETECTION_MIN_SCORE_THRESHOLD_DEFAULT (0.5)

    data->num_layers = MP_PALM_DETECTION_NUM_LAYERS_DEFAULT;
    data->min_scale = MP_PALM_DETECTION_MIN_SCALE_DEFAULT;
    data->max_scale = MP_PALM_DETECTION_MAX_SCALE_DEFAULT;
    data->offset_x = MP_PALM_DETECTION_OFFSET_X_DEFAULT;
    data->offset_y = MP_PALM_DETECTION_OFFSET_Y_DEFAULT;
    data->strides[0] = MP_PALM_DETECTION_STRIDE_0_DEFAULT;
    data->strides[1] = MP_PALM_DETECTION_STRIDE_1_DEFAULT;
    data->strides[2] = MP_PALM_DETECTION_STRIDE_2_DEFAULT;
    data->strides[3] = MP_PALM_DETECTION_STRIDE_3_DEFAULT;
    data->min_score_threshold = MP_PALM_DETECTION_MIN_SCORE_THRESHOLD_DEFAULT;
    data->anchors = g_array_new(FALSE, TRUE, sizeof(anchor));

    return TRUE;
  }
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
bb_init (void **pdata)
{
  /** @todo check if we need to ensure plugin_data is not yet allocated */
  bounding_boxes *bdata;

  bdata = *pdata = g_new0 (bounding_boxes, 1);
  if (bdata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  bdata->mode = BOUNDING_BOX_UNKNOWN;
  bdata->width = 0;
  bdata->height = 0;
  bdata->i_width = 0;
  bdata->i_height = 0;
  bdata->flag_use_label = FALSE;

  initSingleLineSprite (singleLineSprite, rasters, PIXEL_VALUE);

  /* The default values when the user didn't specify */
  return _init_modes (bdata);
}

/** @brief Free bounding_boxes per mode */
static void
_exit_modes (bounding_boxes * bdata)
{
  if (_check_mode_is_mobilenet_ssd (bdata->mode)) {
    /* properties_MOBILENET_SSD *data = &bdata->mobilenet_ssd; */
  } else if (_check_mode_is_mobilenet_ssd_pp (bdata->mode)) {
    /* post processing */
  }
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
bb_exit (void **pdata)
{
  bounding_boxes *bdata = *pdata;

  _free_labels (&bdata->labeldata);

  if (bdata->label_path)
    g_free (bdata->label_path);
  _exit_modes (bdata);

  g_free (*pdata);
  *pdata = NULL;
}

/**
 * @brief Load box-prior data from a file
 * @param[in/out] bdata The internal data.
 * @return TRUE if loaded and configured. FALSE if failed to do so.
 */
static int
_mobilenet_ssd_loadBoxPrior (bounding_boxes * bdata)
{
  properties_MOBILENET_SSD *mobilenet_ssd = &bdata->mobilenet_ssd;
  gboolean failed = FALSE;
  GError *err = NULL;
  gchar **priors;
  gchar *line = NULL;
  gchar *contents = NULL;
  guint row;
  gint prev_reg = -1;

  /* Read file contents */
  if (!g_file_get_contents (mobilenet_ssd->box_prior_path, &contents, NULL,
          &err)) {
    GST_ERROR ("Decoder/Bound-Box/SSD's box prior file %s cannot be read: %s",
        mobilenet_ssd->box_prior_path, err->message);
    g_clear_error (&err);
    return FALSE;
  }

  priors = g_strsplit (contents, "\n", -1);
  /* If given prior file is inappropriate, report back to tensor-decoder */
  if (g_strv_length (priors) < BOX_SIZE) {
    ml_loge ("The given prior file, %s, should have at least %d lines.\n",
        mobilenet_ssd->box_prior_path, BOX_SIZE);
    failed = TRUE;
    goto error;
  }

  for (row = 0; row < BOX_SIZE; row++) {
    gint column = 0, registered = 0;

    line = priors[row];
    if (line) {
      gchar **list = g_strsplit_set (line, " \t,", -1);
      gchar *word;

      while ((word = list[column]) != NULL) {
        column++;

        if (word && *word) {
          if (registered > MOBILENET_SSD_DETECTION_MAX) {
            GST_WARNING
                ("Decoder/Bound-Box/SSD's box prior data file has too many priors. %d >= %d",
                registered, MOBILENET_SSD_DETECTION_MAX);
            break;
          }
          mobilenet_ssd->box_priors[row][registered] =
              (gfloat) g_ascii_strtod (word, NULL);
          registered++;
        }
      }

      g_strfreev (list);
    }

    if (prev_reg != -1 && prev_reg != registered) {
      GST_ERROR
          ("Decoder/Bound-Box/SSD's box prior data file is not consistent.");
      failed = TRUE;
      break;
    }
    prev_reg = registered;
  }

error:
  g_strfreev (priors);
  g_free (contents);
  return !failed;
}


/**
 * @brief Calculate anchor scale
 */
static gfloat
_calculate_scale (float min_scale, float max_scale, int stride_index, int num_strides)
{
  if (num_strides == 1) {
    return (min_scale + max_scale) * 0.5f;
  } else {
    return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
  }
}

/**
 * @brief Generate anchor information
 */
static void
_mp_palm_detection_generate_anchors (properties_MP_PALM_DETECTION *palm_detection)
{
  int layer_id = 0;
  int strides[MP_PALM_DETECTION_PARAMS_STRIDE_SIZE];
  int idx;
  guint i;

  gint num_layers = palm_detection->num_layers;
  gfloat offset_x = palm_detection->offset_x;
  gfloat offset_y = palm_detection->offset_y;

  for (idx = 0; idx < num_layers; idx++) {
    strides[idx] = palm_detection->strides[idx];
  }

  while (layer_id < num_layers) {
    GArray *aspect_ratios = g_array_new(FALSE, TRUE, sizeof(gfloat));
    GArray *scales = g_array_new(FALSE, TRUE, sizeof(gfloat));
    GArray *anchor_height = g_array_new(FALSE, TRUE, sizeof(gfloat));
    GArray *anchor_width = g_array_new(FALSE, TRUE, sizeof(gfloat));

    int last_same_stride_layer = layer_id;

    while (last_same_stride_layer < num_layers
           && strides[last_same_stride_layer] == strides[layer_id]) {
      gfloat scale;
      gfloat ratio = 1.0f;
      g_array_append_val(aspect_ratios, ratio);
      g_array_append_val(aspect_ratios, ratio);
      scale = _calculate_scale(palm_detection->min_scale, palm_detection->max_scale,
                                     last_same_stride_layer, num_layers);
      g_array_append_val(scales, scale);
      scale = _calculate_scale(palm_detection->min_scale, palm_detection->max_scale,
                                     last_same_stride_layer + 1, num_layers);
      g_array_append_val(scales, scale);
      last_same_stride_layer++;
    }

    for (i = 0; i < aspect_ratios->len; ++i) {
      const float ratio_sqrts = sqrt(g_array_index (aspect_ratios, gfloat, i));
      const gfloat sc = g_array_index (scales, gfloat, i);
      gfloat anchor_height_ = sc / ratio_sqrts;
      gfloat anchor_width_ = sc * ratio_sqrts;
      g_array_append_val(anchor_height, anchor_height_);
      g_array_append_val(anchor_width, anchor_width_);
    }

    {
      int feature_map_height = 0;
      int feature_map_width = 0;
      int x, y;
      int anchor_id;

      const int stride = strides[layer_id];
      feature_map_height = ceil(1.0f * 192 / stride);
      feature_map_width = ceil(1.0f * 192 / stride);

      for (y = 0; y < feature_map_height; ++y) {
        for (x = 0; x < feature_map_width; ++x) {
          for (anchor_id = 0; anchor_id < (int)aspect_ratios->len; ++anchor_id) {
            const float x_center = (x + offset_x) * 1.0f / feature_map_width;
            const float y_center = (y + offset_y) * 1.0f / feature_map_height;

            const anchor a = {.x_center = x_center, .y_center = y_center,
              .w = g_array_index (anchor_width, gfloat, anchor_id), .h = g_array_index (anchor_height, gfloat, anchor_id)};
            g_array_append_val(palm_detection->anchors, a);
          }
        }
      }
      layer_id = last_same_stride_layer;
    }

    g_array_free(aspect_ratios, FALSE);
  }
}

#define mp_palm_detection_option(option, type, idx) \
    if (noptions > idx) option = (type)g_strtod (options[idx], NULL)


/** @brief configure per-mode option (option3) */
static int
_setOption_mode (bounding_boxes * bdata, const char *param)
{
  if (_check_mode_is_mobilenet_ssd (bdata->mode)) {
    /* Load prior boxes with the path from option3 */
    properties_MOBILENET_SSD *mobilenet_ssd = &bdata->mobilenet_ssd;
    gchar **options;
    int noptions, idx;
    int ret = 1;

    options = g_strsplit (param, ":", -1);
    noptions = g_strv_length (options);

    if (noptions > (MOBILENET_SSD_PARAMS_MAX + 1))
      noptions = MOBILENET_SSD_PARAMS_MAX + 1;

    if (mobilenet_ssd->box_prior_path)
      g_free (mobilenet_ssd->box_prior_path);

    mobilenet_ssd->box_prior_path = g_strdup (options[0]);

    if (NULL != mobilenet_ssd->box_prior_path) {
      ret = _mobilenet_ssd_loadBoxPrior (bdata);
      if (ret == 0)
        goto exit_mobilenet_ssd;
    }

    for (idx = 1; idx < noptions; idx++) {
      if (strlen (options[idx]) == 0)
        continue;
      mobilenet_ssd->params[idx - 1] = strtod (options[idx], NULL);
    }

    mobilenet_ssd->sigmoid_threshold =
        logit (mobilenet_ssd->params[MOBILENET_SSD_PARAMS_THRESHOLD_IDX]);

  exit_mobilenet_ssd:
    g_strfreev (options);
    return ret;

  } else if (_check_mode_is_mobilenet_ssd_pp (bdata->mode)) {
    properties_MOBILENET_SSD_PP *mobilenet_ssd_pp = &bdata->mobilenet_ssd_pp;
    int threshold_percent;
    int ret = sscanf (param,
        "%i:%i:%i:%i,%i",
        &mobilenet_ssd_pp->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS],
        &mobilenet_ssd_pp->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_CLASSES],
        &mobilenet_ssd_pp->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_SCORES],
        &mobilenet_ssd_pp->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_NUM],
        &threshold_percent);

    if ((ret == EOF) || (ret < 5)) {
      GST_ERROR
          ("Invalid options, must be \"locations idx:classes idx:scores idx:num idx,threshold\"");
      return FALSE;
    }

    GST_INFO ("MOBILENET SSD POST PROCESS output tensors mapping: "
        "locations idx (%d), classes idx (%d), scores idx (%d), num detections idx (%d)",
        mobilenet_ssd_pp->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS],
        mobilenet_ssd_pp->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_CLASSES],
        mobilenet_ssd_pp->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_SCORES],
        mobilenet_ssd_pp->tensor_mapping[MOBILENET_SSD_PP_BBOX_IDX_NUM]
        );

    if ((threshold_percent > 100) || (threshold_percent < 0)) {
      GST_ERROR
          ("Invalid MOBILENET SSD POST PROCESS threshold detection (%i), must be in range [0 100]",
          threshold_percent);
    } else {
      mobilenet_ssd_pp->threshold = threshold_percent / 100.0;
    }

    GST_INFO ("MOBILENET SSD POST PROCESS object detection threshold: %.2f",
        mobilenet_ssd_pp->threshold);
  } else if (_check_mode_is_mp_palm_detection (bdata->mode)) {
    /* Load palm detection info from option3 */
    properties_MP_PALM_DETECTION *palm_detection = &bdata->mp_palm_detection;
    gchar **options;
    int noptions, idx;
    int ret = TRUE;

    options = g_strsplit (param, ":", -1);
    noptions = g_strv_length (options);

    if (noptions > MP_PALM_DETECTION_PARAMS_MAX) {
      GST_ERROR
          ("Invalid MP PALM DETECTION PARAM length: %d", noptions);
      ret = FALSE;
      goto exit_mp_palm_detection;
    }

    mp_palm_detection_option (palm_detection->min_score_threshold, gfloat, 0);
    mp_palm_detection_option (palm_detection->num_layers, gint, 1);
    mp_palm_detection_option (palm_detection->min_scale, gfloat, 2);
    mp_palm_detection_option (palm_detection->max_scale, gfloat, 3);
    mp_palm_detection_option (palm_detection->offset_x, gfloat, 4);
    mp_palm_detection_option (palm_detection->offset_y, gfloat, 5);

    for (idx = 6; idx < palm_detection->num_layers + 6; idx++) {
      mp_palm_detection_option (palm_detection->strides[idx - 6], gint, idx);
    }

    _mp_palm_detection_generate_anchors(palm_detection);

  exit_mp_palm_detection:
    g_strfreev (options);
    return ret;
  }

  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
bb_setOption (void **pdata, int opNum, const char *param)
{
  bounding_boxes *bdata = *pdata;
  if (opNum == 0) {
    /* option1 = Bounding Box Decoding mode */
    bounding_box_modes previous = bdata->mode;
    bdata->mode = find_key_strv (bb_modes, param);

    if (NULL == param || *param == '\0') {
      GST_ERROR ("Please set the valid mode at option1");
      return FALSE;
    }

    if (bdata->mode != previous && bdata->mode != BOUNDING_BOX_UNKNOWN) {
      return _init_modes (bdata);
    }
    return TRUE;

  } else if (opNum == 1) {
    /* option2 = label text file location */

    if (bdata->mode == MP_PALM_DETECTION_BOUNDING_BOX) {
      /* palm detection does not need label information */
      return TRUE;
    }

    if (NULL != bdata->label_path)
      g_free (bdata->label_path);
    bdata->label_path = g_strdup (param);

    if (NULL != bdata->label_path)
      loadImageLabels (bdata->label_path, &bdata->labeldata);

    if (bdata->labeldata.total_labels > 0)
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
    int rank = gst_tensor_parse_dimension (param, dim);

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
    int rank = gst_tensor_parse_dimension (param, dim);

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
 * @brief check the num_tensors is valid
*/
static int
_check_tensors (const GstTensorsConfig * config, const unsigned int limit)
{
  unsigned int i;
  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (config->info.num_tensors >= limit, FALSE);
  if (config->info.num_tensors > limit) {
    GST_WARNING ("tensor-decoder:boundingbox accepts %d or less tensors. "
        "You are wasting the bandwidth by supplying %d tensors.",
        limit, config->info.num_tensors);
  }

  /* tensor-type of the tensors shoule be the same */
  for (i = 1; i < config->info.num_tensors; ++i) {
    g_return_val_if_fail (config->info.info[i - 1].type ==
        config->info.info[i].type, FALSE);
  }
  return TRUE;
}

/**
 * @brief check the label relevant properties are valid
*/
static gboolean
_check_label_props (bounding_boxes * data)
{
  if ((!data->label_path) || (!data->labeldata.labels) ||
      (data->labeldata.total_labels <= 0))
    return FALSE;
  return TRUE;
}

/**
 * @brief set the max_detection
*/
static int
_set_max_detection (bounding_boxes * data, const guint max_detection,
    const unsigned int limit)
{
  /* Check consistency with max_detection */
  if (data->max_detection == 0)
    data->max_detection = max_detection;
  else
    g_return_val_if_fail (max_detection == data->max_detection, FALSE);

  if (data->max_detection > limit) {
    GST_ERROR
        ("Incoming tensor has too large detection-max : %u", max_detection);
    return FALSE;
  }
  return TRUE;
}

/**
 * @brief tensordec-plugin's GstTensorDecoderDef callback
 *
 * [Mobilenet SSD Model]
 * The first tensor is boxes. BOX_SIZE : 1 : #MaxDetection, ANY-TYPE
 * The second tensor is labels. #MaxLabel : #MaxDetection, ANY-TYPE
 * Both tensors are MANDATORY!
 *
 * [Mobilenet SSD Postprocess Model]
 * Tensors mapping is defined through option-3, with following syntax:
 * LOCATIONS_IDX:CLASSES_IDX:SCORES_IDX:NUM_DETECTION_IDX
 *
 * Default configuration is: 3:1:2:0
 *
 * num_detection (default 1st tensor). 1, ANY-TYPE
 * detection_classes (default 2nd tensor). #MaxDetection, ANY-TYPE
 * detection_scores (default 3rd tensor). #MaxDetection, ANY-TYPE
 * detection_boxes (default 4th tensor). BOX_SIZE : #MaxDetection, ANY-TYPE
 *
 * all of tensors are MANDATORY!
 *
 * If there are third or more tensors, such tensors will be ignored.
 */
static GstCaps *
bb_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
  /** @todo this is compatible with "SSD" only. expand the capability! */
  bounding_boxes *data = *pdata;
  GstCaps *caps;
  int i;
  char *str;
  guint max_detection, max_label;

  if (_check_mode_is_mobilenet_ssd (data->mode)) {
    const uint32_t *dim1, *dim2;
    if (!_check_tensors (config, MOBILENET_SSD_MAX_TENSORS))
      return NULL;

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
    g_return_val_if_fail (max_label <= data->labeldata.total_labels, NULL);
    if (max_label < data->labeldata.total_labels)
      GST_WARNING
          ("The given tensor (2nd) has max_label (first dimension: %u) smaller than the number of labels in labels file (%s: %u).",
          max_label, data->label_path, data->labeldata.total_labels);
    g_return_val_if_fail (max_detection == dim2[1], NULL);
    for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++)
      g_return_val_if_fail (dim2[i] == 1, NULL);

    /* Check consistency with max_detection */
    if (!_set_max_detection (data, max_detection, MOBILENET_SSD_DETECTION_MAX)) {
      return NULL;
    }
  } else if (_check_mode_is_mobilenet_ssd_pp (data->mode)) {
    const uint32_t *dim1, *dim2, *dim3, *dim4;
    int locations_idx, classes_idx, scores_idx, num_idx;
    if (!_check_tensors (config, MOBILENET_SSD_PP_MAX_TENSORS))
      return NULL;

    locations_idx =
        _get_mobilenet_ssd_pp_tensor_idx (data,
        MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS);
    classes_idx =
        _get_mobilenet_ssd_pp_tensor_idx (data,
        MOBILENET_SSD_PP_BBOX_IDX_CLASSES);
    scores_idx =
        _get_mobilenet_ssd_pp_tensor_idx (data,
        MOBILENET_SSD_PP_BBOX_IDX_SCORES);
    num_idx =
        _get_mobilenet_ssd_pp_tensor_idx (data, MOBILENET_SSD_PP_BBOX_IDX_NUM);

    /* Check if the number of detections tensor is compatible */
    dim1 = config->info.info[num_idx].dimension;
    g_return_val_if_fail (dim1[0] == 1, NULL);
    for (i = 1; i < NNS_TENSOR_RANK_LIMIT; ++i)
      g_return_val_if_fail (dim1[i] == 1, NULL);

    /* Check if the classes & scores tensors are compatible */
    dim2 = config->info.info[classes_idx].dimension;
    dim3 = config->info.info[scores_idx].dimension;
    g_return_val_if_fail (dim3[0] == dim2[0], NULL);
    max_detection = dim2[0];
    for (i = 1; i < NNS_TENSOR_RANK_LIMIT; ++i) {
      g_return_val_if_fail (dim2[i] == 1, NULL);
      g_return_val_if_fail (dim3[i] == 1, NULL);
    }

    /* Check if the bbox locations tensor is compatible */
    dim4 = config->info.info[locations_idx].dimension;
    g_return_val_if_fail (BOX_SIZE == dim4[0], NULL);
    g_return_val_if_fail (max_detection == dim4[1], NULL);
    for (i = 2; i < NNS_TENSOR_RANK_LIMIT; ++i)
      g_return_val_if_fail (dim4[i] == 1, NULL);

    /* Check consistency with max_detection */
    if (!_set_max_detection (data, max_detection,
            MOBILENET_SSD_PP_DETECTION_MAX)) {
      return NULL;
    }
  } else if ((data->mode == OV_PERSON_DETECTION_BOUNDING_BOX) ||
      (data->mode == OV_FACE_DETECTION_BOUNDING_BOX)) {
    const guint *dim;

    if (!_check_tensors (config, OV_PERSON_DETECTION_MAX_TENSORS))
      return NULL;

    /**
     * The shape of the ouput tensor is [7, N, 1, 1], where N is the maximum
     * number (i.e., 200) of detected bounding boxes.
     */
    dim = config->info.info[0].dimension;
    g_return_val_if_fail (dim[0] == OV_PERSON_DETECTION_SIZE_DETECTION_DESC,
        NULL);
    g_return_val_if_fail (dim[1] == OV_PERSON_DETECTION_MAX, NULL);
    for (i = 2; i < NNS_TENSOR_RANK_LIMIT; ++i)
      g_return_val_if_fail (dim[i] == 1, NULL);
  } else if (data->mode == YOLOV5_BOUNDING_BOX) {
    const guint *dim = config->info.info[0].dimension;
    if (!_check_tensors (config, 1U))
      return NULL;

    data->max_detection = (
        (data->i_width / 32) * (data->i_height / 32) +
        (data->i_width / 16) * (data->i_height / 16) +
        (data->i_width / 8) * (data->i_height / 8)) * 3;

    g_return_val_if_fail (dim[0] ==
        (data->labeldata.total_labels + YOLOV5_DETECTION_NUM_INFO), NULL);
    g_return_val_if_fail (dim[1] == data->max_detection, NULL);
    for (i = 2; i < NNS_TENSOR_RANK_LIMIT; ++i)
      g_return_val_if_fail (dim[i] == 1, NULL);
  } else if (data->mode == MP_PALM_DETECTION_BOUNDING_BOX) {
    const uint32_t *dim1, *dim2;
    if (!_check_tensors (config, MP_PALM_DETECTION_MAX_TENSORS))
      return NULL;

    /* Check if the first tensor is compatible */
    dim1 = config->info.info[0].dimension;

    g_return_val_if_fail (dim1[0] == MP_PALM_DETECTION_INFO_SIZE, NULL);
    max_detection = dim1[1];
    g_return_val_if_fail (max_detection > 0, NULL);
    g_return_val_if_fail (dim1[2] == 1, NULL);
    for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
      g_return_val_if_fail (dim1[i] == 1, NULL);

    /* Check if the second tensor is compatible */
    dim2 = config->info.info[1].dimension;
    g_return_val_if_fail (dim2[0] == 1, NULL);
    g_return_val_if_fail (max_detection == dim2[1], NULL);
    for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++)
      g_return_val_if_fail (dim2[i] == 1, NULL);

    /* Check consistency with max_detection */
    if (!_set_max_detection (data, max_detection, MP_PALM_DETECTION_DETECTION_MAX)) {
      return NULL;
    }
  }

  str = g_strdup_printf ("video/x-raw, format = RGBA, " /* Use alpha channel to make the background transparent */
      "width = %u, height = %u", data->width, data->height);
  caps = gst_caps_from_string (str);
  setFramerateFromConfig (caps, config);
  g_free (str);

  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
bb_getTransformSize (void **pdata, const GstTensorsConfig * config,
    GstCaps * caps, size_t size, GstCaps * othercaps, GstPadDirection direction)
{
  UNUSED (pdata);
  UNUSED (config);
  UNUSED (caps);
  UNUSED (size);
  UNUSED (othercaps);
  UNUSED (direction);

  return 0;
  /** @todo Use appropriate values */
}

/** @brief Represents a detect object */
typedef struct
{
  int valid;
  int class_id;
  int x;
  int y;
  int width;
  int height;
  gfloat prob;
} detectedObject;


#define _expit(x) \
    (1.f / (1.f + expf (- ((float)x))))

/**
 * @brief C++-Template-like box location calculation for box-priors
 * @bug This is not macro-argument safe. Use paranthesis!
 * @param[in] bb The configuration, "bounding_boxes"
 * @param[in] index The index (3rd dimension of BOX_SIZE:1:MOBILENET_SSD_DETECTION_MAX:1)
 * @param[in] total_labels The count of total labels. We can get this from input tensor info. (1st dimension of LABEL_SIZE:MOBILENET_SSD_DETECTION_MAX:1:1)
 * @param[in] boxprior The box prior data from the box file of SSD.
 * @param[in] boxinputptr Cursor pointer of input + byte-per-index * index (box)
 * @param[in] detinputptr Cursor pointer of input + byte-per-index * index (detection)
 * @param[in] result The object returned. (pointer to object)
 */
#define _get_object_i_mobilenet_ssd(bb, index, total_labels, boxprior, boxinputptr, detinputptr, result) \
  do { \
    unsigned int c; \
    properties_MOBILENET_SSD *data = &bb->mobilenet_ssd; \
    float sigmoid_threshold = data->sigmoid_threshold; \
    float y_scale = data->params[MOBILENET_SSD_PARAMS_Y_SCALE_IDX]; \
    float x_scale = data->params[MOBILENET_SSD_PARAMS_X_SCALE_IDX]; \
    float h_scale = data->params[MOBILENET_SSD_PARAMS_H_SCALE_IDX]; \
    float w_scale = data->params[MOBILENET_SSD_PARAMS_W_SCALE_IDX]; \
    result->valid = FALSE; \
    for (c = 1; c < total_labels; c++) { \
      if (detinputptr[c] >= sigmoid_threshold) { \
        gfloat score = _expit (detinputptr[c]); \
        float ycenter = boxinputptr[0] / y_scale * boxprior[2][index] + boxprior[0][index]; \
        float xcenter = boxinputptr[1] / x_scale * boxprior[3][index] + boxprior[1][index]; \
        float h = (float) expf (boxinputptr[2] / h_scale) * boxprior[2][index]; \
        float w = (float) expf (boxinputptr[3] / w_scale) * boxprior[3][index]; \
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
 * @brief C++-Template-like box location calculation for box-priors for Mobilenet SSD Model
 * @param[in] bb The configuration, "bounding_boxes"
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] boxprior The box prior data from the box file of MOBILENET_SSD.
 * @param[in] boxinput Input Tensor Data (Boxes)
 * @param[in] detinput Input Tensor Data (Detection). Null if not available. (numtensor ==1)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mobilenet_ssd(bb, _type, typename, boxprior, boxinput, detinput, config, results) \
  case typename: \
  { \
    int d; \
    _type * boxinput_ = (_type *) boxinput; \
    size_t boxbpi = config->info.info[0].dimension[0]; \
    _type * detinput_ = (_type *) detinput; \
    size_t detbpi = config->info.info[1].dimension[0]; \
    int num = (MOBILENET_SSD_DETECTION_MAX > bb->max_detection) ? bb->max_detection : MOBILENET_SSD_DETECTION_MAX; \
    detectedObject object = { .valid = FALSE, .class_id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .prob = .0 }; \
    for (d = 0; d < num; d++) { \
      _get_object_i_mobilenet_ssd (bb, d, detbpi, boxprior, (boxinput_ + (d * boxbpi)), (detinput_ + (d * detbpi)), (&object)); \
      if (object.valid == TRUE) { \
        g_array_append_val (results, object); \
      } \
    } \
  } \
  break

/** @brief Macro to simplify calling _get_objects_mobilenet_ssd */
#define _get_objects_mobilenet_ssd_(type, typename) \
  _get_objects_mobilenet_ssd (bdata, type, typename, (bdata->mobilenet_ssd.box_priors), (boxes->data), (detections->data), config, results)

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

/**
 * @brief Apply NMS to the given results (objects[MOBILENET_SSD_DETECTION_MAX])
 * @param[in/out] results The results to be filtered with nms
 */
static void
nms (GArray * results, gfloat threshold)
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
          if (iou (a, b) > threshold) {
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
 * @brief C++-Template-like box location calculation for Tensorflow SSD model
 * @param[in] bb The configuration, "bounding_boxes"
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] numinput Input Tensor Data (The number of detections)
 * @param[in] classinput Input Tensor Data (Detected classes)
 * @param[in] scoreinput Input Tensor Data (Detection scores)
 * @param[in] boxesinput Input Tensor Data (Boxes)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mobilenet_ssd_pp(bb, _type, typename, numinput, classinput, scoreinput, boxesinput, config, results) \
  case typename: \
  { \
    int d, num; \
    size_t boxbpi; \
    _type * num_detection_ = (_type *) numinput; \
    _type * classes_ = (_type *) classinput; \
    _type * scores_ = (_type *) scoreinput; \
    _type * boxes_ = (_type *) boxesinput; \
    int locations_idx = _get_mobilenet_ssd_pp_tensor_idx(bb, MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS); \
    num = (int) num_detection_[0]; \
    results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), num); \
    boxbpi = config->info.info[locations_idx].dimension[0]; \
    for (d = 0; d < num; d++) { \
      _type x1, x2, y1, y2; \
      detectedObject object; \
      if (scores_[d] < _get_mobilenet_ssd_pp_threshold(bb)) \
        continue; \
      object.valid = TRUE; \
      object.class_id = (int) classes_[d]; \
      x1 = MIN(MAX(boxes_[d * boxbpi + 1], 0), 1); \
      y1 = MIN(MAX(boxes_[d * boxbpi], 0), 1); \
      x2 = MIN(MAX(boxes_[d * boxbpi + 3], 0), 1); \
      y2 = MIN(MAX(boxes_[d * boxbpi + 2], 0), 1); \
      object.x = (int) (x1 * bb->i_width); \
      object.y = (int) (y1 * bb->i_height); \
      object.width = (int) ((x2 - x1) * bb->i_width); \
      object.height = (int) ((y2 - y1) * bb->i_height); \
      object.prob = scores_[d]; \
      g_array_append_val (results, object); \
    } \
  } \
  break

/** @brief Macro to simplify calling _get_objects_mobilenet_ssd_pp */
#define _get_objects_mobilenet_ssd_pp_(type, typename) \
  _get_objects_mobilenet_ssd_pp (bdata, type, typename, (mem_num->data), (mem_classes->data), (mem_scores->data), (mem_boxes->data), config, results)

/**
 * @brief C++-Template-like box location calculation for OpenVino Person Detection Model
 * @param[in] bb The configuration, "bounding_boxes"
 * @param[in] type The tensor type of inputptr
 * @param[in] intputptr Input tensor Data
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_persons_ov(bb, type, inputptr, typename, results) \
  case typename: \
  { \
    detectedObject object = { .valid = FALSE, .class_id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .prob = .0 }; \
    type *typed_inputptr = (type *) inputptr; \
    guint d; \
    \
    for (d = 1; d <= OV_PERSON_DETECTION_MAX; ++d) { \
      struct { \
        type image_id; \
        type label; \
        type conf; \
        type x_min; \
        type y_min; \
        type x_max; \
        type y_max; \
      } desc; \
      \
      memcpy (&desc, typed_inputptr, sizeof(desc)); \
      typed_inputptr += (sizeof(desc) / sizeof(type)); \
      object.valid = FALSE; \
      \
      if ((int) desc.image_id < 0) { \
        bb->max_detection = (d - 1); \
        break; \
      } \
      object.class_id = -1; \
      object.x = (int) (desc.x_min * (type) bb->i_width); \
      object.y = (int) (desc.y_min * (type) bb->i_height); \
      object.width = (int) ((desc.x_max  - desc.x_min) * (type) bb->i_width); \
      object.height = (int) ((desc.y_max - desc.y_min)* (type) bb->i_height); \
      if (desc.conf < OV_PERSON_DETECTION_CONF_THRESHOLD) \
        continue; \
      object.prob = 1; \
      object.valid = TRUE; \
      g_array_append_val (results, object); \
    } \
  } \
  break


/**
 * @brief C++-Template-like box location calculation for Tensorflow model
 * @param[in] bb The configuration, "bounding_boxes"
 * @param[in] data palm detection configuration, "properties_MP_PALM_DETECTION"
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] scoreinput Input Tensor Data (Detection scores)
 * @param[in] boxesinput Input Tensor Data (Boxes)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mp_palm_detection(bb, data, _type, typename, scoreinput, boxesinput, config, results) \
  case typename: \
  { \
    int d_; \
    _type * scores_ = (_type *) scoreinput; \
    _type * boxes_ = (_type *) boxesinput; \
    guint i_width_ = bb->i_width; \
    guint i_height_ = bb->i_height; \
    int num_ = bb->max_detection; \
    size_t boxbpi_ = config->info.info[0].dimension[0]; \
    results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), num_); \
    for (d_ = 0; d_ < num_; d_++) { \
      gfloat y_center, x_center, h, w; \
      gfloat ymin, xmin; \
      int y, x, width, height; \
      detectedObject object; \
      gfloat score = (gfloat)scores_[d_]; \
      _type * box = boxes_ + boxbpi_ * d_; \
      anchor * a = &g_array_index (data->anchors, anchor, d_); \
      score = MAX(score, -100.0f); \
      score = MIN(score, 100.0f); \
      score = 1.0f / (1.0f + exp (-score)); \
      if (score < data->min_score_threshold) \
        continue; \
      y_center = (box[0] * 1.f) / i_height_ * a->h + a->y_center; \
      x_center = (box[1] * 1.f) / i_width_ * a->w + a->x_center; \
      h = (box[2] * 1.f) / i_height_ * a->h; \
      w = (box[3] * 1.f) / i_width_ * a->w; \
      ymin = y_center - h / 2.f; \
      xmin = x_center - w / 2.f; \
      y = ymin * i_height_; \
      x = xmin * i_width_; \
      width = w * i_width_; \
      height = h * i_height_; \
      object.class_id = 0; \
      object.x = MAX (0, x); \
      object.y = MAX (0, y); \
      object.width = width; \
      object.height = height; \
      object.prob = score; \
      object.valid = TRUE; \
      g_array_append_val (results, object); \
    } \
  } \
  break

/** @brief Macro to simplify calling _get_objects_mp_palm_detection */
#define _get_objects_mp_palm_detection_(type, typename) \
  _get_objects_mp_palm_detection (bdata, data, type, typename, (detections->data), (boxes->data), config, results)

/**
 * @brief Draw with the given results (objects[MOBILENET_SSD_DETECTION_MAX]) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bounding-box internal data.
 * @param[in] results The final results to be drawn.
 */
static void
draw (GstMapInfo * out_info, bounding_boxes * bdata, GArray * results)
{
  uint32_t *frame = (uint32_t *) out_info->data;        /* Let's draw per pixel (4bytes) */
  unsigned int i;

  for (i = 0; i < results->len; i++) {
    int x1, x2, y1, y2;         /* Box positions on the output surface */
    int j;
    uint32_t *pos1, *pos2;
    const char *label;
    int label_len;
    detectedObject *a = &g_array_index (results, detectedObject, i);


    if ((bdata->flag_use_label) &&
        ((a->class_id < 0 ||
                a->class_id >= (int) bdata->labeldata.total_labels))) {
      /** @todo make it "logw_once" after we get logw_once API. */
      ml_logw ("Invalid class found with tensordec-boundingbox.c.\n");
      continue;
    }

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
    if (bdata->flag_use_label) {
      label = bdata->labeldata.labels[a->class_id];
      label_len = strlen (label);
      /* x1 is the same: x1 = MAX (0, (bdata->width * a->x) / bdata->i_width); */
      y1 = MAX (0, (y1 - 14));
      pos1 = &frame[y1 * bdata->width + x1];
      for (j = 0; j < label_len; j++) {
        unsigned int char_index = label[j];
        if ((x1 + 8) > (int) bdata->width)
          break;                /* Stop drawing if it may overfill */
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
        pos1 += 9;              /* charater width + 1px */
      }
    }
  }
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
bb_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  bounding_boxes *bdata = *pdata;
  const size_t size = (size_t) bdata->width * bdata->height * 4; /* RGBA */
  GstMapInfo out_info;
  GstMemory *out_mem;
  GArray *results = NULL;
  const guint num_tensors = config->info.num_tensors;
  gboolean need_output_alloc;

  g_assert (outbuf);
  need_output_alloc = gst_buffer_get_size (outbuf) == 0;

  if (_check_label_props (bdata))
    bdata->flag_use_label = TRUE;
  else
    bdata->flag_use_label = FALSE;

  /* Ensure we have outbuf properly allocated */
  if (need_output_alloc) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    ml_loge ("Cannot map output memory / tensordec-bounding_boxes.\n");
    goto error_free;
  }

  /** reset the buffer with alpha 0 / black */
  memset (out_info.data, 0, size);

  if (_check_mode_is_mobilenet_ssd (bdata->mode)) {
    const GstTensorMemory *boxes, *detections = NULL;
    properties_MOBILENET_SSD *data = &bdata->mobilenet_ssd;
    /**
     * @todo 100 is a heuristic number of objects in a picture frame
     *       We may have better "heuristics" than this.
     *       For the sake of performance, don't make it too small.
     */

    /* Already checked with getOutCaps. Thus, this is an internal bug */
    g_assert (num_tensors >= MOBILENET_SSD_MAX_TENSORS);
    results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), 100);

    boxes = &input[0];
    if (num_tensors >= MOBILENET_SSD_MAX_TENSORS) /* lgtm[cpp/constant-comparison] */
      detections = &input[1];

    switch (config->info.info[0].type) {
        _get_objects_mobilenet_ssd_ (uint8_t, _NNS_UINT8);
        _get_objects_mobilenet_ssd_ (int8_t, _NNS_INT8);
        _get_objects_mobilenet_ssd_ (uint16_t, _NNS_UINT16);
        _get_objects_mobilenet_ssd_ (int16_t, _NNS_INT16);
        _get_objects_mobilenet_ssd_ (uint32_t, _NNS_UINT32);
        _get_objects_mobilenet_ssd_ (int32_t, _NNS_INT32);
        _get_objects_mobilenet_ssd_ (uint64_t, _NNS_UINT64);
        _get_objects_mobilenet_ssd_ (int64_t, _NNS_INT64);
        _get_objects_mobilenet_ssd_ (float, _NNS_FLOAT32);
        _get_objects_mobilenet_ssd_ (double, _NNS_FLOAT64);
      default:
        g_assert (0);
    }
    nms (results, data->params[MOBILENET_SSD_PARAMS_IOU_THRESHOLD_IDX]);
  } else if (_check_mode_is_mobilenet_ssd_pp (bdata->mode)) {
    const GstTensorMemory *mem_num, *mem_classes, *mem_scores, *mem_boxes;
    int locations_idx, classes_idx, scores_idx, num_idx;

    /* Already checked with getOutCaps. Thus, this is an internal bug */
    g_assert (num_tensors >= MOBILENET_SSD_PP_MAX_TENSORS);

    locations_idx =
        _get_mobilenet_ssd_pp_tensor_idx (bdata,
        MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS);
    classes_idx =
        _get_mobilenet_ssd_pp_tensor_idx (bdata,
        MOBILENET_SSD_PP_BBOX_IDX_CLASSES);
    scores_idx =
        _get_mobilenet_ssd_pp_tensor_idx (bdata,
        MOBILENET_SSD_PP_BBOX_IDX_SCORES);
    num_idx =
        _get_mobilenet_ssd_pp_tensor_idx (bdata, MOBILENET_SSD_PP_BBOX_IDX_NUM);

    mem_num = &input[num_idx];
    mem_classes = &input[classes_idx];
    mem_scores = &input[scores_idx];
    mem_boxes = &input[locations_idx];

    switch (config->info.info[num_idx].type) {
        _get_objects_mobilenet_ssd_pp_ (uint8_t, _NNS_UINT8);
        _get_objects_mobilenet_ssd_pp_ (int8_t, _NNS_INT8);
        _get_objects_mobilenet_ssd_pp_ (uint16_t, _NNS_UINT16);
        _get_objects_mobilenet_ssd_pp_ (int16_t, _NNS_INT16);
        _get_objects_mobilenet_ssd_pp_ (uint32_t, _NNS_UINT32);
        _get_objects_mobilenet_ssd_pp_ (int32_t, _NNS_INT32);
        _get_objects_mobilenet_ssd_pp_ (uint64_t, _NNS_UINT64);
        _get_objects_mobilenet_ssd_pp_ (int64_t, _NNS_INT64);
        _get_objects_mobilenet_ssd_pp_ (float, _NNS_FLOAT32);
        _get_objects_mobilenet_ssd_pp_ (double, _NNS_FLOAT64);
      default:
        g_assert (0);
    }
  } else if ((bdata->mode == OV_PERSON_DETECTION_BOUNDING_BOX) ||
      (bdata->mode == OV_FACE_DETECTION_BOUNDING_BOX)) {
    /* Already checked with getOutCaps. Thus, this is an internal bug */
    g_assert (num_tensors >= OV_PERSON_DETECTION_MAX_TENSORS);

    results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject),
        OV_PERSON_DETECTION_MAX);
    switch (config->info.info[0].type) {
        _get_persons_ov (bdata, uint8_t, input[0].data, _NNS_UINT8, results);
        _get_persons_ov (bdata, int8_t, input[0].data, _NNS_INT8, results);
        _get_persons_ov (bdata, uint16_t, input[0].data, _NNS_UINT16, results);
        _get_persons_ov (bdata, int16_t, input[0].data, _NNS_INT16, results);
        _get_persons_ov (bdata, uint32_t, input[0].data, _NNS_UINT32, results);
        _get_persons_ov (bdata, int32_t, input[0].data, _NNS_INT32, results);
        _get_persons_ov (bdata, uint64_t, input[0].data, _NNS_UINT64, results);
        _get_persons_ov (bdata, int64_t, input[0].data, _NNS_INT64, results);
        _get_persons_ov (bdata, float, input[0].data, _NNS_FLOAT32, results);
        _get_persons_ov (bdata, double, input[0].data, _NNS_FLOAT64, results);
      default:
        g_assert (0);
    }
  } else if (bdata->mode == YOLOV5_BOUNDING_BOX) {
    int bIdx, numTotalBox;
    int cIdx, numTotalClass, cStartIdx, cIdxMax;
    float *boxinput;

    numTotalBox = bdata->max_detection;
    numTotalClass = bdata->labeldata.total_labels;
    cStartIdx = YOLOV5_DETECTION_NUM_INFO;
    cIdxMax = numTotalClass + cStartIdx;

    boxinput = (float *) input[0].data; // boxinput[1][1][numTotalBox][cIdxMax]

    /** Only support for float type model */
    g_assert (config->info.info[0].type == _NNS_FLOAT32);

    results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), numTotalBox);
    for (bIdx = 0; bIdx < numTotalBox; ++bIdx) {
      float maxClassConfVal = -INFINITY;
      int maxClassIdx = -1;
      for (cIdx = cStartIdx; cIdx < cIdxMax; ++cIdx) {
        if (boxinput[bIdx * cIdxMax + cIdx] > maxClassConfVal) {
          maxClassConfVal = boxinput[bIdx * cIdxMax + cIdx];
          maxClassIdx = cIdx;
        }
      }

      if (maxClassConfVal * boxinput[bIdx * cIdxMax + 4] >
          YOLOV5_DETECTION_CONF_THRESHOLD) {
        detectedObject object;
        float cx, cy, w, h;
        cx = boxinput[bIdx * cIdxMax + 0] * (float) bdata->i_width;
        cy = boxinput[bIdx * cIdxMax + 1] * (float) bdata->i_height;
        w = boxinput[bIdx * cIdxMax + 2] * (float) bdata->i_width;
        h = boxinput[bIdx * cIdxMax + 3] * (float) bdata->i_height;

        object.x = (int) (MAX (0.f, (cx - w / 2.f)));
        object.y = (int) (MAX (0.f, (cy - h / 2.f)));
        object.width = (int) (MIN ((float) bdata->i_width, w));
        object.height = (int) (MIN ((float) bdata->i_height, h));

        object.prob = maxClassConfVal * boxinput[bIdx * cIdxMax + 4];
        object.class_id = maxClassIdx - YOLOV5_DETECTION_NUM_INFO;
        object.valid = TRUE;
        g_array_append_val (results, object);
      }
    }

    nms (results, YOLOV5_DETECTION_IOU_THRESHOLD);
  } else if (bdata->mode == MP_PALM_DETECTION_BOUNDING_BOX) {
    const GstTensorMemory *boxes = NULL;
    const GstTensorMemory *detections = NULL;
    properties_MP_PALM_DETECTION *data = &bdata->mp_palm_detection;

    /* Already checked with getOutCaps. Thus, this is an internal bug */
    g_assert (num_tensors >= MP_PALM_DETECTION_MAX_TENSORS);
    results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), 100);

    boxes = &input[0];
    detections = &input[1];

    switch (config->info.info[0].type) {
      _get_objects_mp_palm_detection_ (uint8_t, _NNS_UINT8);
      _get_objects_mp_palm_detection_ (int8_t, _NNS_INT8);
      _get_objects_mp_palm_detection_ (uint16_t, _NNS_UINT16);
      _get_objects_mp_palm_detection_ (int16_t, _NNS_INT16);
      _get_objects_mp_palm_detection_ (uint32_t, _NNS_UINT32);
      _get_objects_mp_palm_detection_ (int32_t, _NNS_INT32);
      _get_objects_mp_palm_detection_ (uint64_t, _NNS_UINT64);
      _get_objects_mp_palm_detection_ (int64_t, _NNS_INT64);
      _get_objects_mp_palm_detection_ (float, _NNS_FLOAT32);
      _get_objects_mp_palm_detection_ (double, _NNS_FLOAT64);

      default:
        g_assert (0);
    }
    nms (results, 0.05f);
  } else {
    GST_ERROR ("Failed to get output buffer, unknown mode %d.", bdata->mode);
    goto error_unmap;
  }

  draw (&out_info, bdata, results);
  g_array_free (results, TRUE);

  gst_memory_unmap (out_mem, &out_info);

  if (need_output_alloc)
    gst_buffer_append_memory (outbuf, out_mem);
  else
    gst_memory_unref (out_mem);

  return GST_FLOW_OK;

error_unmap:
  gst_memory_unmap (out_mem, &out_info);
error_free:
  gst_memory_unref (out_mem);

  return GST_FLOW_ERROR;
}

static gchar decoder_subplugin_bounding_box[] = "bounding_boxes";

/** @brief Bounding box tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef boundingBox = {
  .modename = decoder_subplugin_bounding_box,
  .init = bb_init,
  .exit = bb_exit,
  .setOption = bb_setOption,
  .getOutCaps = bb_getOutCaps,
  .getTransformSize = bb_getTransformSize,
  .decode = bb_decode
};

/** @brief Initialize this object for tensordec-plugin */
void
init_bb (void)
{
  nnstreamer_decoder_probe (&boundingBox);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_bb (void)
{
  nnstreamer_decoder_exit (boundingBox.modename);
}
