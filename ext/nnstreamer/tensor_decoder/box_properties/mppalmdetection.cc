/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor-decoder bounding box properties
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 */
/**
 * @file        mppalmdetection.cc
 * @date        13 May 2024
 * @brief       NNStreamer tensor-decoder bounding box properties
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Yelin Jeong <yelini.jeong@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include "../tensordec-boundingbox.h"

#define INFO_SIZE (18)
#define MAX_TENSORS (2U)
#define MAX_DETECTION (2016)

#define NUM_LAYERS_DEFAULT (4)
#define MIN_SCALE_DEFAULT (1.0)
#define MAX_SCALE_DEFAULT (1.0)
#define OFFSET_X_DEFAULT (0.5)
#define OFFSET_Y_DEFAULT (0.5)
#define STRIDE_0_DEFAULT (8)
#define STRIDE_1_DEFAULT (16)
#define STRIDE_2_DEFAULT (16)
#define STRIDE_3_DEFAULT (16)
#define MIN_SCORE_THRESHOLD_DEFAULT (0.5)

#define PARAMS_STRIDE_SIZE (8)

/**
 * @brief C++-Template-like box location calculation for Tensorflow model
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] scoreinput Input Tensor Data (Detection scores)
 * @param[in] boxesinput Input Tensor Data (Boxes)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mp_palm_detection(_type, typename, scoreinput, boxesinput, config) \
  case typename:                                                                        \
    {                                                                                   \
      int d_;                                                                           \
      _type *scores_ = (_type *) scoreinput;                                            \
      _type *boxes_ = (_type *) boxesinput;                                             \
      int num_ = max_detection;                                                         \
      size_t boxbpi_ = config->info.info[0].dimension[0];                               \
      results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), num_);         \
      for (d_ = 0; d_ < num_; d_++) {                                                   \
        gfloat y_center, x_center, h, w;                                                \
        gfloat ymin, xmin;                                                              \
        int y, x, width, height;                                                        \
        detectedObject object;                                                          \
        gfloat score = (gfloat) scores_[d_];                                            \
        _type *box = boxes_ + boxbpi_ * d_;                                             \
        anchor *a = &g_array_index (this->anchors, anchor, d_);                         \
        score = MAX (score, -100.0f);                                                   \
        score = MIN (score, 100.0f);                                                    \
        score = 1.0f / (1.0f + exp (-score));                                           \
        if (score < min_score_threshold)                                                \
          continue;                                                                     \
        y_center = (box[0] * 1.f) / i_height * a->h + a->y_center;                      \
        x_center = (box[1] * 1.f) / i_width * a->w + a->x_center;                       \
        h = (box[2] * 1.f) / i_height * a->h;                                           \
        w = (box[3] * 1.f) / i_width * a->w;                                            \
        ymin = y_center - h / 2.f;                                                      \
        xmin = x_center - w / 2.f;                                                      \
        y = ymin * i_height;                                                            \
        x = xmin * i_width;                                                             \
        width = w * i_width;                                                            \
        height = h * i_height;                                                          \
        object.class_id = 0;                                                            \
        object.x = MAX (0, x);                                                          \
        object.y = MAX (0, y);                                                          \
        object.width = width;                                                           \
        object.height = height;                                                         \
        object.prob = score;                                                            \
        object.valid = TRUE;                                                            \
        g_array_append_val (results, object);                                           \
      }                                                                                 \
    }                                                                                   \
    break

/** @brief Macro to simplify calling _get_objects_mp_palm_detection */
#define _get_objects_mp_palm_detection_(type, typename) \
  _get_objects_mp_palm_detection (type, typename, (detections->data), (boxes->data), config)

#define mp_palm_detection_option(option, type, idx) \
  if (noptions > idx)                               \
  option = (type) g_strtod (options[idx], NULL)

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
void
MpPalmDetection::mp_palm_detection_generate_anchors ()
{
  int layer_id = 0;
  guint i;

  while (layer_id < num_layers) {
    GArray *aspect_ratios = g_array_new (FALSE, TRUE, sizeof (gfloat));
    GArray *scales = g_array_new (FALSE, TRUE, sizeof (gfloat));
    GArray *anchor_height = g_array_new (FALSE, TRUE, sizeof (gfloat));
    GArray *anchor_width = g_array_new (FALSE, TRUE, sizeof (gfloat));

    int last_same_stride_layer = layer_id;

    while (last_same_stride_layer < num_layers
           && strides[last_same_stride_layer] == strides[layer_id]) {
      gfloat scale;
      gfloat ratio = 1.0f;
      g_array_append_val (aspect_ratios, ratio);
      g_array_append_val (aspect_ratios, ratio);
      scale = _calculate_scale (min_scale, max_scale, last_same_stride_layer, num_layers);
      g_array_append_val (scales, scale);
      scale = _calculate_scale (min_scale, max_scale, last_same_stride_layer + 1, num_layers);
      g_array_append_val (scales, scale);
      last_same_stride_layer++;
    }

    for (i = 0; i < aspect_ratios->len; ++i) {
      const float ratio_sqrts = sqrt (g_array_index (aspect_ratios, gfloat, i));
      const gfloat sc = g_array_index (scales, gfloat, i);
      gfloat anchor_height_ = sc / ratio_sqrts;
      gfloat anchor_width_ = sc * ratio_sqrts;
      g_array_append_val (anchor_height, anchor_height_);
      g_array_append_val (anchor_width, anchor_width_);
    }

    {
      int feature_map_height = 0;
      int feature_map_width = 0;
      int x, y;
      int anchor_id;

      const int stride = strides[layer_id];
      feature_map_height = ceil (1.0f * 192 / stride);
      feature_map_width = ceil (1.0f * 192 / stride);
      for (y = 0; y < feature_map_height; ++y) {
        for (x = 0; x < feature_map_width; ++x) {
          for (anchor_id = 0; anchor_id < (int) aspect_ratios->len; ++anchor_id) {
            const float x_center = (x + offset_x) * 1.0f / feature_map_width;
            const float y_center = (y + offset_y) * 1.0f / feature_map_height;

            const anchor a = { .x_center = x_center,
              .y_center = y_center,
              .w = g_array_index (anchor_width, gfloat, anchor_id),
              .h = g_array_index (anchor_height, gfloat, anchor_id) };
            g_array_append_val (anchors, a);
          }
        }
      }
      layer_id = last_same_stride_layer;
    }

    g_array_free (anchor_height, TRUE);
    g_array_free (anchor_width, TRUE);
    g_array_free (aspect_ratios, TRUE);
    g_array_free (scales, TRUE);
  }
}

/** @brief Constructor of MpPalmDetection */
MpPalmDetection::MpPalmDetection ()
{
  num_layers = NUM_LAYERS_DEFAULT;
  min_scale = MIN_SCALE_DEFAULT;
  max_scale = MAX_SCALE_DEFAULT;
  offset_x = OFFSET_X_DEFAULT;
  offset_y = OFFSET_Y_DEFAULT;
  strides[0] = STRIDE_0_DEFAULT;
  strides[1] = STRIDE_1_DEFAULT;
  strides[2] = STRIDE_2_DEFAULT;
  strides[3] = STRIDE_3_DEFAULT;
  min_score_threshold = MIN_SCORE_THRESHOLD_DEFAULT;
  anchors = g_array_new (FALSE, TRUE, sizeof (anchor));
}

/** @brief Destructor of MpPalmDetection */
MpPalmDetection::~MpPalmDetection ()
{
  if (anchors)
    g_array_free (anchors, TRUE);
  anchors = NULL;
}

/** @brief Set internal option of MpPalmDetection
 *  @param[in] param The option string.
 */
int
MpPalmDetection::setOptionInternal (const char *param)
{
  /* Load palm detection info from option3 */
  gchar **options;
  int noptions, idx;
  int ret = TRUE;

  options = g_strsplit (param, ":", -1);
  noptions = g_strv_length (options);

  if (noptions > PARAMS_MAX) {
    GST_ERROR ("Invalid MP PALM DETECTION PARAM length: %d", noptions);
    ret = FALSE;
    goto exit_mp_palm_detection;
  }

  mp_palm_detection_option (min_score_threshold, gfloat, 0);
  mp_palm_detection_option (num_layers, gint, 1);
  mp_palm_detection_option (min_scale, gfloat, 2);
  mp_palm_detection_option (max_scale, gfloat, 3);
  mp_palm_detection_option (offset_x, gfloat, 4);
  mp_palm_detection_option (offset_y, gfloat, 5);

  for (idx = 6; idx < num_layers + 6; idx++) {
    mp_palm_detection_option (strides[idx - 6], gint, idx);
  }
  mp_palm_detection_generate_anchors ();

exit_mp_palm_detection:
  g_strfreev (options);
  return ret;
}

/** @brief Check compatibility of given tensors config */
int
MpPalmDetection::checkCompatible (const GstTensorsConfig *config)
{
  const uint32_t *dim1, *dim2;
  int i;
  if (!check_tensors (config, MAX_TENSORS))
    return FALSE;

  /* Check if the first tensor is compatible */
  dim1 = config->info.info[0].dimension;

  g_return_val_if_fail (dim1[0] == INFO_SIZE, FALSE);
  max_detection = dim1[1];
  g_return_val_if_fail (max_detection > 0, FALSE);
  g_return_val_if_fail (dim1[2] == 1, FALSE);
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim1[i] == 0 || dim1[i] == 1, FALSE);

  /* Check if the second tensor is compatible */
  dim2 = config->info.info[1].dimension;
  g_return_val_if_fail (dim2[0] == 1, FALSE);
  g_return_val_if_fail (max_detection == dim2[1], FALSE);
  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim2[i] == 0 || dim2[i] == 1, FALSE);

  /* Check consistency with max_detection */
  if (this->max_detection == 0)
    this->max_detection = max_detection;
  else
    g_return_val_if_fail (max_detection == this->max_detection, FALSE);

  if (this->max_detection > MAX_DETECTION) {
    GST_ERROR ("Incoming tensor has too large detection-max : %u", max_detection);
    return FALSE;
  }
  return TRUE;
}

/**
 * @brief Decode input memory to out buffer
 * @param[in] config The structure of input tensor info.
 * @param[in] input The array of input tensor data. The maximum array size of input data is NNS_TENSOR_SIZE_LIMIT.
 */
GArray *
MpPalmDetection::decode (const GstTensorsConfig *config, const GstTensorMemory *input)
{
  GArray *results = NULL;
  const GstTensorMemory *boxes = NULL;
  const GstTensorMemory *detections = NULL;
  const guint num_tensors = config->info.num_tensors;

  /* Already checked with getOutCaps. Thus, this is an internal bug */
  g_assert (num_tensors >= MAX_TENSORS);

  /* results will be allocated by _get_objects_mp_palm_detection_ */
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
  return results;
}
