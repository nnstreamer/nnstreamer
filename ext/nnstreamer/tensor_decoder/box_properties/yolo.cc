/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor-decoder bounding box properties
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 */
/**
 * @file        yolo.cc
 * @date        13 May 2024
 * @brief       NNStreamer tensor-decoder bounding box properties
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Yelin Jeong <yelini.jeong@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include <nnstreamer_plugin_api_util.h>
#include "../tensordec-boundingbox.h"

#define YOLO_DETECTION_CONF_THRESHOLD (0.25)
#define YOLO_DETECTION_IOU_THRESHOLD (0.45)
#define DEFAULT_DETECTION_NUM_INFO_YOLO5 (5)
#define DEFAULT_DETECTION_NUM_INFO_YOLO8 (4)

/** @brief Constructor of YoloV5 */
YoloV5::YoloV5 ()
{
  scaled_output = 0;
  conf_threshold = YOLO_DETECTION_CONF_THRESHOLD;
  iou_threshold = YOLO_DETECTION_IOU_THRESHOLD;
}

/** @brief Set internal option of YoloV5
 *  @param[in] param The option string.
 */
int
YoloV5::setOptionInternal (const char *param)
{
  gchar **options;
  int noptions;

  options = g_strsplit (param, ":", -1);
  noptions = g_strv_length (options);
  if (noptions > 0)
    scaled_output = (int) g_ascii_strtoll (options[0], NULL, 10);
  if (noptions > 1)
    conf_threshold = (gfloat) g_ascii_strtod (options[1], NULL);
  if (noptions > 2)
    iou_threshold = (gfloat) g_ascii_strtod (options[2], NULL);

  nns_logi ("Setting YOLOV5/YOLOV8 decoder as scaled_output: %d, conf_threshold: %.2f, iou_threshold: %.2f",
      scaled_output, conf_threshold, iou_threshold);

  g_strfreev (options);
  return TRUE;
}

/** @brief Check compatibility of given tensors config
 *  @param[in] param The option string.
 */
int
YoloV5::checkCompatible (const GstTensorsConfig *config)
{
  const guint *dim = config->info.info[0].dimension;
  int i;

  if (!check_tensors (config, 1U))
    return FALSE;

  max_detection = ((i_width / 32) * (i_height / 32) + (i_width / 16) * (i_height / 16)
                      + (i_width / 8) * (i_height / 8))
                  * 3;

  g_return_val_if_fail (dim[0] == (total_labels + DEFAULT_DETECTION_NUM_INFO_YOLO5), FALSE);
  g_return_val_if_fail (dim[1] == max_detection, FALSE);
  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; ++i)
    g_return_val_if_fail (dim[i] == 0 || dim[i] == 1, FALSE);
  return TRUE;
}

/**
 * @brief Decode input memory to out buffer
 * @param[in] config The structure of input tensor info.
 * @param[in] input The array of input tensor data. The maximum array size of input data is NNS_TENSOR_SIZE_LIMIT.
 */
GArray *
YoloV5::decode (const GstTensorsConfig *config, const GstTensorMemory *input)
{
  GArray *results = NULL;

  int bIdx, numTotalBox;
  int cIdx, numTotalClass, cStartIdx, cIdxMax;
  float *boxinput;
  int is_output_scaled = scaled_output;

  numTotalBox = max_detection;
  numTotalClass = total_labels;
  cStartIdx = DEFAULT_DETECTION_NUM_INFO_YOLO5;
  cIdxMax = numTotalClass + cStartIdx;

  /* boxinput[numTotalBox][cIdxMax] */
  boxinput = (float *) input[0].data;

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

    if (maxClassConfVal * boxinput[bIdx * cIdxMax + 4] > conf_threshold) {
      detectedObject object;
      float cx, cy, w, h;
      cx = boxinput[bIdx * cIdxMax + 0];
      cy = boxinput[bIdx * cIdxMax + 1];
      w = boxinput[bIdx * cIdxMax + 2];
      h = boxinput[bIdx * cIdxMax + 3];

      if (!is_output_scaled) {
        cx *= (float) i_width;
        cy *= (float) i_height;
        w *= (float) i_width;
        h *= (float) i_height;
      }

      object.x = (int) (MAX (0.f, (cx - w / 2.f)));
      object.y = (int) (MAX (0.f, (cy - h / 2.f)));
      object.width = (int) (MIN ((float) i_width, w));
      object.height = (int) (MIN ((float) i_height, h));

      object.prob = maxClassConfVal * boxinput[bIdx * cIdxMax + 4];
      object.class_id = maxClassIdx - DEFAULT_DETECTION_NUM_INFO_YOLO5;
      object.tracking_id = 0;
      object.valid = TRUE;
      g_array_append_val (results, object);
    }
  }

  nms (results, iou_threshold);
  return results;
}

/** @brief Constructor of YoloV8 */
YoloV8::YoloV8 ()
{
  scaled_output = 0;
  conf_threshold = YOLO_DETECTION_CONF_THRESHOLD;
  iou_threshold = YOLO_DETECTION_IOU_THRESHOLD;
}

/** @brief Set internal option of YoloV8 */
int
YoloV8::setOptionInternal (const char *param)
{
  gchar **options;
  int noptions;

  options = g_strsplit (param, ":", -1);
  noptions = g_strv_length (options);
  if (noptions > 0)
    scaled_output = (int) g_ascii_strtoll (options[0], NULL, 10);
  if (noptions > 1)
    conf_threshold = (gfloat) g_ascii_strtod (options[1], NULL);
  if (noptions > 2)
    iou_threshold = (gfloat) g_ascii_strtod (options[2], NULL);

  nns_logi ("Setting YOLOV5/YOLOV8 decoder as scaled_output: %d, conf_threshold: %.2f, iou_threshold: %.2f",
      scaled_output, conf_threshold, iou_threshold);

  g_strfreev (options);
  return TRUE;
}

/** @brief Check compatibility of given tensors config */
int
YoloV8::checkCompatible (const GstTensorsConfig *config)
{
  const guint *dim = config->info.info[0].dimension;
  int i;
  if (!check_tensors (config, 1U)) {
    gchar *typestr = gst_tensors_info_to_string (&config->info);
    nns_loge ("Yolov8 bounding-box decoder needs at least 1 valid tensor. The given input tensor is: %s.",
        typestr);
    g_free (typestr);
    return FALSE;
  }
  /** Only support for float type model */
  if (config->info.info[0].type != _NNS_FLOAT32) {
    gchar *typestr = gst_tensors_info_to_string (&config->info);
    nns_loge ("Yolov8 bounding-box decoder accepts float32 input tensors only. The given input tensor is: %s.",
        typestr);
    g_free (typestr);
    return FALSE;
  }

  max_detection = (i_width / 32) * (i_height / 32) + (i_width / 16) * (i_height / 16)
                  + (i_width / 8) * (i_height / 8);

  if (dim[0] != (total_labels + DEFAULT_DETECTION_NUM_INFO_YOLO8) || dim[1] != max_detection) {
    nns_loge ("yolov8 boundingbox decoder requires the input shape to be %d:%d:1. But given shape is %d:%d:1. `tensor_transform mode=transpose` would be helpful.",
        total_labels + DEFAULT_DETECTION_NUM_INFO_YOLO8, max_detection, dim[0], dim[1]);
    return FALSE;
  }

  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; ++i)
    if (dim[i] != 0 && dim[i] != 1) {
      gchar *typestr = gst_tensors_info_to_string (&config->info);
      nns_loge ("Yolov8 bounding-box decoder accepts RANK=2 tensors (3rd and later dimensions should be 1 or 0). The given input tensor is: %s.",
          typestr);
      g_free (typestr);
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
YoloV8::decode (const GstTensorsConfig *config, const GstTensorMemory *input)
{
  GArray *results = NULL;
  int bIdx, numTotalBox;
  int cIdx, numTotalClass, cStartIdx, cIdxMax;
  float *boxinput;
  int is_output_scaled = scaled_output;
  UNUSED (config);

  numTotalBox = max_detection;
  numTotalClass = total_labels;
  cStartIdx = DEFAULT_DETECTION_NUM_INFO_YOLO8;
  cIdxMax = numTotalClass + cStartIdx;

  /* boxinput[numTotalBox][cIdxMax] */
  boxinput = (float *) input[0].data;

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

    if (maxClassConfVal > conf_threshold) {
      detectedObject object;
      float cx, cy, w, h;
      cx = boxinput[bIdx * cIdxMax + 0];
      cy = boxinput[bIdx * cIdxMax + 1];
      w = boxinput[bIdx * cIdxMax + 2];
      h = boxinput[bIdx * cIdxMax + 3];

      if (!is_output_scaled) {
        cx *= (float) i_width;
        cy *= (float) i_height;
        w *= (float) i_width;
        h *= (float) i_height;
      }

      object.x = (int) (MAX (0.f, (cx - w / 2.f)));
      object.y = (int) (MAX (0.f, (cy - h / 2.f)));
      object.width = (int) (MIN ((float) i_width, w));
      object.height = (int) (MIN ((float) i_height, h));

      object.prob = maxClassConfVal;
      object.class_id = maxClassIdx - DEFAULT_DETECTION_NUM_INFO_YOLO8;
      object.tracking_id = 0;
      object.valid = TRUE;
      g_array_append_val (results, object);
    }
  }

  nms (results, iou_threshold);
  return results;
}
