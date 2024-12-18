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

/**
 * @brief Class for YoloV5 box properties
 */
class YoloV5 : public BoxProperties
{
  public:
  YoloV5 ();
  ~YoloV5 ();
  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  private:
  /* From option3, whether the output values are scaled or not */
  int scaled_output;
  gfloat conf_threshold;
  gfloat iou_threshold;
};

/**
 * @brief Class for YoloV8 box properties
 */
class YoloV8 : public BoxProperties
{
  public:
  YoloV8 ();
  ~YoloV8 ();
  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  private:
  /* From option3, whether the output values are scaled or not */
  int scaled_output;
  gfloat conf_threshold;
  gfloat iou_threshold;
};

/**
 * @brief Class for YoloV10 box properties
 */
class YoloV10 : public BoxProperties
{
  public:
  YoloV10 ();
  ~YoloV10 ();
  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  private:
  gfloat conf_threshold;
};

static BoxProperties *yolo5 = nullptr;
static BoxProperties *yolo8 = nullptr;
static BoxProperties *yolo10 = nullptr;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_properties_yolo5 (void) __attribute__ ((constructor));
void fini_properties_yolo5 (void) __attribute__ ((destructor));

void init_properties_yolo8 (void) __attribute__ ((constructor));
void fini_properties_yolo8 (void) __attribute__ ((destructor));

void init_properties_yolo10 (void) __attribute__ ((constructor));
void fini_properties_yolo10 (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/** @brief Constructor of YoloV5 */
YoloV5::YoloV5 ()
{
  scaled_output = 0;
  conf_threshold = YOLO_DETECTION_CONF_THRESHOLD;
  iou_threshold = YOLO_DETECTION_IOU_THRESHOLD;
  name = g_strdup_printf ("yolov5");
}

/** @brief Destructor of YoloV5 */
YoloV5::~YoloV5 ()
{
  g_free (name);
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
  GstTensorInfo *info = nullptr;
  const guint *dim;
  int i;

  info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) &config->info, 0);
  dim = info->dimension;
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
  GstTensorInfo *info = nullptr;

  numTotalBox = max_detection;
  numTotalClass = total_labels;
  cStartIdx = DEFAULT_DETECTION_NUM_INFO_YOLO5;
  cIdxMax = numTotalClass + cStartIdx;

  /* boxinput[numTotalBox][cIdxMax] */
  boxinput = (float *) input[0].data;

  /** Only support for float type model */

  info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) &config->info, 0);
  g_assert (info->type == _NNS_FLOAT32);

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

  nms (results, iou_threshold, YOLOV5_BOUNDING_BOX);
  return results;
}

/** @brief Constructor of YoloV8 */
YoloV8::YoloV8 ()
{
  scaled_output = 0;
  conf_threshold = YOLO_DETECTION_CONF_THRESHOLD;
  iou_threshold = YOLO_DETECTION_IOU_THRESHOLD;
  name = g_strdup_printf ("yolov8");
}

/** @brief Destructor of YoloV8 */
YoloV8::~YoloV8 ()
{
  g_free (name);
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
  g_autofree gchar *info_str = NULL;
  int i;
  if (!check_tensors (config, 1U)) {
    info_str = gst_tensors_info_to_string (&config->info);
    nns_loge ("Yolov8 bounding-box decoder needs at least 1 valid tensor. The given input tensor is: %s.",
        info_str);
    return FALSE;
  }
  /** Only support for float type model */
  if (config->info.info[0].type != _NNS_FLOAT32) {
    info_str = gst_tensors_info_to_string (&config->info);
    nns_loge ("Yolov8 bounding-box decoder accepts float32 input tensors only. The given input tensor is: %s.",
        info_str);
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
      info_str = gst_tensors_info_to_string (&config->info);
      nns_loge ("Yolov8 bounding-box decoder accepts RANK=2 tensors (3rd and later dimensions should be 1 or 0). The given input tensor is: %s.",
          info_str);
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

  nms (results, iou_threshold, YOLOV8_BOUNDING_BOX);
  return results;
}

/** @brief Constructor of YoloV10 */
YoloV10::YoloV10 ()
{
  conf_threshold = YOLO_DETECTION_CONF_THRESHOLD;
  name = g_strdup_printf ("yolov10");
}

/** @brief Destructor of YoloV10 */
YoloV10::~YoloV10 ()
{
  g_free (name);
}

/** @brief Set internal option of YoloV10 */
int
YoloV10::setOptionInternal (const char *param)
{
  gchar **options;
  int noptions;

  options = g_strsplit (param, ":", -1);
  noptions = g_strv_length (options);

  if (noptions > 1)
    conf_threshold = (gfloat) g_ascii_strtod (options[1], NULL);

  nns_logi ("Setting YOLOV10 decoder as conf_threshold: %.2f", conf_threshold);

  g_strfreev (options);
  return TRUE;
}

/** @brief Check compatibility of given tensors config */
int
YoloV10::checkCompatible (const GstTensorsConfig *config)
{
  const guint *dim = config->info.info[0].dimension;
  g_autofree gchar *info_str = NULL;
  int i;

  if (!check_tensors (config, 1U)) {
    info_str = gst_tensors_info_to_string (&config->info);
    nns_loge ("YoloV10 bounding-box decoder needs at least 1 valid tensor. The given input tensor is: %s.",
        info_str);
    return FALSE;
  }

  /** Only support for float type model */
  if (config->info.info[0].type != _NNS_FLOAT32) {
    info_str = gst_tensors_info_to_string (&config->info);
    nns_loge ("YoloV10 bounding-box decoder accepts float32 input tensors only. The given input tensor is: %s.",
        info_str);
    return FALSE;
  }

  /* Expected shape is 6:#MAX_DET:1 */
  if (dim[0] != 6U) {
    nns_loge ("YoloV10 boundingbox decoder requires the input shape to be 6:#MAX_DET:1. But given shape is %u:%u:1. Check the output shape of yolov10 model.",
        dim[0], dim[1]);
    return FALSE;
  }

  max_detection = dim[1];

  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    if (dim[i] != 0 && dim[i] != 1) {
      info_str = gst_tensors_info_to_string (&config->info);
      nns_loge ("YoloV10 bounding-box decoder accepts RANK=2 tensors (3rd and later dimensions should be 1 or 0). The given input tensor is: %s.",
          info_str);

      return FALSE;
    }
  }

  return TRUE;
}

/**
 * @brief Decode input memory to out buffer
 * @param[in] config The structure of input tensor info.
 * @param[in] input The array of input tensor data.
 */
GArray *
YoloV10::decode (const GstTensorsConfig *config, const GstTensorMemory *input)
{
  GArray *results = NULL;
  guint bIdx;
  float *boxinput;
  UNUSED (config);

  /* boxinput[MAX_DET][6] */
  boxinput = (float *) input[0].data;

  results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), max_detection);
  for (bIdx = 0; bIdx < max_detection; ++bIdx) {
    detectedObject object;
    float x1, x2, y1, y2, confidence, class_index;

    /* parse output of yolov10 */
    x1 = boxinput[bIdx * 6 + 0];
    y1 = boxinput[bIdx * 6 + 1];
    x2 = boxinput[bIdx * 6 + 2];
    y2 = boxinput[bIdx * 6 + 3];
    confidence = boxinput[bIdx * 6 + 4];
    class_index = boxinput[bIdx * 6 + 5];

    /* output of yolov10 is sorted */
    if (confidence < conf_threshold) {
      /* break once confidence value falls */
      break;
    }

    /* scale to given width and height */
    y1 *= (float) i_height;
    x1 *= (float) i_width;
    x2 *= (float) i_width;
    y2 *= (float) i_height;

    object.x = (int) (MAX (0.f, x1));
    object.y = (int) (MAX (0.f, y1));
    object.width = (int) (MIN ((float) i_width, x2 - x1));
    object.height = (int) (MIN ((float) i_height, y2 - y1));
    object.class_id = (int) class_index;
    object.prob = confidence;

    object.tracking_id = 0;
    object.valid = TRUE;

    if (object.class_id >= (int) total_labels) {
      nns_logw ("Class id %d is out of range (%u). Skip this object.",
          object.class_id, total_labels);
      continue;
    }

    g_array_append_val (results, object);
  }

  return results;
}

/** @brief Initialize this object for tensor decoder bounding box */
void
init_properties_yolo5 ()
{
  yolo5 = new YoloV5 ();
  BoundingBox::addProperties (yolo5);
}

/** @brief Destruct this object for tensor decoder bounding box */
void
fini_properties_yolo5 ()
{
  delete yolo5;
}

/** @brief Initialize this object for tensor decoder bounding box */
void
init_properties_yolo8 ()
{
  yolo8 = new YoloV8 ();
  BoundingBox::addProperties (yolo8);
}

/** @brief Destruct this object for tensor decoder bounding box */
void
fini_properties_yolo8 ()
{
  delete yolo8;
}

/** @brief Initialize this object for tensor decoder bounding box */
void
init_properties_yolo10 ()
{
  yolo10 = new YoloV10 ();
  BoundingBox::addProperties (yolo10);
}

/** @brief Destruct this object for tensor decoder bounding box */
void
fini_properties_yolo10 ()
{
  delete yolo10;
}
