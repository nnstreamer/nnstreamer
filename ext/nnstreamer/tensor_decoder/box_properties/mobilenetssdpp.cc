/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor-decoder bounding box properties
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 */
/**
 * @file        mobilenetssdpp.cc
 * @date        13 May 2024
 * @brief       NNStreamer tensor-decoder bounding box properties
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Yelin Jeong <yelini.jeong@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include <stdio.h>
#include "../tensordec-boundingbox.h"

#define BOX_SIZE (4)
#define DETECTION_MAX (100)
#define LOCATIONS_IDX (0)
#define CLASSES_IDX (1)
#define SCORES_IDX (2)
#define NUM_IDX (3)
#define MAX_TENSORS (4U)

#define LOCATIONS_DEFAULT (3)
#define CLASSES_DEFAULT (1)
#define SCORES_DEFAULT (2)
#define NUM_DEFAULT (0)
#define THRESHOLD_DEFAULT (G_MINFLOAT)

/**
 * @brief Class for MobilenetSSDPP box properties
 */
class MobilenetSSDPP : public BoxProperties
{
  public:
  MobilenetSSDPP ();
  ~MobilenetSSDPP ();
  int get_mobilenet_ssd_pp_tensor_idx (int idx);

  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  private:
  gint tensor_mapping[MAX_TENSORS]; /* Output tensor index mapping */
  gfloat threshold; /* Detection threshold */
};

/**
 * @brief C++-Template-like box location calculation for Tensorflow SSD model
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] numinput Input Tensor Data (The number of detections)
 * @param[in] classinput Input Tensor Data (Detected classes)
 * @param[in] scoreinput Input Tensor Data (Detection scores)
 * @param[in] boxesinput Input Tensor Data (Boxes)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mobilenet_ssd_pp(_type, typename, numinput, classinput,       \
    scoreinput, boxesinput, config, results, i_width, i_height)                    \
  case typename:                                                                   \
    {                                                                              \
      int d, num;                                                                  \
      size_t boxbpi;                                                               \
      _type *num_detection_ = (_type *) numinput;                                  \
      _type *classes_ = (_type *) classinput;                                      \
      _type *scores_ = (_type *) scoreinput;                                       \
      _type *boxes_ = (_type *) boxesinput;                                        \
      int locations_idx                                                            \
          = get_mobilenet_ssd_pp_tensor_idx (MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS); \
      num = (int) num_detection_[0];                                               \
      results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), num);     \
      boxbpi = config->info.info[locations_idx].dimension[0];                      \
      for (d = 0; d < num; d++) {                                                  \
        _type x1, x2, y1, y2;                                                      \
        detectedObject object;                                                     \
        if (scores_[d] < threshold)                                                \
          continue;                                                                \
        object.valid = TRUE;                                                       \
        object.class_id = (int) classes_[d];                                       \
        x1 = MIN (MAX (boxes_[d * boxbpi + 1], 0), 1);                             \
        y1 = MIN (MAX (boxes_[d * boxbpi], 0), 1);                                 \
        x2 = MIN (MAX (boxes_[d * boxbpi + 3], 0), 1);                             \
        y2 = MIN (MAX (boxes_[d * boxbpi + 2], 0), 1);                             \
        object.x = (int) (x1 * i_width);                                           \
        object.y = (int) (y1 * i_height);                                          \
        object.width = (int) ((x2 - x1) * i_width);                                \
        object.height = (int) ((y2 - y1) * i_height);                              \
        object.prob = scores_[d];                                                  \
        g_array_append_val (results, object);                                      \
      }                                                                            \
    }                                                                              \
    break

/** @brief Macro to simplify calling _get_objects_mobilenet_ssd_pp */
#define _get_objects_mobilenet_ssd_pp_(type, typename)                                 \
  _get_objects_mobilenet_ssd_pp (type, typename, (mem_num->data), (mem_classes->data), \
      (mem_scores->data), (mem_boxes->data), config, results, i_width, i_height)

static BoxProperties *mobilenetpp = nullptr;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_properties_mobilenetssd_pp (void) __attribute__ ((constructor));
void fini_properties_mobilenetssd_pp (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief MOBILENET SSD PostProcess Output tensor feature mapping.
 */
typedef enum {
  MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS = 0,
  MOBILENET_SSD_PP_BBOX_IDX_CLASSES = 1,
  MOBILENET_SSD_PP_BBOX_IDX_SCORES = 2,
  MOBILENET_SSD_PP_BBOX_IDX_NUM = 3,
  MOBILENET_SSD_PP_BBOX_IDX_UNKNOWN
} mobilenet_ssd_pp_bbox_idx_t;


/** @brief Constructor of MobilenetSSDPP */
MobilenetSSDPP::MobilenetSSDPP ()
{
  tensor_mapping[LOCATIONS_IDX] = LOCATIONS_DEFAULT;
  tensor_mapping[CLASSES_IDX] = CLASSES_DEFAULT;
  tensor_mapping[SCORES_IDX] = SCORES_DEFAULT;
  tensor_mapping[NUM_IDX] = NUM_DEFAULT;
  threshold = THRESHOLD_DEFAULT;
  name = g_strdup_printf ("mobilenet-ssd-postprocess");
}


/** @brief Destructor of MobilenetSSDPP */
MobilenetSSDPP::~MobilenetSSDPP ()
{
  g_free (name);
}

/** @brief Helper to retrieve tensor index by feature */
int
MobilenetSSDPP::get_mobilenet_ssd_pp_tensor_idx (int idx)
{
  return tensor_mapping[idx];
}

/** @brief Set internal option of MobilenetSSDPP
 *  @param[in] param The option string.
 */
int
MobilenetSSDPP::setOptionInternal (const char *param)
{
  int threshold_percent;
  int ret = sscanf (param, "%i:%i:%i:%i,%i", &tensor_mapping[LOCATIONS_IDX],
      &tensor_mapping[CLASSES_IDX], &tensor_mapping[SCORES_IDX],
      &tensor_mapping[NUM_IDX], &threshold_percent);

  if ((ret == EOF) || (ret < 5)) {
    GST_ERROR ("Invalid options, must be \"locations idx:classes idx:scores idx:num idx,threshold\"");
    return FALSE;
  }

  GST_INFO ("MOBILENET SSD POST PROCESS output tensors mapping: "
            "locations idx (%d), classes idx (%d), scores idx (%d), num detections idx (%d)",
      tensor_mapping[LOCATIONS_IDX], tensor_mapping[CLASSES_IDX],
      tensor_mapping[SCORES_IDX], tensor_mapping[NUM_IDX]);

  if ((threshold_percent > 100) || (threshold_percent < 0)) {
    GST_ERROR ("Invalid MOBILENET SSD POST PROCESS threshold detection (%i), must be in range [0 100]",
        threshold_percent);
  } else {
    threshold = threshold_percent / 100.0;
  }

  GST_INFO ("MOBILENET SSD POST PROCESS object detection threshold: %.2f", threshold);

  return TRUE;
}

/** @brief Check compatibility of given tensors config */
int
MobilenetSSDPP::checkCompatible (const GstTensorsConfig *config)
{
  const uint32_t *dim1, *dim2, *dim3, *dim4;
  int locations_idx, classes_idx, scores_idx, num_idx, i;

  if (!check_tensors (config, MAX_TENSORS))
    return FALSE;

  locations_idx = get_mobilenet_ssd_pp_tensor_idx (LOCATIONS_IDX);
  classes_idx = get_mobilenet_ssd_pp_tensor_idx (CLASSES_IDX);
  scores_idx = get_mobilenet_ssd_pp_tensor_idx (SCORES_IDX);
  num_idx = get_mobilenet_ssd_pp_tensor_idx (NUM_IDX);

  /* Check if the number of detections tensor is compatible */
  dim1 = config->info.info[num_idx].dimension;
  g_return_val_if_fail (dim1[0] == 1, FALSE);
  for (i = 1; i < NNS_TENSOR_RANK_LIMIT; ++i)
    g_return_val_if_fail (dim1[i] == 0 || dim1[i] == 1, FALSE);

  /* Check if the classes & scores tensors are compatible */
  dim2 = config->info.info[classes_idx].dimension;
  dim3 = config->info.info[scores_idx].dimension;
  g_return_val_if_fail (dim3[0] == dim2[0], FALSE);
  max_detection = dim2[0];
  for (i = 1; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    g_return_val_if_fail (dim2[i] == 0 || dim2[i] == 1, FALSE);
    g_return_val_if_fail (dim3[i] == 0 || dim3[i] == 1, FALSE);
  }

  /* Check if the bbox locations tensor is compatible */
  dim4 = config->info.info[locations_idx].dimension;
  g_return_val_if_fail (BOX_SIZE == dim4[0], FALSE);
  g_return_val_if_fail (max_detection == dim4[1], FALSE);
  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; ++i)
    g_return_val_if_fail (dim4[i] == 0 || dim4[i] == 1, FALSE);

  /* Check consistency with max_detection */
  if (this->max_detection == 0)
    this->max_detection = max_detection;
  else
    g_return_val_if_fail (max_detection == this->max_detection, FALSE);

  if (this->max_detection > DETECTION_MAX) {
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
MobilenetSSDPP::decode (const GstTensorsConfig *config, const GstTensorMemory *input)
{

  const GstTensorMemory *mem_num, *mem_classes, *mem_scores, *mem_boxes;
  int locations_idx, classes_idx, scores_idx, num_idx;
  GArray *results = NULL;
  const guint num_tensors = config->info.num_tensors;

  /* Already checked with getOutCaps. Thus, this is an internal bug */
  g_assert (num_tensors >= MAX_TENSORS);

  locations_idx = get_mobilenet_ssd_pp_tensor_idx (LOCATIONS_IDX);
  classes_idx = get_mobilenet_ssd_pp_tensor_idx (CLASSES_IDX);
  scores_idx = get_mobilenet_ssd_pp_tensor_idx (SCORES_IDX);
  num_idx = get_mobilenet_ssd_pp_tensor_idx (NUM_IDX);

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
  return results;
}

/** @brief Initialize this object for tensor decoder bounding box */
void
init_properties_mobilenetssd_pp ()
{
  mobilenetpp = new MobilenetSSDPP ();
  BoundingBox::addProperties (mobilenetpp);
}

/** @brief Destruct this object for tensor decoder bounding box */
void
fini_properties_mobilenetssd_pp ()
{
  delete mobilenetpp;
}
