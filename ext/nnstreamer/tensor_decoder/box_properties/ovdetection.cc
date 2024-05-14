/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor-decoder bounding box properties
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 */
/**
 * @file        ovdetection.cc
 * @date        13 May 2024
 * @brief       NNStreamer tensor-decoder bounding box properties
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Yelin Jeong <yelini.jeong@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include "../tensordec-boundingbox.h"

#define OV_PERSON_DETECTION_CONF_THRESHOLD (0.8)
#define DETECTION_MAX (200U)
#define DEFAULT_MAX_TENSORS (1)
#define DEFAULT_SIZE_DETECTION_DESC (7)

/**
 * @brief Class for OVDetection box properties
 */
class OVDetection : public BoxProperties
{
  public:
  OVDetection ();
  ~OVDetection ();
  int setOptionInternal (const char *param)
  {
    UNUSED (param);
    return TRUE;
  }
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);
};

/**
 * @brief C++-Template-like box location calculation for OpenVino Person Detection Model
 * @param[in] type The tensor type of inputptr
 * @param[in] intputptr Input tensor Data
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_persons_ov(type, inputptr, typename, results)                                                   \
  case typename:                                                                                             \
    {                                                                                                        \
      detectedObject object = {                                                                              \
        .valid = FALSE, .class_id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .prob = .0, .tracking_id = 0 \
      };                                                                                                     \
      type *typed_inputptr = (type *) inputptr;                                                              \
      guint d;                                                                                               \
                                                                                                             \
      for (d = 1; d <= DETECTION_MAX; ++d) {                                                                 \
        struct {                                                                                             \
          type image_id;                                                                                     \
          type label;                                                                                        \
          type conf;                                                                                         \
          type x_min;                                                                                        \
          type y_min;                                                                                        \
          type x_max;                                                                                        \
          type y_max;                                                                                        \
        } desc;                                                                                              \
                                                                                                             \
        memcpy (&desc, typed_inputptr, sizeof (desc));                                                       \
        typed_inputptr += (sizeof (desc) / sizeof (type));                                                   \
        object.valid = FALSE;                                                                                \
                                                                                                             \
        if ((int) desc.image_id < 0) {                                                                       \
          max_detection = (d - 1);                                                                           \
          break;                                                                                             \
        }                                                                                                    \
        object.class_id = -1;                                                                                \
        object.x = (int) (desc.x_min * (type) i_width);                                                      \
        object.y = (int) (desc.y_min * (type) i_height);                                                     \
        object.width = (int) ((desc.x_max - desc.x_min) * (type) i_width);                                   \
        object.height = (int) ((desc.y_max - desc.y_min) * (type) i_height);                                 \
        if (desc.conf < OV_PERSON_DETECTION_CONF_THRESHOLD)                                                  \
          continue;                                                                                          \
        object.prob = 1;                                                                                     \
        object.valid = TRUE;                                                                                 \
        g_array_append_val (results, object);                                                                \
      }                                                                                                      \
    }                                                                                                        \
    break

static BoxProperties *ov_detection = nullptr;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_properties_ovdetection (void) __attribute__ ((constructor));
void fini_properties_ovdetection (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/** @brief Constructor of OVDetection */
OVDetection::OVDetection ()
{
  name = g_strdup_printf ("ov-person-detection");
}

/** @brief Destructor of OVDetection */
OVDetection::~OVDetection ()
{
  g_free (name);
}

/** @brief Check compatibility of given tensors config */
int
OVDetection::checkCompatible (const GstTensorsConfig *config)
{
  const guint *dim;
  int i;
  UNUSED (total_labels);

  if (!check_tensors (config, DEFAULT_MAX_TENSORS))
    return FALSE;

  /**
   * The shape of the ouput tensor is [7, N, 1, 1], where N is the maximum
   * number (i.e., 200) of detected bounding boxes.
   */
  dim = config->info.info[0].dimension;
  g_return_val_if_fail (dim[0] == DEFAULT_SIZE_DETECTION_DESC, FALSE);
  g_return_val_if_fail (dim[1] == DETECTION_MAX, FALSE);
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
OVDetection::decode (const GstTensorsConfig *config, const GstTensorMemory *input)
{
  GArray *results = NULL;
  const guint num_tensors = config->info.num_tensors;

  /* Already checked with getOutCaps. Thus, this is an internal bug */
  g_assert (num_tensors >= DEFAULT_MAX_TENSORS);

  results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), DETECTION_MAX);
  switch (config->info.info[0].type) {
    _get_persons_ov (uint8_t, input[0].data, _NNS_UINT8, results);
    _get_persons_ov (int8_t, input[0].data, _NNS_INT8, results);
    _get_persons_ov (uint16_t, input[0].data, _NNS_UINT16, results);
    _get_persons_ov (int16_t, input[0].data, _NNS_INT16, results);
    _get_persons_ov (uint32_t, input[0].data, _NNS_UINT32, results);
    _get_persons_ov (int32_t, input[0].data, _NNS_INT32, results);
    _get_persons_ov (uint64_t, input[0].data, _NNS_UINT64, results);
    _get_persons_ov (int64_t, input[0].data, _NNS_INT64, results);
    _get_persons_ov (float, input[0].data, _NNS_FLOAT32, results);
    _get_persons_ov (double, input[0].data, _NNS_FLOAT64, results);
    default:
      g_assert (0);
  }
  return results;
}

/** @brief Initialize this object for tensor decoder bounding box */
void
init_properties_ovdetection ()
{
  ov_detection = new OVDetection ();
  BoundingBox::addProperties (ov_detection);
}

/** @brief Destruct this object for tensor decoder bounding box */
void
fini_properties_ovdetection ()
{
  delete ov_detection;
}
