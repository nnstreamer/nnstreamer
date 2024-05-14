/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor-decoder bounding box properties
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 */
/**
 * @file        mobilenetssd.cc
 * @date        13 May 2024
 * @brief       NNStreamer tensor-decoder bounding box properties
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Yelin Jeong <yelini.jeong@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include "../tensordec-boundingbox.h"

#define MAX_TENSORS (2U)

#define THRESHOLD_IDX (0)
#define Y_SCALE_IDX (1)
#define X_SCALE_IDX (2)
#define H_SCALE_IDX (3)
#define W_SCALE_IDX (4)
#define IOU_THRESHOLD_IDX (5)

#define DETECTION_THRESHOLD_DEFAULT (0.5f)
#define THRESHOLD_IOU_DEFAULT (0.5f)
#define Y_SCALE_DEFAULT (10.0f)
#define X_SCALE_DEFAULT (10.0f)
#define H_SCALE_DEFAULT (5.0f)
#define W_SCALE_DEFAULT (5.0f)

#define BOX_SIZE (4)
#define DETECTION_MAX (2034) /* add ssd_mobilenet v3 support */
#define PARAMS_MAX (6)

#define _expit(x) (1.f / (1.f + expf (-((float) x))))

/**
 * @brief Class for MobilenetSSD box properties
 */
class MobilenetSSD : public BoxProperties
{
  public:
  MobilenetSSD ();
  ~MobilenetSSD ();
  int mobilenet_ssd_loadBoxPrior ();

  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  private:
  char *box_prior_path; /**< Box Prior file path */
  gfloat box_priors[BOX_SIZE][DETECTION_MAX + 1]; /** loaded box prior */
  gfloat params[PARAMS_MAX]; /** Post Processing parameters */
  gfloat sigmoid_threshold; /** Inverse value of valid detection threshold in sigmoid domain */
};

/**
 * @brief C++-Template-like box location calculation for box-priors
 * @bug This is not macro-argument safe. Use paranthesis!
 * @param[in] bb The configuration, "bounding_boxes"
 * @param[in] index The index (3rd dimension of BOX_SIZE:1:DETECTION_MAX:1)
 * @param[in] total_labels The count of total labels. We can get this from input tensor info. (1st dimension of LABEL_SIZE:DETECTION_MAX:1:1)
 * @param[in] boxprior The box prior data from the box file of SSD.
 * @param[in] boxinputptr Cursor pointer of input + byte-per-index * index (box)
 * @param[in] detinputptr Cursor pointer of input + byte-per-index * index (detection)
 * @param[in] result The object returned. (pointer to object)
 */
#define _get_object_i_mobilenet_ssd(index, total_labels, boxprior,                \
    boxinputptr, detinputptr, result, i_width, i_height)                          \
  do {                                                                            \
    unsigned int c;                                                               \
    gfloat highscore = -FLT_MAX;                                                  \
    float y_scale = params[Y_SCALE_IDX];                                          \
    float x_scale = params[X_SCALE_IDX];                                          \
    float h_scale = params[H_SCALE_IDX];                                          \
    float w_scale = params[W_SCALE_IDX];                                          \
    result->valid = FALSE;                                                        \
    for (c = 1; c < total_labels; c++) {                                          \
      if (detinputptr[c] >= sigmoid_threshold) {                                  \
        gfloat score = _expit (detinputptr[c]);                                   \
        float ycenter                                                             \
            = boxinputptr[0] / y_scale * boxprior[2][index] + boxprior[0][index]; \
        float xcenter                                                             \
            = boxinputptr[1] / x_scale * boxprior[3][index] + boxprior[1][index]; \
        float h = (float) expf (boxinputptr[2] / h_scale) * boxprior[2][index];   \
        float w = (float) expf (boxinputptr[3] / w_scale) * boxprior[3][index];   \
        float ymin = ycenter - h / 2.f;                                           \
        float xmin = xcenter - w / 2.f;                                           \
        int x = xmin * i_width;                                                   \
        int y = ymin * i_height;                                                  \
        int width = w * i_width;                                                  \
        int height = h * i_height;                                                \
        if (highscore < score) {                                                  \
          result->class_id = c;                                                   \
          result->x = MAX (0, x);                                                 \
          result->y = MAX (0, y);                                                 \
          result->width = width;                                                  \
          result->height = height;                                                \
          result->prob = score;                                                   \
          result->valid = TRUE;                                                   \
        }                                                                         \
      }                                                                           \
    }                                                                             \
  } while (0);

/**
 * @brief C++-Template-like box location calculation for box-priors for Mobilenet SSD Model
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] boxprior The box prior data from the box file of MOBILENET_SSD.
 * @param[in] boxinput Input Tensor Data (Boxes)
 * @param[in] detinput Input Tensor Data (Detection). Null if not available. (numtensor ==1)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mobilenet_ssd(_type, typename, boxprior, boxinput,                                      \
    detinput, config, results, i_width, i_height, max_detection)                                             \
  case typename:                                                                                             \
    {                                                                                                        \
      int d;                                                                                                 \
      _type *boxinput_ = (_type *) boxinput;                                                                 \
      size_t boxbpi = config->info.info[0].dimension[0];                                                     \
      _type *detinput_ = (_type *) detinput;                                                                 \
      size_t detbpi = config->info.info[1].dimension[0];                                                     \
      int num = (DETECTION_MAX > max_detection) ? max_detection : DETECTION_MAX;                             \
      detectedObject object = {                                                                              \
        .valid = FALSE, .class_id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .prob = .0, .tracking_id = 0 \
      };                                                                                                     \
      for (d = 0; d < num; d++) {                                                                            \
        _get_object_i_mobilenet_ssd (d, detbpi, boxprior, (boxinput_ + (d * boxbpi)),                        \
            (detinput_ + (d * detbpi)), (&object), i_width, i_height);                                       \
        if (object.valid == TRUE) {                                                                          \
          g_array_append_val (results, object);                                                              \
        }                                                                                                    \
      }                                                                                                      \
    }                                                                                                        \
    break


/** @brief Macro to simplify calling _get_objects_mobilenet_ssd */
#define _get_objects_mobilenet_ssd_(type, typename)                      \
  _get_objects_mobilenet_ssd (type, typename, box_priors, (boxes->data), \
      (detections->data), config, results, i_width, i_height, max_detection)

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

static BoxProperties *mobilenet = nullptr;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_properties_mobilenetssd (void) __attribute__ ((constructor));
void fini_properties_mobilenetssd (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/** @brief Constructor of MobilenetSSD */
MobilenetSSD::MobilenetSSD ()
{
  params[THRESHOLD_IDX] = DETECTION_THRESHOLD_DEFAULT;
  params[Y_SCALE_IDX] = Y_SCALE_DEFAULT;
  params[X_SCALE_IDX] = X_SCALE_DEFAULT;
  params[H_SCALE_IDX] = H_SCALE_DEFAULT;
  params[W_SCALE_IDX] = W_SCALE_DEFAULT;
  params[IOU_THRESHOLD_IDX] = THRESHOLD_IOU_DEFAULT;
  sigmoid_threshold = logit (DETECTION_THRESHOLD_DEFAULT);

  max_detection = 0;
  total_labels = 0;
  box_prior_path = nullptr;
  name = g_strdup_printf ("mobilenet-ssd");
}

/** @brief Destructor of MobilenetSSD */
MobilenetSSD::~MobilenetSSD ()
{
  g_free (name);
}

/**
 * @brief Load box-prior data from a file
 * @param[in/out] bdata The internal data.
 * @return TRUE if loaded and configured. FALSE if failed to do so.
 */
int
MobilenetSSD::mobilenet_ssd_loadBoxPrior ()
{
  gboolean failed = FALSE;
  GError *err = NULL;
  gchar **priors;
  gchar *line = NULL;
  gchar *contents = NULL;
  guint row;
  gint prev_reg = -1;

  /* Read file contents */
  if (!g_file_get_contents (box_prior_path, &contents, NULL, &err)) {
    GST_ERROR ("Decoder/Bound-Box/SSD's box prior file %s cannot be read: %s",
        box_prior_path, err->message);
    g_clear_error (&err);
    return FALSE;
  }

  priors = g_strsplit (contents, "\n", -1);
  /* If given prior file is inappropriate, report back to tensor-decoder */
  if (g_strv_length (priors) < BOX_SIZE) {
    ml_loge ("The given prior file, %s, should have at least %d lines.\n",
        box_prior_path, BOX_SIZE);
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
          if (registered > DETECTION_MAX) {
            GST_WARNING ("Decoder/Bound-Box/SSD's box prior data file has too many priors. %d >= %d",
                registered, DETECTION_MAX);
            break;
          }
          box_priors[row][registered] = (gfloat) g_ascii_strtod (word, NULL);
          registered++;
        }
      }

      g_strfreev (list);
    }

    if (prev_reg != -1 && prev_reg != registered) {
      GST_ERROR ("Decoder/Bound-Box/SSD's box prior data file is not consistent.");
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

/** @brief Set internal option of MobilenetSSD
 *  @param[in] param The option string.
 */
int
MobilenetSSD::setOptionInternal (const char *param)
{
  gchar **options;
  int noptions, idx;
  int ret = 1;

  options = g_strsplit (param, ":", -1);
  noptions = g_strv_length (options);

  if (noptions > (PARAMS_MAX + 1))
    noptions = PARAMS_MAX + 1;

  if (box_prior_path) {
    g_free (box_prior_path);
    box_prior_path = nullptr;
  }

  box_prior_path = g_strdup (options[0]);

  if (NULL != box_prior_path) {
    ret = mobilenet_ssd_loadBoxPrior ();
    if (ret == 0)
      goto exit_mobilenet_ssd;
  }

  for (idx = 1; idx < noptions; idx++) {
    if (strlen (options[idx]) == 0)
      continue;
    params[idx - 1] = strtod (options[idx], NULL);
  }

  sigmoid_threshold = logit (params[THRESHOLD_IDX]);

  return TRUE;

exit_mobilenet_ssd:
  g_strfreev (options);
  return ret;
}

/** @brief Check compatibility of given tensors config
 *  @param[in] config The tensors config to check compatibility
 */
int
MobilenetSSD::checkCompatible (const GstTensorsConfig *config)
{
  const uint32_t *dim1, *dim2;
  int i;
  guint max_detection, max_label;

  if (!check_tensors (config, MAX_TENSORS))
    return FALSE;

  /* Check if the first tensor is compatible */
  dim1 = config->info.info[0].dimension;
  g_return_val_if_fail (dim1[0] == BOX_SIZE, FALSE);
  g_return_val_if_fail (dim1[1] == 1, FALSE);
  max_detection = dim1[2];
  g_return_val_if_fail (max_detection > 0, FALSE);

  /** @todo unused dimension value should be 0 */
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim1[i] == 0 || dim1[i] == 1, FALSE);

  /* Check if the second tensor is compatible */
  dim2 = config->info.info[1].dimension;

  max_label = dim2[0];
  g_return_val_if_fail (max_label <= total_labels, FALSE);
  if (max_label < total_labels)
    GST_WARNING ("The given tensor (2nd) has max_label (first dimension: %u) smaller than the number of labels in labels file (%u).",
        max_label, total_labels);
  g_return_val_if_fail (max_detection == dim2[1], FALSE);
  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim2[i] == 0 || dim2[i] == 1, FALSE);

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
MobilenetSSD::decode (const GstTensorsConfig *config, const GstTensorMemory *input)
{
  const GstTensorMemory *boxes, *detections = NULL;
  GArray *results;
  const guint num_tensors = config->info.num_tensors;

  /**
   * @todo 100 is a heuristic number of objects in a picture frame
   *       We may have better "heuristics" than this.
   *       For the sake of performance, don't make it too small.
   */

  /* Already checked with getOutCaps. Thus, this is an internal bug */
  g_assert (num_tensors >= MAX_TENSORS);
  results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), 100);

  boxes = &input[0];
  if (num_tensors >= MAX_TENSORS) /* lgtm[cpp/constant-comparison] */
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
  nms (results, params[IOU_THRESHOLD_IDX]);
  return results;
}

/** @brief Initialize this object for tensor decoder bounding box */
void
init_properties_mobilenetssd ()
{
  mobilenet = new MobilenetSSD ();
  BoundingBox::addProperties (mobilenet);
}

/** @brief Destruct this object for tensor decoder bounding box */
void
fini_properties_mobilenetssd ()
{
  delete mobilenet;
}
