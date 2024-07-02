/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "tensor_region"
 * Copyright (C) 2023 Harsh Jain <hjain24in@gmail.com>
 */
/**
 * @file        tensordec-tensor_region.c
 * @date        15th June, 2023
 * @brief       NNStreamer tensor-decoder subplugin, "tensor region",
 *              which converts tensors to cropping info for tensor _crop element
 *              This code is NYI/WIP and not compilable.
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Harsh Jain <hjain24in@gmail.com>
 * @bug         No known bugs except for NYI items
 *
 * option1: number of cropping regions required (default is 1)
 * option2: Location of label file
 *          This is independent from option1
 * option3:
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
 * option4: Video input Dimension (WIDTH:HEIGHT) (default 300:300)
 *          This is independent from option1
 *
 * @todo Remove duplicate codes*
 * @todo give support for other models*
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <glib.h>
#include <gst/gst.h>
#include <math.h> /** expf */
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_util.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensordecutil.h"

#define _tensor_region_size_default_ 1
void init_tr (void) __attribute__ ((constructor));
void fini_tr (void) __attribute__ ((destructor));

#define BOX_SIZE (4)
#define MOBILENET_SSD_DETECTION_MAX (2034) /**add ssd_mobilenet v3 */
#define MOBILENET_SSD_MAX_TENSORS (2U)
#define INPUT_VIDEO_WIDTH_DEFAULT (300)
#define INPUT_VIDEO_HEIGHT_DEFAULT (300)
/**
 * @brief There can be different schemes for input tensor.
 */
typedef enum
{
  MOBILENET_SSD_BOUNDING_BOX = 0,
  BOUNDING_BOX_UNKNOWN,
} tensor_region_modes;

/** @brief Internal data structure for identifying cropping region */
typedef struct {
  guint x;
  guint y;
  guint w;
  guint h;
} crop_region;

/**
 * @brief Data structure for SSD tensor_region info for mobilenet ssd model.
 */
typedef struct {
  /**From option3, box prior data */
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

/** @brief Internal data structure for tensor region */
typedef struct {
  tensor_region_modes mode; /**< When a mode is being changed, _cleanup_mode_properties () should be called before changing it */
  union {
    properties_MOBILENET_SSD mobilenet_ssd; /**< Properties for mobilenet_ssd  */
  };
  imglabel_t labeldata;
  char *label_path;

  /**From option4 */
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */

  /**From option1 */
  guint num;  /**number of cropped regions required */

  guint max_detections;
  GArray *regions;
  gboolean flag_use_label;

} tensor_region;

/** @brief Internal function for mode change preparation */
static void
_cleanup_mode_properties (tensor_region *tr)
{
  switch (tr->mode) {
  case MOBILENET_SSD_BOUNDING_BOX: {
    properties_MOBILENET_SSD *mobilenet_ssd = &tr->mobilenet_ssd;
    g_free (mobilenet_ssd->box_prior_path);
    break;
  }
  default:
    break;
  }
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

/**
 * @brief Load box-prior data from a file
 * @param[in/out] trData The internal data.
 * @return TRUE if loaded and configured. FALSE if failed to do so.
 */
static int
_mobilenet_ssd_loadBoxPrior (tensor_region *trData)
{
  properties_MOBILENET_SSD *mobilenet_ssd = &trData->mobilenet_ssd;
  gboolean failed = FALSE;
  GError *err = NULL;
  gchar **priors;
  gchar *line = NULL;
  gchar *contents = NULL;
  guint row;
  gint prev_reg = -1;

  /**Read file contents */
  if (!g_file_get_contents (mobilenet_ssd->box_prior_path, &contents, NULL, &err)) {
    GST_ERROR ("Decoder/Tensor-Region/SSD's box prior file %s cannot be read: %s",
        mobilenet_ssd->box_prior_path, err->message);
    g_clear_error (&err);
    return FALSE;
  }

  priors = g_strsplit (contents, "\n", -1);
  /**If given prior file is inappropriate, report back to tensor-decoder */
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
            GST_WARNING ("Decoder/Tensor-region/SSD's box prior data file has too many priors. %d >= %d",
                registered, MOBILENET_SSD_DETECTION_MAX);
            break;
          }
          mobilenet_ssd->box_priors[row][registered]
              = (gfloat) g_ascii_strtod (word, NULL);
          registered++;
        }
      }

      g_strfreev (list);
    }

    if (prev_reg != -1 && prev_reg != registered) {
      GST_ERROR ("Decoder/Tensor-Region/SSD's box prior data file is not consistent.");
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


/** @brief Initialize tensor_region per mode */
static int
_init_modes (tensor_region * trData){
  if(trData->mode == MOBILENET_SSD_BOUNDING_BOX){
    properties_MOBILENET_SSD *data = &trData->mobilenet_ssd;
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
  }
  return TRUE;
}


/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
tr_init (void **pdata)
{
  tensor_region *trData;
  trData = *pdata = g_new0 (tensor_region, 1);
  if (*pdata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }
  trData->mode = MOBILENET_SSD_BOUNDING_BOX;
  trData->num = 1;
  trData->max_detections = 0;
  trData->regions = NULL;
  trData->flag_use_label = FALSE;
  trData->i_width = INPUT_VIDEO_WIDTH_DEFAULT;
  trData->i_height = INPUT_VIDEO_HEIGHT_DEFAULT;
  return _init_modes(trData);
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
tr_exit (void **pdata)
{
  tensor_region *trData = *pdata;
  g_array_free (trData->regions, TRUE);
  _free_labels (&trData->labeldata);
  _cleanup_mode_properties (trData);

  if (trData->label_path)
    g_free (trData->label_path);
  g_free (*pdata);
  *pdata = NULL;
}

/** @brief configure per-mode option (option1) */
static int
_setOption_mode (tensor_region *trData, const char *param)
{
  if (trData->mode == MOBILENET_SSD_BOUNDING_BOX) {
    properties_MOBILENET_SSD *mobilenet_ssd = &trData->mobilenet_ssd;
    gchar **options;
    int noptions, idx;
    int ret = 1;

    options = g_strsplit (param, ":", -1);
    noptions = g_strv_length (options);

    if (mobilenet_ssd->box_prior_path)
      g_free (mobilenet_ssd->box_prior_path);

    mobilenet_ssd->box_prior_path = g_strdup (options[0]);

    if (NULL != mobilenet_ssd->box_prior_path) {
      ret = _mobilenet_ssd_loadBoxPrior (trData);
      if (ret == 0) {
        g_strfreev (options);
        return ret;
      }
    }

    for (idx = 1; idx < noptions; idx++) {
      if (strlen (options[idx]) == 0)
        continue;
      mobilenet_ssd->params[idx - 1] = strtod (options[idx], NULL);
    }

    mobilenet_ssd->sigmoid_threshold
        = logit (mobilenet_ssd->params[MOBILENET_SSD_PARAMS_THRESHOLD_IDX]);
    g_strfreev (options);
    return ret;
  }
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
tr_setOption (void **pdata, int opNum, const char *param)
{
  tensor_region *trData = *pdata;
  if (opNum == 0) {
    /**option1 number of crop regions required */
    trData->num = atoi (param);
    return TRUE;
  } else if (opNum == 1) {
    /**option2 label path for mobilenet_ssd model */
    if (NULL != trData->label_path)
      g_free (trData->label_path);
    trData->label_path = g_strdup (param);

    if (NULL != trData->label_path)
      loadImageLabels (trData->label_path, &trData->labeldata);

    if (trData->labeldata.total_labels > 0)
      return TRUE;
    else
      return FALSE;
  } else if (opNum == 2){
    /**option3 setting box prior path for mobilenet ssd model */
    return _setOption_mode (trData, param);
  }
  else if (opNum == 3) {
    /**option4 = input model size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    trData->i_width = 0;
    trData->i_height = 0;
    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR ("mode-option-4 of tensor region is input video dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE; /**Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING ("mode-option-4 of tensor region is input video dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    trData->i_width = dim[0];
    trData->i_height = dim[1];
    return TRUE;
  }

  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

typedef struct {
  int valid;
  int class_id;
  gfloat score;
  int x;
  int y;
  int height;
  int width;
} detected_object;

/**
 * @brief transfer crop region info with the given results to the output buffer
 * @param[out] out_info The output buffer
 * @param[in] data The Tensor_region internal data.
 * @param[in] results The final results to be transferred.
 */
static void
gst_tensor_top_detectedObjects_cropInfo (GstMapInfo *out_info, const tensor_region *data, GArray *results)
{

  guint i;
  gsize size = sizeof (crop_region); /**Assuming crop_region is a structure with four integer fields */
  guint *out_data = (guint *) out_info->data;
  crop_region region;
  guint maxx = MIN (results->len, data->num);
  for (i = 0; i < maxx; i++) {
    detected_object *temp = &g_array_index (results, detected_object, i);
    region.x = temp->x;
    region.y = temp->y;
    region.w = temp->width;
    region.h = temp->height;
    memcpy (out_data, &region, size);
    out_data += size / sizeof (guint);
  }
}

#define _expit(x) (1.f / (1.f + expf (-((float) x))))


/**
 * @brief C++-Template-like box location calculation for box-priors
 * @bug This is not macro-argument safe. Use parenthesis!
 * @param[in] bb The configuration, "tensor region"
 * @param[in] index The index (3rd dimension of BOX_SIZE:1:MOBILENET_SSD_DETECTION_MAX:1)
 * @param[in] total_labels The count of total labels. We can get this from input tensor info. (1st dimension of LABEL_SIZE:MOBILENET_SSD_DETECTION_MAX:1:1)
 * @param[in] boxprior The box prior data from the box file of SSD.
 * @param[in] boxinputptr Cursor pointer of input + byte-per-index * index (box)
 * @param[in] detinputptr Cursor pointer of input + byte-per-index * index (detection)
 * @param[in] result The object returned. (pointer to object)
 */
#define _get_object_i_mobilenet_ssd(                                              \
    bb, index, total_labels, boxprior, boxinputptr, detinputptr, result)          \
  do {                                                                            \
    unsigned int c;                                                               \
    properties_MOBILENET_SSD *data = &bb->mobilenet_ssd;                          \
    float sigmoid_threshold = data->sigmoid_threshold;                            \
    float y_scale = data->params[MOBILENET_SSD_PARAMS_Y_SCALE_IDX];               \
    float x_scale = data->params[MOBILENET_SSD_PARAMS_X_SCALE_IDX];               \
    float h_scale = data->params[MOBILENET_SSD_PARAMS_H_SCALE_IDX];               \
    float w_scale = data->params[MOBILENET_SSD_PARAMS_W_SCALE_IDX];               \
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
        int x = xmin * bb->i_width;                                               \
        int y = ymin * bb->i_height;                                              \
        int width = w * bb->i_width;                                              \
        int height = h * bb->i_height;                                            \
        result->class_id = c;                                                     \
        result->x = MAX (0, x);                                                   \
        result->y = MAX (0, y);                                                   \
        result->width = width;                                                    \
        result->height = height;                                                  \
        result->score = score;                                                    \
        result->valid = TRUE;                                                     \
        break;                                                                    \
      }                                                                           \
    }                                                                             \
  } while (0);

/**
 * @brief C++-Template-like box location calculation for box-priors for Mobilenet SSD Model
 * @param[in] bb The configuration, "tensor region"
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] boxprior The box prior data from the box file of MOBILENET_SSD.
 * @param[in] boxinput Input Tensor Data (Boxes)
 * @param[in] detinput Input Tensor Data (Detection). Null if not available. (numtensor ==1)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mobilenet_ssd(                                                         \
    bb, _type, typename, boxprior, boxinput, detinput, config, results)                     \
  case typename:                                                                            \
    {                                                                                       \
      int d;                                                                                \
      _type *boxinput_ = (_type *) boxinput;                                                \
      size_t boxbpi = config->info.info[0].dimension[0];                                    \
      _type *detinput_ = (_type *) detinput;                                                \
      size_t detbpi = config->info.info[1].dimension[0];                                    \
      int num = (MOBILENET_SSD_DETECTION_MAX > bb->max_detections) ?                        \
                    bb->max_detections :                                                    \
                    MOBILENET_SSD_DETECTION_MAX;                                            \
      detected_object object = {                                                            \
        .valid = FALSE, .class_id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .score = .0 \
      };                                                                                    \
      for (d = 0; d < num; d++) {                                                           \
        _get_object_i_mobilenet_ssd (bb, d, detbpi, boxprior,                               \
            (boxinput_ + (d * boxbpi)), (detinput_ + (d * detbpi)), (&object));             \
        if (object.valid == TRUE) {                                                         \
          g_array_append_val (results, object);                                             \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
    break

/** @brief Macro to simplify calling _get_objects_mobilenet_ssd */
#define _get_objects_mobilenet_ssd_(type, typename)                                       \
  _get_objects_mobilenet_ssd (trData, type, typename, (trData->mobilenet_ssd.box_priors), \
      (boxes->data), (detections->data), config, results)

/**
 * @brief Compare Function for g_array_sort with detectedObject.
 */
static gint
compare_detection (gconstpointer _a, gconstpointer _b)
{
  const detected_object *a = _a;
  const detected_object *b = _b;

  /**Larger comes first */
  return (a->score > b->score) ? -1 : ((a->score == b->score) ? 0 : 1);
}

/**
 * @brief Calculate the intersected surface
 */
static gfloat
iou (detected_object *a, detected_object *b)
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
 * @brief Apply NMS to the given results (objects[])
 * @param[in/out] results The results to be filtered with nms
 */
static void
nms (GArray *results, gfloat threshold)
{
  guint boxes_size;
  guint i, j;

  boxes_size = results->len;
  if (boxes_size == 0U)
    return;

  g_array_sort (results, compare_detection);

  for (i = 0; i < boxes_size; i++) {
    detected_object *a = &g_array_index (results, detected_object, i);
    if (a->valid == TRUE) {
      for (j = i + 1; j < boxes_size; j++) {
        detected_object *b = &g_array_index (results, detected_object, j);
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
    detected_object *a = &g_array_index (results, detected_object, i);
    if (a->valid == FALSE)
      g_array_remove_index (results, i);
    else
      i++;
  } while (i < results->len);
}

/**
 * @brief Private function to initialize the meta info
 */
static void
init_meta(GstTensorMetaInfo * meta, const tensor_region * trData){
  gst_tensor_meta_info_init (meta);
  meta->type = _NNS_UINT32;
  meta->dimension[0] = BOX_SIZE;
  meta->dimension[1] = trData->num;
  meta->media_type = _NNS_TENSOR;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
tr_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  tensor_region *trData = *pdata;
  GstTensorMetaInfo meta;
  GstMapInfo out_info;
  GstMemory *out_mem, *tmp_mem;
  GArray *results = NULL;
  const guint num_tensors = config->info.num_tensors;
  gboolean need_output_alloc = gst_buffer_get_size (outbuf) == 0;
  const size_t size = (size_t) 4 * trData->num * sizeof(uint32_t); /**4 field per block */

  g_assert (outbuf);
  /** Ensure we have outbuf properly allocated */
  if (need_output_alloc) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    ml_loge ("Cannot map output memory / tensordec-tensor_region.\n");
    goto error_free;
  }

  /** reset the buffer with 0 */
  memset (out_info.data, 0, size);
  if(trData->mode == MOBILENET_SSD_BOUNDING_BOX){
    const GstTensorMemory *boxes, *detections = NULL;
    properties_MOBILENET_SSD *data = &trData->mobilenet_ssd;

    g_assert (num_tensors >= MOBILENET_SSD_MAX_TENSORS);
    results = g_array_sized_new (FALSE, TRUE, sizeof (detected_object), 100);

    boxes = &input[0];
    if (num_tensors >= MOBILENET_SSD_MAX_TENSORS) /**lgtm[cpp/constant-comparison] */
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
  } else {
    GST_ERROR ("Failed to get output buffer, unknown mode %d.", trData->mode);
    goto error_unmap;
  }
  gst_tensor_top_detectedObjects_cropInfo (&out_info, trData, results);

  g_array_free (results, TRUE);

  gst_memory_unmap (out_mem, &out_info);

  /** converting to Flexible tensor since
   * info pad of tensor_crop has capability for flexible tensor stream
   */
  init_meta (&meta, trData);
  tmp_mem = out_mem;
  out_mem = gst_tensor_meta_info_append_header (&meta, tmp_mem);
  gst_memory_unref (tmp_mem);

  if (need_output_alloc) {
    gst_buffer_append_memory (outbuf, out_mem);
  } else {
    gst_buffer_replace_all_memory (outbuf, out_mem);
    gst_memory_unref (out_mem);
  }

  return GST_FLOW_OK;
error_unmap:
  gst_memory_unmap (out_mem, &out_info);
error_free:
  gst_memory_unref (out_mem);

  return GST_FLOW_ERROR;
}

/**
 * @brief set the max_detection
 */
static int
_set_max_detection (tensor_region *data, guint max_detection, unsigned int limit)
{
  /**Check consistency with max_detection */
  if (data->max_detections == 0)
    data->max_detections = max_detection;
  else
    g_return_val_if_fail (max_detection == data->max_detections, FALSE);

  if (data->max_detections > limit) {
    GST_ERROR ("Incoming tensor has too large detection-max : %u", max_detection);
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
 * If there are third or more tensors, such tensors will be ignored.
 */

static GstCaps *
tr_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  tensor_region *data = *pdata;
  GstCaps *caps;
  char *str;
  guint max_detection, max_label;
  const uint32_t *dim1, *dim2;
  int i;

  /**Check if the first tensor is compatible */
  dim1 = config->info.info[0].dimension;
  g_return_val_if_fail (dim1[0] == BOX_SIZE, NULL);
  g_return_val_if_fail (dim1[1] == 1, NULL);
  max_detection = dim1[2];
  g_return_val_if_fail (max_detection > 0, NULL);
  /** @todo unused dimension value should be 0 */
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim1[i] == 0 || dim1[i] == 1, NULL);

  /**Check if the second tensor is compatible */
  dim2 = config->info.info[1].dimension;
  max_label = dim2[0];
  g_return_val_if_fail (max_label <= data->labeldata.total_labels, NULL);
  if (max_label < data->labeldata.total_labels)
    GST_WARNING ("The given tensor (2nd) has max_label (first dimension: %u) smaller than the number of labels in labels file (%s: %u).",
        max_label, data->label_path, data->labeldata.total_labels);
  g_return_val_if_fail (max_detection == dim2[1], NULL);
  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim2[i] == 0 || dim2[i] == 1, NULL);

  /**Check consistency with max_detection */
  if (!_set_max_detection (data, max_detection, MOBILENET_SSD_DETECTION_MAX)) {
    return NULL;
  }
  str = g_strdup_printf("other/tensors,format=flexible");
  caps = gst_caps_from_string (str);
  setFramerateFromConfig (caps, config);
  g_free (str);
  (void) *pdata;
  return caps;
}


static gchar decoder_subplugin_tensor_region[] = "tensor_region";

/** @brief Tensor Region tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef tensorRegion = { .modename = decoder_subplugin_tensor_region,
  .init = tr_init,
  .exit = tr_exit,
  .setOption = tr_setOption,
  .getOutCaps = tr_getOutCaps,
  /** .getTransformSize = tr_getTransformsize, */
  .decode = tr_decode };

/** @brief Initialize this object for tensordec-plugin */
void
init_tr (void)
{
  nnstreamer_decoder_probe (&tensorRegion);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_tr (void)
{
  nnstreamer_decoder_exit (tensorRegion.modename);
}
