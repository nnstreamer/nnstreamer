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
 * @file        tensordec-boundingbox.cc
 * @date        15 Nov 2018
 * @brief       NNStreamer tensor-decoder subplugin, "bounding boxes",
 *              which converts tensors to video stream w/ boxes on
 *              transparent background.
 *              This code is NYI/WIP and not compilable.
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs except for NYI items
 */

/** @todo _GNU_SOURCE fix build warning expf (nested-externs). remove this later. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <glib.h>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "tensordec-boundingbox.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_bb (void) __attribute__ ((constructor));
void fini_bb (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief mutex for box properties table.
 */
G_LOCK_DEFINE_STATIC (box_properties_table);

/* font.c */
extern uint8_t rasters[][13];

/**
 * @todo Fill in the value at build time or hardcode this. It's const value
 * @brief The bitmap of characters
 * [Character (ASCII)][Height][Width]
 */
static singleLineSprite_t singleLineSprite;

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
  [YOLOV8_BOUNDING_BOX] = "yolov8",
  NULL,
};

/**
 * @brief Change deprecated mode name
 */
static const char *
updateDecodingMode (const char *param)
{
  if (g_strcmp0 (param, bb_modes[OLDNAME_MOBILENET_SSD_BOUNDING_BOX]) == 0) {
    return bb_modes[MOBILENET_SSD_BOUNDING_BOX];
  }

  if (g_strcmp0 (param, bb_modes[OLDNAME_MOBILENET_SSD_PP_BOUNDING_BOX]) == 0) {
    return bb_modes[MOBILENET_SSD_PP_BOUNDING_BOX];
  }

  return param;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
bb_init (void **pdata)
{
  /** @todo check if we need to ensure plugin_data is not yet allocated */
  BoundingBox *bdata = new BoundingBox ();
  *pdata = bdata;

  if (bdata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  initSingleLineSprite (singleLineSprite, rasters, PIXEL_VALUE);

  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
bb_exit (void **pdata)
{
  BoundingBox *bdata = static_cast<BoundingBox *> (*pdata);
  delete bdata;
  *pdata = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
bb_setOption (void **pdata, int opNum, const char *param)
{
  BoundingBox *bdata = static_cast<BoundingBox *> (*pdata);
  BoundingBoxOption option = static_cast<BoundingBoxOption> (opNum);
  return bdata->setOption (option, param);
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
bb_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  /** @todo this is compatible with "SSD" only. expand the capability! */
  BoundingBox *bdata = static_cast<BoundingBox *> (*pdata);
  return bdata->getOutCaps (config);
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
bb_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  BoundingBox *bdata = static_cast<BoundingBox *> (*pdata);
  return bdata->decode (config, input, outbuf);
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
bb_getTransformSize (void **pdata, const GstTensorsConfig *config,
    GstCaps *caps, size_t size, GstCaps *othercaps, GstPadDirection direction)
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

static gchar decoder_subplugin_bounding_box[] = "bounding_boxes";

/** @brief Bounding box tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef boundingBox = { .modename = decoder_subplugin_bounding_box,
  .init = bb_init,
  .exit = bb_exit,
  .setOption = bb_setOption,
  .getOutCaps = bb_getOutCaps,
  .decode = bb_decode,
  .getTransformSize = bb_getTransformSize };

static gchar *custom_prop_desc = NULL;

/** @brief Initialize this object for tensordec-plugin */
void
init_bb (void)
{
  nnstreamer_decoder_probe (&boundingBox);

  {
    g_autofree gchar *sub_desc = g_strjoinv ("|", (GStrv) bb_modes);

    g_free (custom_prop_desc);
    custom_prop_desc = g_strdup_printf ("Decoder mode of bounding box: [%s]", sub_desc);

    nnstreamer_decoder_set_custom_property_desc (decoder_subplugin_bounding_box,
        "option1", custom_prop_desc, "option2",
        "Location of the label file. This is independent from option1.", "option3",
        "Sub-option values that depend on option1;\n"
        "\tfor yolov5 and yolov8 mode:\n"
        "\t\tThe option3 requires up to 3 numbers, which tell\n"
        "\t\t- whether the output values are scaled or not\n"
        "\t\t   0: not scaled (default), 1: scaled (e.g., 0.0 ~ 1.0)\n"
        "\t\t- the threshold of confidence (optional, default set to 0.25)\n"
        "\t\t- the threshold of IOU (optional, default set to 0.45)\n"
        "\t\tAn example of option3 is option3 = 0: 0.65:0.6 \n"
        "\tfor mobilenet-ssd mode:\n"
        "\t\tThe option3 definition scheme is, in order, as follows\n"
        "\t\t- box priors location file (mandatory)\n"
        "\t\t- detection threshold (optional, default set to 0.5)box priors location file (mandatory)\n"
        "\t\t- Y box scale (optional, default set to 10.0)\n"
        "\t\t- X box scale (optional, default set to 10.0)\n"
        "\t\t- H box scale (optional, default set to 5.0)\n"
        "\t\t- W box scale (optional, default set to 5.0)\n"
        "\t\tThe default parameters value could be set in the following ways:\n"
        "\t\t option3=box-priors.txt:0.5:10.0:10.0:5.0:5.0:0.5\n"
        "\t\t option3=box-priors.txt\n"
        "\t\t option3=box-priors.txt::::::\n"
        "\t\tIt's possible to set only few values, using the default values for those not specified through the command line.\n"
        "\t\tYou could specify respectively the detection and IOU thresholds to 0.65 and 0.6 with the option3 parameter as follow:\n"
        "\t\t option3=box-priors.txt:0.65:::::0.6\n"
        "\tfor mobilenet-ssd-postprocess mode:\n"
        "\t\tThe option3 is required to have 5 integer numbers, which tell the tensor-dec how to interpret the given tensor inputs.\n"
        "\t\tThe first 4 numbers separated by colon, \':\', designate which are location:class:score:number of the tensors.\n"
        "\t\tThe last number separated by comma, ',\' from the first 4 numbers designate the threshold in percent.\n"
        "\t\tIn other words, \"option3=%i:%i:%i:%i,%i\"\n"
        "\tfor mp-palm-detection mode:\n"
        "\t\tThe option3 is required to have five float numbers, as follows;\n"
        "\t\t- box score threshold (mandatory)\n"
        "\t\t- number of layers for anchor generation (optional, default set to 4)\n"
        "\t\t- minimum scale factor for anchor generation (optional, default set to 1.0)\n"
        "\t\t- maximum scale factor for anchor generation (optional, default set to 1.0)\n"
        "\t\t- X offset (optional, default set to 0.5)\n"
        "\t\t- Y offset (optional, default set to 0.5)\n"
        "\t\t- strides for each layer for anchor generation (optional, default set to 8:16:16:16)\n"
        "\t\tThe default parameter value could be set in the following ways:\n"
        "\t\t option3=0.5\n"
        "\t\t option3=0.5:4:0.2:0.8\n"
        "\t\t option3=0.5:4:1.0:1.0:0.5:0.5:8:16:16:16",
        "option4", "Video Output Dimension (WIDTH:HEIGHT). This is independent from option1.",
        "option5", "Input Dimension (WIDTH:HEIGHT). This is independent from option1.", "option6",
        "Whether to track result bounding boxes or not\n"
        "\t\t 0 (default, do not track)\n"
        "\t\t 1 (track result bounding boxes, with naive centroid based algorithm)",
        "option7",
        "Whether to log the result bounding boxes or not\n"
        "\t\t 0 (default, do not log)\n"
        "\t\t 1 (log result bounding boxes)"
        "\tThis is independent from option1",
        "option8", "Box Style (NYI)", NULL);
  }
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_bb (void)
{
  g_free (custom_prop_desc);
  custom_prop_desc = NULL;
  nnstreamer_decoder_exit (boundingBox.modename);
}

/** @brief Compare function for sorting distances. */
static int
distance_compare (const void *a, const void *b)
{
  const distanceArrayData *da = (const distanceArrayData *) a;
  const distanceArrayData *db = (const distanceArrayData *) b;

  if (da->distance < db->distance)
    return -1;
  if (da->distance > db->distance)
    return 1;
  return 0;
}

/**
 * @brief Compare Function for g_array_sort with detectedObject.
 */
static gint
compare_detection (gconstpointer _a, gconstpointer _b)
{
  const detectedObject *a = static_cast<const detectedObject *> (_a);
  const detectedObject *b = static_cast<const detectedObject *> (_b);

  /* Larger comes first */
  return (a->prob > b->prob) ? -1 : ((a->prob == b->prob) ? 0 : 1);
}

/**
 * @brief Calculate the intersected surface
 */
static gfloat
iou (detectedObject *a, detectedObject *b)
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
 * @brief Apply NMS to the given results (objects[DETECTION_MAX])
 */
void
nms (GArray *results, gfloat threshold, bounding_box_modes mode)
{
  guint boxes_size;
  guint i, j;

  boxes_size = results->len;
  if (boxes_size == 0U)
    return;

  g_array_sort (results, compare_detection);

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
 * @brief check the num_tensors is valid
 */
int
check_tensors (const GstTensorsConfig *config, const unsigned int limit)
{
  unsigned int i;
  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (config->info.num_tensors >= limit, FALSE);
  if (config->info.num_tensors > limit) {
    GST_WARNING ("tensor-decoder:boundingbox accepts %d or less tensors. "
                 "You are wasting the bandwidth by supplying %d tensors.",
        limit, config->info.num_tensors);
  }

  /* tensor-type of the tensors should be the same */
  for (i = 1; i < config->info.num_tensors; ++i) {
    g_return_val_if_fail (
        gst_tensors_info_get_nth_info ((GstTensorsInfo *) &config->info, i - 1)->type
            == gst_tensors_info_get_nth_info ((GstTensorsInfo *) &config->info, i)
                   ->type,
        FALSE);
  }
  return TRUE;
}

/** @brief Constructor of BoundingBox */
BoundingBox::BoundingBox ()
{
  mode = BOUNDING_BOX_UNKNOWN;
  width = 0;
  height = 0;
  flag_use_label = FALSE;
  do_log = 0;

  /* for track */
  is_track = 0;
  centroids_last_id = 0U;
  max_centroids_num = 100U;
  consecutive_disappear_threshold = 100U;
  centroids = g_array_sized_new (TRUE, TRUE, sizeof (centroid), max_centroids_num);
  distanceArray = g_array_sized_new (TRUE, TRUE, sizeof (distanceArrayData),
      max_centroids_num * max_centroids_num);

  label_path = nullptr;
  labeldata.labels = nullptr;
  labeldata.max_word_length = 0;
  labeldata.total_labels = 0;
  bdata = nullptr;
}

/** @brief destructor of BoundingBox */
BoundingBox::~BoundingBox ()
{
  _free_labels (&labeldata);

  if (label_path)
    g_free (label_path);

  g_array_free (centroids, TRUE);
  g_array_free (distanceArray, TRUE);

  G_LOCK (box_properties_table);
  g_hash_table_destroy (properties_table);
  properties_table = nullptr;
  G_UNLOCK (box_properties_table);
}

/**
 * @brief Update centroids with given bounding boxes.
 */
void
BoundingBox::updateCentroids (GArray *boxes)
{
  guint i, j;
  if (boxes->len > max_centroids_num) {
    nns_logw ("updateCentroids: too many detected objects");
    return;
  }
  /* remove disappeared centroids */
  i = 0;
  while (i < centroids->len) {
    centroid *c = &g_array_index (centroids, centroid, i);
    if (c->consecutive_disappeared_frames >= consecutive_disappear_threshold) {
      g_array_remove_index (centroids, i);
    } else {
      i++;
    }
  }

  if (centroids->len > max_centroids_num) {
    nns_logw ("update_centroids: too many detected centroids");
    return;
  }
  /* if boxes is empty */
  if (boxes->len == 0U) {
    guint i;
    for (i = 0; i < centroids->len; i++) {
      centroid *c = &g_array_index (centroids, centroid, i);

      if (c->id > 0)
        c->consecutive_disappeared_frames++;
    }

    return;
  }
  /* initialize centroids with given boxes */
  if (centroids->len == 0U) {
    guint i;
    for (i = 0; i < boxes->len; i++) {
      detectedObject *box = &g_array_index (boxes, detectedObject, i);
      centroid c;

      centroids_last_id++;
      c.id = centroids_last_id;
      c.consecutive_disappeared_frames = 0;
      c.cx = box->x + box->width / 2;
      c.cy = box->y + box->height / 2;
      c.matched_box_idx = i;

      g_array_append_val (centroids, c);

      box->tracking_id = c.id;
    }

    return;
  }
  /* calculate the distance among centroids and boxes */
  g_array_set_size (distanceArray, centroids->len * boxes->len);

  for (i = 0; i < centroids->len; i++) {
    centroid *c = &g_array_index (centroids, centroid, i);
    c->matched_box_idx = G_MAXUINT32;

    for (j = 0; j < boxes->len; j++) {
      detectedObject *box = &g_array_index (boxes, detectedObject, j);
      distanceArrayData *d
          = &g_array_index (distanceArray, distanceArrayData, i * centroids->len + j);

      d->centroid_idx = i;
      d->box_idx = j;

      /* invalid centroid */
      if (c->id == 0) {
        d->distance = G_MAXUINT64;
      } else {
        /* calculate euclidean distance */
        int bcx = box->x + box->width / 2;
        int bcy = box->y + box->height / 2;

        d->distance = (guint64) (c->cx - bcx) * (c->cx - bcx)
                      + (guint64) (c->cy - bcy) * (c->cy - bcy);
      }
    }
  }

  g_array_sort (distanceArray, distance_compare);

  {
    /* Starting from the least distance pair (centroid, box), matching each other */
    guint dIdx, cIdx, bIdx;

    for (dIdx = 0; dIdx < distanceArray->len; dIdx++) {
      distanceArrayData *d = &g_array_index (distanceArray, distanceArrayData, dIdx);
      centroid *c = &g_array_index (centroids, centroid, d->centroid_idx);
      detectedObject *box = &g_array_index (boxes, detectedObject, d->box_idx);

      bIdx = d->box_idx;

      /* the centroid is invalid */
      if (c->id == 0) {
        continue;
      }
      /* the box is already assigned to a centroid */
      if (box->tracking_id != 0) {
        continue;
      }
      /* the centroid is already assigned to a box */
      if (c->matched_box_idx != G_MAXUINT32) {
        continue;
      }
      /* now match the box with the centroid */
      c->matched_box_idx = bIdx;
      box->tracking_id = c->id;
      c->consecutive_disappeared_frames = 0;
    }

    /* increase consecutive_disappeared_frames of unmatched centroids */
    for (cIdx = 0; cIdx < centroids->len; cIdx++) {
      centroid *c = &g_array_index (centroids, centroid, cIdx);

      if (c->id == 0) {
        continue;
      }

      if (c->matched_box_idx == G_MAXUINT32) {
        c->consecutive_disappeared_frames++;
      }
    }

    /* for those unmatched boxes - register as new centroids */
    for (bIdx = 0; bIdx < boxes->len; bIdx++) {
      detectedObject *box = &g_array_index (boxes, detectedObject, bIdx);
      centroid c;

      if (box->tracking_id != 0) {
        continue;
      }

      centroids_last_id++;
      c.id = centroids_last_id;
      c.consecutive_disappeared_frames = 0;
      c.cx = box->x + box->width / 2;
      c.cy = box->y + box->height / 2;
      c.matched_box_idx = bIdx;

      g_array_append_val (centroids, c);

      box->tracking_id = c.id;
    }
  }
}

/**
 * @brief Draw with the given results (objects[DETECTION_MAX]) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bounding-box internal data.
 * @param[in] results The final results to be drawn.
 */
void
BoundingBox::draw (GstMapInfo *out_info, GArray *results)
{
  uint32_t *frame = (uint32_t *) out_info->data; /* Let's draw per pixel (4bytes) */
  unsigned int i;
  guint i_width, i_height;

  i_width = bdata->getInputWidth ();
  i_height = bdata->getInputHeight ();

  for (i = 0; i < results->len; i++) {
    int x1, x2, y1, y2; /* Box positions on the output surface */
    int j;
    uint32_t *pos1, *pos2;
    detectedObject *a = &g_array_index (results, detectedObject, i);

    if ((flag_use_label)
        && ((a->class_id < 0 || a->class_id >= (int) labeldata.total_labels))) {
      /** @todo make it "logw_once" after we get logw_once API. */
      ml_logw ("Invalid class found with tensordec-boundingbox.\n");
      continue;
    }

    /* 1. Draw Boxes */
    x1 = (width * a->x) / i_width;
    x2 = MIN (width - 1, (width * (a->x + a->width)) / i_width);
    y1 = (height * a->y) / i_height;
    y2 = MIN (height - 1, (height * (a->y + a->height)) / i_height);

    /* 1-1. Horizontal */
    pos1 = &frame[y1 * width + x1];
    pos2 = &frame[y2 * width + x1];
    for (j = x1; j <= x2; j++) {
      *pos1 = PIXEL_VALUE;
      *pos2 = PIXEL_VALUE;
      pos1++;
      pos2++;
    }

    /* 1-2. Vertical */
    pos1 = &frame[(y1 + 1) * width + x1];
    pos2 = &frame[(y1 + 1) * width + x2];
    for (j = y1 + 1; j < y2; j++) {
      *pos1 = PIXEL_VALUE;
      *pos2 = PIXEL_VALUE;
      pos1 += width;
      pos2 += width;
    }

    /* 2. Write Labels + tracking ID */
    if (flag_use_label) {
      g_autofree gchar *label = NULL;
      gsize k, label_len;

      if (is_track != 0) {
        label = g_strdup_printf ("%s-%d", labeldata.labels[a->class_id], a->tracking_id);
      } else {
        label = g_strdup_printf ("%s", labeldata.labels[a->class_id]);
      }

      label_len = label ? strlen (label) : 0;

      /* x1 is the same: x1 = MAX (0, (width * a->x) / i_width); */
      y1 = MAX (0, (y1 - 14));
      pos1 = &frame[y1 * width + x1];
      for (k = 0; k < label_len; k++) {
        unsigned int char_index = label[k];
        if ((x1 + 8) > (int) width)
          break; /* Stop drawing if it may overfill */
        pos2 = pos1;
        for (y2 = 0; y2 < 13; y2++) {
          /* 13 : character height */
          for (x2 = 0; x2 < 8; x2++) {
            /* 8: character width */
            *(pos2 + x2) = singleLineSprite[char_index][y2][x2];
          }
          pos2 += width;
        }
        x1 += 9;
        pos1 += 9; /* character width + 1px */
      }
    }
  }
}

/**
 * @brief Log the given results
 */
void
BoundingBox::logBoxes (GArray *results)
{
  guint i;

  nns_logi ("Detect %u boxes in %u x %u input image", results->len,
      bdata->getInputWidth (), bdata->getInputHeight ());
  for (i = 0; i < results->len; i++) {
    detectedObject *b = &g_array_index (results, detectedObject, i);
    if (labeldata.total_labels > 0)
      nns_logi ("[%s] x:%d y:%d w:%d h:%d prob:%.4f",
          labeldata.labels[b->class_id], b->x, b->y, b->width, b->height, b->prob);
    else
      nns_logi ("x:%d y:%d w:%d h:%d prob:%.4f", b->x, b->y, b->width, b->height, b->prob);
  }
}

/**
 * @brief Check the label relevant properties are valid
 */
gboolean
BoundingBox::checkLabelProps ()
{
  if ((!label_path) || (!labeldata.labels) || (labeldata.total_labels <= 0))
    return FALSE;
  return TRUE;
}

/**
 * @brief Set mode of bounding box
 */
int
BoundingBox::setBoxDecodingMode (const char *param)
{
  if (NULL == param || *param == '\0') {
    GST_ERROR ("Please set the valid mode at option1 to set box decoding mode");
    return FALSE;
  }

  bdata = getProperties (updateDecodingMode (param));
  if (bdata == nullptr) {
    nns_loge ("Could not find box properties name %s", param);
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Set label path of bounding box
 */
int
BoundingBox::setLabelPath (const char *param)
{
  if (mode == MP_PALM_DETECTION_BOUNDING_BOX) {
    /* palm detection does not need label information */
    return TRUE;
  }

  if (NULL != label_path)
    g_free (label_path);
  label_path = g_strdup (param);

  if (NULL != label_path)
    loadImageLabels (label_path, &labeldata);

  if (labeldata.total_labels > 0) {
    bdata->setTotalLabels (labeldata.total_labels);
    return TRUE;
  } else
    return FALSE;
  /** @todo Do not die for this */
}

/**
 * @brief Set video size of bounding box
 */
int
BoundingBox::setVideoSize (const char *param)
{
  tensor_dim dim;
  int rank;

  if (param == NULL || *param == '\0')
    return TRUE;
  rank = gst_tensor_parse_dimension (param, dim);
  width = 0;
  height = 0;

  if (rank < 2) {
    GST_ERROR ("mode-option-2 of boundingbox is video output dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
        param);
    return TRUE; /* Ignore this param */
  }
  if (rank > 2) {
    GST_WARNING ("mode-option-2 of boundingbox is video output dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
        param);
  }
  width = dim[0];
  height = dim[1];
  return TRUE;
}

/**
 * @brief Set input model size of bounding box
 */
int
BoundingBox::setInputModelSize (const char *param)
{
  tensor_dim dim;
  int rank;
  if (param == NULL || *param == '\0')
    return TRUE;

  rank = gst_tensor_parse_dimension (param, dim);
  bdata->setInputWidth (0);
  bdata->setInputHeight (0);

  if (rank < 2) {
    GST_ERROR ("mode-option-3 of boundingbox is input video dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
        param);
    return TRUE; /* Ignore this param */
  }
  if (rank > 2) {
    GST_WARNING ("mode-option-3 of boundingbox is input video dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
        param);
  }
  bdata->setInputWidth (dim[0]);
  bdata->setInputHeight (dim[1]);
  return TRUE;
}

/**
 * @brief Set option of bounding box
 */
int
BoundingBox::setOption (BoundingBoxOption option, const char *param)
{
  if (option == BoundingBoxOption::MODE) {
    return setBoxDecodingMode (param);
  } else if (option == BoundingBoxOption::LABEL_PATH) {
    return setLabelPath (param);
  } else if (option == BoundingBoxOption::INTERNAL) {
    /* option3 = per-decoding-mode option */
    return bdata->setOptionInternal (param);
  } else if (option == BoundingBoxOption::VIDEO_SIZE) {
    return setVideoSize (param);
  } else if (option == BoundingBoxOption::INPUT_MODEL_SIZE) {
    return setInputModelSize (param);
  } else if (option == BoundingBoxOption::TRACK) {
    is_track = (int) g_ascii_strtoll (param, NULL, 10);
    return TRUE;
  } else if (option == BoundingBoxOption::LOG) {
    do_log = (int) g_ascii_strtoll (param, NULL, 10);
    return TRUE;
  }

  /**
   * @todo Accept color / border-width / ... with option-2
   */
  GST_INFO ("Property mode-option-%d is ignored", static_cast<int> (option) + 1);
  return TRUE;
}

/**
 * @brief Get out caps of bounding box
 */
GstCaps *
BoundingBox::getOutCaps (const GstTensorsConfig *config)
{
  GstCaps *caps;
  char *str;

  int ret = bdata->checkCompatible (config);
  if (!ret)
    return NULL;

  str = g_strdup_printf ("video/x-raw, format = RGBA, " /* Use alpha channel to make the background transparent */
                         "width = %u, height = %u",
      width, height);
  caps = gst_caps_from_string (str);
  setFramerateFromConfig (caps, config);
  g_free (str);

  return caps;
}

/**
 * @brief Decode input memory to out buffer
 * @param[in] config The structure of input tensor info.
 * @param[in] input The array of input tensor data. The maximum array size of input data is NNS_TENSOR_SIZE_LIMIT.
 * @param[out] outbuf A sub-plugin should update or append proper memory for the negotiated media type.
 */
GstFlowReturn
BoundingBox::decode (const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  const size_t size = (size_t) width * height * 4; /* RGBA */
  GstMapInfo out_info;
  GstMemory *out_mem;
  GArray *results = NULL;
  gboolean need_output_alloc;

  g_assert (outbuf);
  need_output_alloc = gst_buffer_get_size (outbuf) == 0;

  if (checkLabelProps ())
    flag_use_label = TRUE;
  else
    flag_use_label = FALSE;

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

  /* reset the buffer with alpha 0 / black */
  memset (out_info.data, 0, size);

  results = bdata->decode (config, input);
  if (results == NULL) {
    GST_ERROR ("Failed to get output buffer, unknown mode %d.", mode);
    goto error_unmap;
  }

  if (do_log != 0) {
    logBoxes (results);
  }

  if (is_track != 0) {
    updateCentroids (results);
  }

  draw (&out_info, results);
  g_array_free (results, TRUE);

  gst_memory_unmap (out_mem, &out_info);

  if (need_output_alloc)
    gst_buffer_append_memory (outbuf, out_mem);
  else
    gst_buffer_replace_all_memory (outbuf, out_mem);

  return GST_FLOW_OK;

error_unmap:
  gst_memory_unmap (out_mem, &out_info);
error_free:
  gst_memory_unref (out_mem);

  return GST_FLOW_ERROR;
}

/**
 * @brief Get bounding box properties from hash table
 */
BoxProperties *
BoundingBox::getProperties (const gchar *properties_name)
{
  gpointer data;
  G_LOCK (box_properties_table);
  if (properties_table == nullptr) {
    properties_table = g_hash_table_new (g_str_hash, g_str_equal);
  }
  data = g_hash_table_lookup (properties_table, properties_name);
  G_UNLOCK (box_properties_table);

  return static_cast<BoxProperties *> (data);
}

/**
 * @brief Add bounding box properties into hash table
 */
gboolean
BoundingBox::addProperties (BoxProperties *boxProperties)
{
  BoxProperties *data;
  gboolean ret;

  data = getProperties (boxProperties->name);
  if (NULL != data) {
    return TRUE;
  }

  G_LOCK (box_properties_table);
  ret = g_hash_table_insert (properties_table, boxProperties->name, boxProperties);
  G_UNLOCK (box_properties_table);

  return ret;
}
