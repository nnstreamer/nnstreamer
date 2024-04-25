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
 *          for yolov5 and yolov8 mode:
 *            The option3 requires up to 3 numbers, which tell
 *              - whether the output values are scaled or not
 *                0: not scaled (default), 1: scaled (e.g., 0.0 ~ 1.0)
 *              - the threshold of confidence (optional, default set to 0.25)
 *              - the threshold of IOU (optional, default set to 0.45)
 *            An example of option3 is "option3=0:0.65:0.6"
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
 * option6: Whether to track result bounding boxes or not
 *          0 (default, do not track)
 *          1 (track result bounding boxes, with naive centroid based algorithm)
 * option7: Whether to log the result bounding boxes or not
 *          0 (default, do not log)
 *          1 (log result bounding boxes)
 * option8: Box Style (NYI)
 *
 * MAJOR TODO: Support other colorspaces natively from _decode for performance gain
 * (e.g., BGRA, ARGB, ...)
 *
 */

/** @todo _GNU_SOURCE fix build warning expf (nested-externs). remove this later. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_util.h>
#include <stdint.h>
#include <stdio.h>
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

/* font.c */
extern uint8_t rasters[][13];

/**
 * @todo Fill in the value at build time or hardcode this. It's const value
 * @brief The bitmap of characters
 * [Character (ASCII)][Height][Width]
 */
static singleLineSprite_t singleLineSprite;

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

/**
 * @brief check the num_tensors is valid
 */
static int
_check_tensors (const GstTensorsConfig *config, const unsigned int limit)
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
    g_return_val_if_fail (config->info.info[i - 1].type == config->info.info[i].type, FALSE);
  }
  return TRUE;
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
  bdata = nullptr;
}

BoundingBox::~BoundingBox ()
{
  _free_labels (&labeldata);

  if (label_path)
    g_free (label_path);

  if (bdata) {
    delete bdata;
    bdata = nullptr;
  }
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
      ml_logw ("Invalid class found with tensordec-boundingbox.c.\n");
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
      guint j;
      gsize label_len;

      if (is_track != 0) {
        label = g_strdup_printf ("%s-%d", labeldata.labels[a->class_id], a->tracking_id);
      } else {
        label = g_strdup_printf ("%s", labeldata.labels[a->class_id]);
      }

      label_len = strlen (label);
      /* x1 is the same: x1 = MAX (0, (width * a->x) / i_width); */
      y1 = MAX (0, (y1 - 14));
      pos1 = &frame[y1 * width + x1];
      for (j = 0; j < label_len; j++) {
        unsigned int char_index = label[j];
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
        pos1 += 9; /* charater width + 1px */
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
 * @brief check the label relevant properties are valid
 */
gboolean
BoundingBox::checkLabelProps ()
{
  if ((!label_path) || (!labeldata.labels) || (labeldata.total_labels <= 0))
    return FALSE;
  return TRUE;
}

int
BoundingBox::setBoxDecodingMode (const char *param)
{
  bounding_box_modes previous = mode;

  if (NULL == param || *param == '\0') {
    GST_ERROR ("Please set the valid mode at option1 to set box decoding mode");
    return FALSE;
  }

  mode = static_cast<bounding_box_modes> (find_key_strv (bb_modes, param));

  if (mode != previous && mode != BOUNDING_BOX_UNKNOWN) {
    if (previous != BOUNDING_BOX_UNKNOWN) {
      delete bdata;
    }

    switch (mode) {
      case MOBILENET_SSD_BOUNDING_BOX:
      case OLDNAME_MOBILENET_SSD_BOUNDING_BOX:
        bdata = new MobilenetSSD ();
        break;
      case MOBILENET_SSD_PP_BOUNDING_BOX:
      case OLDNAME_MOBILENET_SSD_PP_BOUNDING_BOX:
        bdata = new MobilenetSSDPP ();
        break;
      case OV_PERSON_DETECTION_BOUNDING_BOX:
      case OV_FACE_DETECTION_BOUNDING_BOX:
        bdata = new OVDetection ();
        break;
      case YOLOV5_BOUNDING_BOX:
        bdata = new YoloV5 ();
        break;
      case YOLOV8_BOUNDING_BOX:
        bdata = new YoloV8 ();
        break;
      case MP_PALM_DETECTION_BOUNDING_BOX:
        bdata = new MpPalmDetection ();
        break;
      default:
        return FALSE;
    }
  }

  return TRUE;
}

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
    gst_memory_unref (out_mem);

  return GST_FLOW_OK;

error_unmap:
  gst_memory_unmap (out_mem, &out_info);
error_free:
  gst_memory_unref (out_mem);

  return GST_FLOW_ERROR;
}

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

int
MobilenetSSD::checkCompatible (const GstTensorsConfig *config)
{
  const uint32_t *dim1, *dim2;
  int i;
  guint max_detection, max_label;

  if (!_check_tensors (config, MAX_TENSORS))
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

MobilenetSSDPP::MobilenetSSDPP ()
{
  tensor_mapping[LOCATIONS_IDX] = LOCATIONS_DEFAULT;
  tensor_mapping[CLASSES_IDX] = CLASSES_DEFAULT;
  tensor_mapping[SCORES_IDX] = SCORES_DEFAULT;
  tensor_mapping[NUM_IDX] = NUM_DEFAULT;
  threshold = THRESHOLD_DEFAULT;
}

/** @brief Helper to retrieve tensor index by feature */
int
MobilenetSSDPP::get_mobilenet_ssd_pp_tensor_idx (int idx)
{
  return tensor_mapping[idx];
}

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

int
MobilenetSSDPP::checkCompatible (const GstTensorsConfig *config)
{
  const uint32_t *dim1, *dim2, *dim3, *dim4;
  int locations_idx, classes_idx, scores_idx, num_idx, i;

  if (!_check_tensors (config, MAX_TENSORS))
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

int
OVDetection::checkCompatible (const GstTensorsConfig *config)
{
  const guint *dim;
  int i;
  UNUSED (total_labels);

  if (!_check_tensors (config, DEFAULT_MAX_TENSORS))
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

int
YoloV5::checkCompatible (const GstTensorsConfig *config)
{
  const guint *dim = config->info.info[0].dimension;
  int i;

  if (!_check_tensors (config, 1U))
    return FALSE;

  max_detection = ((i_width / 32) * (i_height / 32) + (i_width / 16) * (i_height / 16)
                      + (i_width / 8) * (i_height / 8))
                  * 3;

  g_return_val_if_fail (dim[0] == (total_labels + DEFAULT_DETECTION_NUM_INFO), FALSE);
  g_return_val_if_fail (dim[1] == max_detection, FALSE);
  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; ++i)
    g_return_val_if_fail (dim[i] == 0 || dim[i] == 1, FALSE);
  return TRUE;
}

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
  cStartIdx = DEFAULT_DETECTION_NUM_INFO;
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
      object.class_id = maxClassIdx - DEFAULT_DETECTION_NUM_INFO;
      object.tracking_id = 0;
      object.valid = TRUE;
      g_array_append_val (results, object);
    }
  }

  nms (results, iou_threshold);
  return results;
}


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

int
YoloV8::checkCompatible (const GstTensorsConfig *config)
{
  const guint *dim = config->info.info[0].dimension;
  int i;
  if (!_check_tensors (config, 1U))
    return FALSE;

  /** Only support for float type model */
  g_return_val_if_fail (config->info.info[0].type == _NNS_FLOAT32, FALSE);

  max_detection = (i_width / 32) * (i_height / 32) + (i_width / 16) * (i_height / 16)
                  + (i_width / 8) * (i_height / 8);

  if (dim[0] != (total_labels + DEFAULT_DETECTION_NUM_INFO) || dim[1] != max_detection) {
    nns_loge ("yolov8 boundingbox decoder requires the input shape to be %d:%d:1. But given shape is %d:%d:1. `tensor_transform mode=transpose` would be helpful.",
        total_labels + DEFAULT_DETECTION_NUM_INFO, max_detection, dim[0], dim[1]);
    return FALSE;
  }

  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; ++i)
    g_return_val_if_fail (dim[i] == 0 || dim[i] == 1, FALSE);
  return TRUE;
}

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
  cStartIdx = DEFAULT_DETECTION_NUM_INFO;
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
      object.class_id = maxClassIdx - DEFAULT_DETECTION_NUM_INFO;
      object.tracking_id = 0;
      object.valid = TRUE;
      g_array_append_val (results, object);
    }
  }

  nms (results, iou_threshold);
  return results;
}

#define mp_palm_detection_option(option, type, idx) \
  if (noptions > idx)                               \
  option = (type) g_strtod (options[idx], NULL)

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

MpPalmDetection::~MpPalmDetection ()
{
  if (anchors)
    g_array_free (anchors, TRUE);
  anchors = NULL;
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

int
MpPalmDetection::checkCompatible (const GstTensorsConfig *config)
{
  const uint32_t *dim1, *dim2;
  int i;
  if (!_check_tensors (config, MAX_TENSORS))
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
