/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2018 Jinhyuck Park <jinhyuck83.park@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file   tensordec_bounding_box_core.cc
 * @author Jinhyuck Park <jinhyuck83.park@samsung.com>
 * @date   11/06/2018
 * @brief  c++ apis for bounding boxes of tensor decoder.
 *
 * @bug     No known bugs.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <unistd.h>
#include <glib.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include "tensordec.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Macro for debug message.
 */
#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

#define _expit(x) \
    (1.f / (1.f + expf (-x)))

/**
 * @brief Data structure for bounding_boxes_core.
 */
typedef struct
{
  gboolean running; /**< true when app is running */
  GMutex mutex; /**< mutex for processing */
  Mode_boundig_boxes boxes_info; /**< tflite model info */
    std::vector < DetectedObject > detected_objects;
} core_Data;

/**
 * @brief Data for pipeline and result.
 */
static core_Data core;

/**
 * @brief Compare score of detected objects.
 */
static bool
gst_tensordec_compare_objs (DetectedObject & a, DetectedObject & b)
{
  return a.prob > b.prob;
}

/**
 * @brief Read strings from file.
 */
extern gboolean
gst_tensordec_read_lines (const gchar * file_name, GList ** lines)
{
  std::ifstream file (file_name);
  if (!file) {
    _print_log ("Failed to open file %s", file_name);
    return FALSE;
  }

  std::string str;
  while (std::getline (file, str)) {
    *lines = g_list_append (*lines, g_strdup (str.c_str ()));
  }

  return TRUE;
}

/**
 * @brief Intersection of union
 */
static gfloat
gst_tensordec_iou (DetectedObject & A, DetectedObject & B)
{
  int x1 = std::max (A.x, B.x);
  int y1 = std::max (A.y, B.y);
  int x2 = std::min (A.x + A.width, B.x + B.width);
  int y2 = std::min (A.y + A.height, B.y + B.height);
  int w = std::max (0, (x2 - x1 + 1));
  int h = std::max (0, (y2 - y1 + 1));
  float inter = w * h;
  float areaA = A.width * A.height;
  float areaB = B.width * B.height;
  float o = inter / (areaA + areaB - inter);
  return (o >= 0) ? o : 0;
}

/**
 * @brief NMS (non-maximum suppression)
 */
static void
gst_tensordec_nms (std::vector < DetectedObject > &detected)
{
  const float threshold_iou = .5f;
  guint boxes_size;
  guint i, j;

  std::sort (detected.begin (), detected.end (), gst_tensordec_compare_objs);
  boxes_size = detected.size ();

  std::vector < bool > del (boxes_size, false);
  for (i = 0; i < boxes_size; i++) {
    if (!del[i]) {
      for (j = i + 1; j < boxes_size; j++) {
        if (gst_tensordec_iou (detected.at (i),
                detected.at (j)) > threshold_iou) {
          del[j] = true;
        }
      }
    }
  }

  /* update result */
  g_mutex_lock (&core.mutex);

  core.detected_objects.clear ();
  for (i = 0; i < boxes_size; i++) {
    if (!del[i]) {
      core.detected_objects.push_back (detected[i]);

      if (DBG) {
        _print_log ("==============================");
        _print_log ("Label           : %s",
            (gchar *) g_list_nth_data (core.boxes_info.labels,
                detected[i].class_id));
        _print_log ("x               : %d", detected[i].x);
        _print_log ("y               : %d", detected[i].y);
        _print_log ("width           : %d", detected[i].width);
        _print_log ("height          : %d", detected[i].height);
        _print_log ("Confidence Score: %f", detected[i].prob);
      }
    }
  }

  g_mutex_unlock (&core.mutex);
}

/**
 * @brief Get detected objects.
 */
extern void
gst_tensordec_get_detected_objects (gfloat * detections, gfloat * boxes)
{
  const float threshold_score = .5f;
  std::vector < DetectedObject > detected;

  for (int d = 0; d < DETECTION_MAX; d++) {
    float ycenter =
        boxes[0] / Y_SCALE * core.boxes_info.box_priors[2][d] +
        core.boxes_info.box_priors[0][d];
    float xcenter =
        boxes[1] / X_SCALE * core.boxes_info.box_priors[3][d] +
        core.boxes_info.box_priors[1][d];
    float h =
        (float) expf (boxes[2] / H_SCALE) * core.boxes_info.box_priors[2][d];
    float w =
        (float) expf (boxes[3] / W_SCALE) * core.boxes_info.box_priors[3][d];

    float ymin = ycenter - h / 2.f;
    float xmin = xcenter - w / 2.f;
    float ymax = ycenter + h / 2.f;
    float xmax = xcenter + w / 2.f;

    int x = xmin * MODEL_WIDTH;
    int y = ymin * MODEL_HEIGHT;
    int width = (xmax - xmin) * MODEL_WIDTH;
    int height = (ymax - ymin) * MODEL_HEIGHT;

    for (int c = 1; c < LABEL_SIZE; c++) {
      gfloat score = _expit (detections[c]);
      /**
       * This score cutoff is taken from Tensorflow's demo app.
       * There are quite a lot of nodes to be run to convert it to the useful possibility
       * scores. As a result of that, this cutoff will cause it to lose good detections in
       * some scenarios and generate too much noise in other scenario.
       */
      if (score < threshold_score)
        continue;

      DetectedObject object;

      object.class_id = c;
      object.x = x;
      object.y = y;
      object.width = width;
      object.height = height;
      object.prob = score;

      detected.push_back (object);
    }

    detections += LABEL_SIZE;
    boxes += BOX_SIZE;
  }

  gst_tensordec_nms (detected);
}
