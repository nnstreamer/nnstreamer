/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "Pose estimation"
 * Copyright (C) 2019 Samsung Electronics Co. Ltd.
 * Copyright (C) 2019 Jijoong Moon <jijoong.moon@samsung.com>
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
 * @file        tensordec-pose.c
 * @date        13 May 2019
 * @brief       NNStreamer tensor-decoder subplugin, "pose estimation",
 *              which converts tensors to video stream w/ pose on
 *              transparent background.
 *              This code is NYI/WIP and not compilable.
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 * option1: Video Output Dimension (WIDTH:HEIGHT)
 * option2: Input Dimension (WIDTH:HEIGHT)
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api.h>
#include "tensordecutil.h"

void init_pose (void) __attribute__ ((constructor));
void finish_pose (void) __attribute__ ((destructor));

/* font.c */
extern uint8_t rasters[][13];

#define POSE_SIZE                  14
#define PIXEL_VALUE               (0xFFFFFFFF)

/**
 * @todo Fill in the value at build time or hardcode this. It's const value
 * @brief The bitmap of characters
 * [Character (ASCII)][Height][Width]
 */
static singleLineSprite_t singleLineSprite;

/**
 * @brief Data structure for boundig box info.
 */
typedef struct
{
  /* From option1 */
  guint width; /**< Output Video Width */
  guint height; /**< Output Video Height */

  /* From option2 */
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */

} pose_data;

/** @brief tensordec-plugin's TensorDecDef callback */
static int
pose_init (void **pdata)
{
  pose_data *data;

  data = *pdata = g_new0 (pose_data, 1);
  if (data == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  data->width = 0;
  data->height = 0;
  data->i_width = 0;
  data->i_height = 0;

  initSingleLineSprite (singleLineSprite, rasters, PIXEL_VALUE);

  return TRUE;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static void
pose_exit (void **pdata)
{
  g_free (*pdata);
  *pdata = NULL;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static int
pose_setOption (void **pdata, int opNum, const char *param)
{
  pose_data *data = *pdata;

  if (opNum == 0) {
    /* option1 = output video size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    data->width = 0;
    data->height = 0;
    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR
          ("mode-option-1 of pose estimation is video output dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE;              /* Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING
          ("mode-option-1 of pose estimation is video output dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    data->width = dim[0];
    data->height = dim[1];
    return TRUE;
  } else if (opNum == 1) {
    /* option1 = input model size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    data->i_width = 0;
    data->i_height = 0;
    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR
          ("mode-option-2 of pose estimation is input video dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE;
    }
    if (rank > 2) {
      GST_WARNING
          ("mode-option-2 of pose esitmiation is input video dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    data->i_width = dim[0];
    data->i_height = dim[1];
    return TRUE;
  }

  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

/**
 * @brief check the num_tensors is valid
*/
static int
_check_tensors (const GstTensorsConfig * config)
{
  int i;
  g_return_val_if_fail (config != NULL, FALSE);

  for (i = 1; i < config->info.num_tensors; ++i) {
    g_return_val_if_fail (config->info.info[i - 1].type ==
        config->info.info[i].type, FALSE);
  }
  return TRUE;
}

/**
 * @brief tensordec-plugin's TensorDecDef callback
 *
 * [Pose Estimation]
 * Just one tensor with [ 14 (#Joint), WIDTH, HEIGHT, 1]
 * One WIDTH:HEIGHT for the each joint.
 * Have to find max value after Gaussian Blur
 *
 */
static GstCaps *
pose_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
  pose_data *data = *pdata;
  GstCaps *caps;
  int i;
  char *str;

  const uint32_t *dim1;

  if (!_check_tensors (config))
    return NULL;

  /* Check if the first tensor is compatible */
  dim1 = config->info.info[0].dimension;
  g_return_val_if_fail (dim1[0] == POSE_SIZE, NULL);
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim1[i] == 1, NULL);

  str = g_strdup_printf ("video/x-raw, format = RGBA, " /* Use alpha channel to make the background transparent */
      "width = %u, height = %u"
      /** @todo Configure framerate! */
      , data->width, data->height);
  caps = gst_caps_from_string (str);
  g_free (str);

  return caps;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static size_t
pose_getTransformSize (void **pdata, const GstTensorsConfig * config,
    GstCaps * caps, size_t size, GstCaps * othercaps, GstPadDirection direction)
{
  return 0;
}

/** @brief Represents a pose */
typedef struct
{
  int valid;
  int x;
  int y;
  gfloat prob;
} pose;

/**
 * @brief Fill in pixel with PIXEL_VALUE at x,y position. Make thicker (x+1, y+1)
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bouding-box internal data.
 * @param[in] coordinate of pixel
 */
static void
setpixel (uint32_t * frame, pose_data * data, int x, int y)
{
  uint32_t *pos = &frame[y * data->width + x];
  *pos = PIXEL_VALUE;

  if (x + 1 < data->width) {
    pos = &frame[y * data->width + x + 1];
    *pos = PIXEL_VALUE;
  }
  if (y + 1 < data->height) {
    pos = &frame[(y + 1) * data->width + x];
    *pos = PIXEL_VALUE;
  }
}

/**
 * @brief Draw line with dot at the end of line
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bouding-box internal data.
 * @param[in] coordinate of two end point of line
 */
static void
draw_line_with_dot (uint32_t * frame, pose_data * data, int x1, int y1, int x2,
    int y2)
{
  int i, dx, sx, dy, sy, err;
  uint32_t *pos;
  int xx[40] =
      { -4, 0, 4, 0, -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3
  };
  int yy[40] =
      { 0, -4, 0, 4, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -3, -2,
    -1, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, -1, 0, 1
  };

  int xs = (x1 * data->width) / data->i_width;
  int ys = (y1 * data->height) / data->i_height;
  int xe = (x2 * data->width) / data->i_width;
  int ye = (y2 * data->height) / data->i_height;

  if (xs > xe) {
    xs = (x2 * data->width) / data->i_width;
    ys = (y2 * data->height) / data->i_height;
    xe = (x1 * data->width) / data->i_width;
    ye = (y1 * data->height) / data->i_height;
  }


  for (i = 0; i < 40; i++) {
    if ((ys + yy[i] >= 0) && (ys + yy[i] < data->height) && (xs + xx[i] >= 0)
        && (xs + xx[i] < data->width)) {
      pos = &frame[(ys + yy[i]) * data->width + xs + xx[i]];
      *pos = PIXEL_VALUE;
    }
    if ((ye + yy[i] >= 0) && (ye + yy[i] < data->height) && (xe + xx[i] >= 0)
        && (xe + xx[i] < data->width)) {
      pos = &frame[(ye + yy[i]) * data->width + xe + xx[i]];
      *pos = PIXEL_VALUE;
    }
  }


  dx = abs (xe - xs);
  sx = xs < xe ? 1 : -1;
  dy = abs (ye - ys);
  sy = ys < ye ? 1 : -1;
  err = (dx > dy ? dx : -dy) / 2;

  while (setpixel (frame, data, xs, ys), xs != xe || ys != ye) {
    int e2 = err;
    if (e2 > -dx) {
      err -= dy;
      xs += sx;
    }
    if (e2 < dy) {
      err += dx;
      ys += sy;
    }
  }
}

/**
 * @brief Draw lable with the given results (pose) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bouding-box internal data.
 * @param[in] results The final results to be drawn.
 */
static void
draw_label (uint32_t * frame, pose_data * data, pose * xydata)
{
  int i, j, x1, y1, x2, y2;
  int label_len;
  uint32_t *pos1, *pos2;
  const char *label[POSE_SIZE] =
      { "top", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder",
    "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee",
    "l_ankle"
  };

  for (i = 0; i < POSE_SIZE; i++) {
    if (xydata[i].valid) {
      x1 = (xydata[i].x * data->width) / data->i_width;
      y1 = (xydata[i].y * data->height) / data->i_height;
      label_len = strlen (label[i]);
      y1 = MAX (0, (y1 - 14));
      pos1 = &frame[y1 * data->width + x1];
      for (j = 0; j < label_len; j++) {
        unsigned int char_index = label[i][j];
        if ((x1 + 8) > data->width)
          break;
        pos2 = pos1;
        for (y2 = 0; y2 < 13; y2++) {
          for (x2 = 0; x2 < 8; x2++) {
            *(pos2 + x2) = singleLineSprite[char_index][y2][x2];
          }
          pos2 += data->width;
        }
        x1 += 9;
        pos1 += 9;
      }
    }
  }
}

/**
 * @brief Draw with the given results (pose) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bouding-box internal data.
 * @param[in] results The final results to be drawn.
 */
static void
draw (GstMapInfo * out_info, pose_data * data, GArray * results)
{
  int i;
  uint32_t *frame = (uint32_t *) out_info->data;        /* Let's draw per pixel (4bytes) */
  pose *XYdata[POSE_SIZE];
  for (i = 0; i < POSE_SIZE; i++) {
    XYdata[i] = &g_array_index (results, pose, i);
    if (XYdata[i]->prob < 0.5) {
      XYdata[i]->valid = FALSE;
    }
  }

  if (XYdata[0]->valid && XYdata[1]->valid)
    draw_line_with_dot (frame, data, XYdata[0]->x, XYdata[0]->y, XYdata[1]->x,
        XYdata[1]->y);
  if (XYdata[1]->valid && XYdata[2]->valid)
    draw_line_with_dot (frame, data, XYdata[1]->x, XYdata[1]->y, XYdata[2]->x,
        XYdata[2]->y);
  if (XYdata[2]->valid && XYdata[3]->valid)
    draw_line_with_dot (frame, data, XYdata[2]->x, XYdata[2]->y, XYdata[3]->x,
        XYdata[3]->y);
  if (XYdata[3]->valid && XYdata[4]->valid)
    draw_line_with_dot (frame, data, XYdata[3]->x, XYdata[3]->y, XYdata[4]->x,
        XYdata[4]->y);
  if (XYdata[1]->valid && XYdata[5]->valid)
    draw_line_with_dot (frame, data, XYdata[1]->x, XYdata[1]->y, XYdata[5]->x,
        XYdata[5]->y);
  if (XYdata[5]->valid && XYdata[6]->valid)
    draw_line_with_dot (frame, data, XYdata[5]->x, XYdata[5]->y, XYdata[6]->x,
        XYdata[6]->y);
  if (XYdata[6]->valid && XYdata[7]->valid)
    draw_line_with_dot (frame, data, XYdata[6]->x, XYdata[6]->y, XYdata[7]->x,
        XYdata[7]->y);
  if (XYdata[1]->valid && XYdata[8]->valid)
    draw_line_with_dot (frame, data, XYdata[1]->x, XYdata[1]->y, XYdata[8]->x,
        XYdata[8]->y);
  if (XYdata[8]->valid && XYdata[9]->valid)
    draw_line_with_dot (frame, data, XYdata[8]->x, XYdata[8]->y, XYdata[9]->x,
        XYdata[9]->y);
  if (XYdata[9]->valid && XYdata[10]->valid)
    draw_line_with_dot (frame, data, XYdata[9]->x, XYdata[9]->y,
        XYdata[10]->x, XYdata[10]->y);
  if (XYdata[1]->valid && XYdata[11]->valid)
    draw_line_with_dot (frame, data, XYdata[1]->x, XYdata[1]->y,
        XYdata[11]->x, XYdata[11]->y);
  if (XYdata[11]->valid && XYdata[12]->valid)
    draw_line_with_dot (frame, data, XYdata[11]->x, XYdata[11]->y,
        XYdata[12]->x, XYdata[12]->y);
  if (XYdata[12]->valid && XYdata[13]->valid)
    draw_line_with_dot (frame, data, XYdata[12]->x, XYdata[12]->y,
        XYdata[13]->x, XYdata[13]->y);
  draw_label (frame, data, *XYdata);
}

/** @brief tensordec-plugin's TensorDecDef callback */
static GstFlowReturn
pose_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  pose_data *data = *pdata;
  const size_t size = data->width * data->height * 4;   /* RGBA */
  GstMapInfo out_info;
  GstMemory *out_mem;
  GArray *results = NULL;
  const GstTensorMemory *detections = NULL;
  float *arr;
  int index, i, j;
  gboolean status;

  g_assert (outbuf);
  /* Ensure we have outbuf properly allocated */
  if (gst_buffer_get_size (outbuf) == 0) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  status = gst_memory_map (out_mem, &out_info, GST_MAP_WRITE);
  g_assert (status);
  /** reset the buffer with alpha 0 / black */
  memset (out_info.data, 0, size);

  results = g_array_sized_new (FALSE, TRUE, sizeof (pose), POSE_SIZE);
  detections = &input[0];
  arr = detections->data;
  for (index = 0; index < POSE_SIZE; index++) {
    int maxX = 0;
    int maxY = 0;
    float max = 0.0;
    pose p;
    for (j = 0; j < data->i_height; j++) {
      for (i = 0; i < data->i_width; i++) {
        float cen = arr[i * POSE_SIZE + j * data->i_width * POSE_SIZE + index];
        if (cen > max) {
          max = cen;
          maxX = i;
          maxY = j;
        }
      }
    }
    p.valid = TRUE;
    p.x = maxX;
    p.y = maxY;
    p.prob = max;
    g_array_append_val (results, p);
  }

  draw (&out_info, data, results);
  g_array_free (results, TRUE);
  gst_memory_unmap (out_mem, &out_info);
  if (gst_buffer_get_size (outbuf) == 0)
    gst_buffer_append_memory (outbuf, out_mem);
  return GST_FLOW_OK;
}

static gchar decoder_subplugin_pose_estimation[] = "pose_estimation";
/** @brief Pose Estimation tensordec-plugin TensorDecDef instance */
static GstTensorDecoderDef poseEstimation = {
  .modename = decoder_subplugin_pose_estimation,
  .init = pose_init,
  .exit = pose_exit,
  .setOption = pose_setOption,
  .getOutCaps = pose_getOutCaps,
  .getTransformSize = pose_getTransformSize,
  .decode = pose_decode
};

/** @brief Initialize this object for tensordec-plugin */
void
init_pose (void)
{
  nnstreamer_decoder_probe (&poseEstimation);
}

/** @brief Destruct this object for tensordec-plugin */
void
finish_pose (void)
{
  nnstreamer_decoder_exit (poseEstimation.modename);
}
