/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "image segment"
 * Copyright (C) 2019 Jihoon Lee <ulla4571@gmail.com>
 * Copyright (C) 2019 niklasjang <niklasjang@gmail.com>
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
 * @file	tensordec-imagesegment.c
 * @date	19 Oct 2019
 * @brief	NNStreamer tensor-decoder subplugin, "image segment",
 *              which detects objects and paints their regions.
 *
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author  Jihoon Lee <ulla4571@gmail.com>
 *          niklasjang <niklasjang@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 * option1: Decoder mode of image segmentation
 *          Available : tflite-deeplab
 *
 * pipeline:
 * filesrc
 *    |
 * decodebin
 *    |
 * videoconvert
 *    |
 * videoscale
 *    |
 * imagefreeze -- tee ----------------------------------------------- videomixer -- videoconvert -- autovideosink
 *                 |                                                       |
 *          tensor_converter -- tensor_transform -- tensor_filter -- tensor_decoder
 *
 * - Used model is deeplabv3_257_mv_gpu.tflite.
 * - Resize image into 257:257 at the first videoscale.
 * - Transfrom RGB value into float32 in range [0,1] at tensor_transform.
 *
 * gst-launch-1.0 -v \
 *    filesrc location=cat.png ! decodebin ! videoconvert ! videoscale ! imagefreeze !\
 *    video/x-raw,format=RGB,width=257,height=257,framerate=10/1 ! tee name=t \
 *    t. ! queue ! mix. \
 *    t. ! queue ! tensor_converter !\
 *    tensor_transform mode=arithmetic option=typecast:float32,add:0.0,div:255.0 !\
 *    tensor_filter framework=tensorflow-lite model=deeplabv3_257_mv_gpu.tflite !\
 *    tensor_decoder mode=image_segment option1=tflite-deeplab ! mix. \
 *    videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.6 ! videoconvert !  videoscale ! autovideosink \
 */

#include <string.h>
#include <glib.h>
#include <gst/video/video-format.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api.h>

void init_is (void) __attribute__ ((constructor));
void fini_is (void) __attribute__ ((destructor));

#define RGBA_CHANNEL                   4
#define TFLITE_DEEPLAB_TOTAL_LABELS    21

const static float DETECTION_THRESHOLD = 0.5f;

/**
 * @brief There can be different schemes for image segmentation
 */
typedef enum
{
  MODE_TFLITE_DEEPLAB = 0,
  MODE_UNKNOWN,
} image_segment_modes;

/**
 * @brief List of image-segmentation decoding schemes in string
 */
static const char *is_modes[] = {
  [MODE_TFLITE_DEEPLAB] = "tflite-deeplab",
  NULL,
};

/**
 * @brief Data structure for image segmentation info
 */
typedef struct
{
  image_segment_modes mode; /**< The image segmentation decoding mode */
  guint **segment_map;  /**< The image segmentated map */

  guint width; /**< Input video width */
  guint height; /**< Input video height */
} image_segments;

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
is_init (void **pdata)
{
  image_segments *idata;

  idata = *pdata = g_new0 (image_segments, 1);
  if (idata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  idata->mode = MODE_UNKNOWN;
  idata->width = 0;
  idata->height = 0;
  idata->segment_map = NULL;

  return TRUE;
}

/** @brief Free the allocated segment_map */
static void
_free_segment_map (image_segments * idata)
{
  int i;

  if (idata->segment_map) {
    for (i = 0; i < idata->height; i++) {
      g_free (idata->segment_map[i]);
    }
    g_free (idata->segment_map);
  }

  idata->segment_map = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
is_exit (void **pdata)
{
  image_segments *idata = *pdata;

  _free_segment_map (idata);

  g_free (*pdata);
  *pdata = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
is_setOption (void **pdata, int op_num, const char *param)
{
  image_segments *idata = *pdata;

  if (op_num == 0) {
    /* The first option indicates mode of image segmentation decoder */
    image_segment_modes previous = idata->mode;
    idata->mode = find_key_strv (is_modes, param);

    if (NULL == param || *param == '\0') {
      GST_ERROR ("Please set the valid mode at option1");
      return FALSE;
    }

    if (idata->mode != previous && idata->mode != MODE_UNKNOWN) {
      return TRUE;
    }
    return TRUE;
  }

  GST_WARNING ("mode-option-\"%d\" is not definded.", op_num);
  return TRUE;
}

/** @brief Initialize image_segments per mode */
static gboolean
_init_modes (image_segments * idata)
{
  if (idata->mode == MODE_TFLITE_DEEPLAB) {
    int i;

    idata->segment_map = g_new0 (guint *, idata->height);
    g_assert (idata->segment_map != NULL);
    for (i = 0; i < idata->height; i++) {
      idata->segment_map[i] = g_new0 (guint, idata->width);
      g_assert (idata->segment_map[i] != NULL);
    }

    return TRUE;
  }

  GST_ERROR ("Failed to initialize, unknown mode %d.", idata->mode);
  return FALSE;
}

/**
 * @brief tensordec-plugin's GstTensorDecoderDef callback
 *
 * [DeeplabV3 model]
 * Just one tensor with [21(#labels):width:height:1], float32
 * Probability that each pixel is assumed to be labeled object.
 */
static GstCaps *
is_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
  image_segments *idata = *pdata;
  gint fn, fd;
  GstCaps *caps;
  char *str;

  if (idata->mode == MODE_TFLITE_DEEPLAB) {
    g_return_val_if_fail (config != NULL, NULL);
    GST_INFO ("Num Tensors = %d", config->info.num_tensors);
    g_return_val_if_fail (config->info.num_tensors >= 1, NULL);

    if (idata->width == 0 || idata->height == 0) {
      idata->width = config->info.info[0].dimension[1];
      idata->height = config->info.info[0].dimension[2];
    }
  }

  str = g_strdup_printf ("video/x-raw, format = RGBA, "
      "width = %u, height = %u", idata->width, idata->height);
  caps = gst_caps_from_string (str);
  fn = config->rate_n; /** @todo Verify if this rate is ok */
  fd = config->rate_d;

  if (fn >= 0 && fd > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, fn, fd, NULL);
  }
  g_free (str);

  return gst_caps_simplify (caps);
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
is_getTransformSize (void **pdata, const GstTensorsConfig * config,
    GstCaps * caps, size_t size, GstCaps * othercaps, GstPadDirection direction)
{
  return 0;
  /** @todo Use appropriate values */
}


/** @brief Set color according to each pixel's max probability  */
static void
set_color_according_to_label (image_segments * idata, GstMapInfo * out_info)
{
  uint32_t *frame = (uint32_t *) out_info->data;
  uint32_t *pos;
  int i, j;
  const uint32_t label_color[21] = {
    0xFF000040, 0xFF800000, 0xFFFFEFD5, 0xFF40E0D0, 0xFFFFA500,
    0xFF00FF00, 0xFFDC143C, 0xFFF0F8FF, 0xFF008000, 0xFFEE82EE,
    0xFF808080, 0xFF4169E1, 0xFF008080, 0xFFFF6347, 0xFF000000,
    0xFFFF4500, 0xFFDA70D6, 0xFFEEE8AA, 0xFF98FB98, 0xFFAFEEEE,
    0xFFFFF5EE
  };

  for (i = 0; i < idata->height; i++) {
    for (j = 0; j < idata->width; j++) {
      int label_idx = idata->segment_map[i][j];
      g_assert (label_idx >= 0 && label_idx <= 20);
      pos = &frame[i * idata->width + j];
      if (label_idx == 0)
        continue;               /*Do not set color for background */
      *pos = label_color[label_idx];
    }
  }
}


/** @brief Set label index according to each pixel's label probabilities */
static void
set_label_index (image_segments * idata, void *data)
{
  float *prob_map = (float *) data;
  int idx, i, j;
  int max_idx;
  float max_prob;

  for (i = 0; i < idata->height; i++) {
    memset (idata->segment_map[i], 0, idata->width * sizeof (guint));
  }

  for (i = 0; i < idata->height; i++) {
    for (j = 0; j < idata->width; j++) {
      max_idx = 0;
      max_prob = prob_map[i * idata->width * TFLITE_DEEPLAB_TOTAL_LABELS
          + j * TFLITE_DEEPLAB_TOTAL_LABELS];
      for (idx = 1; idx < TFLITE_DEEPLAB_TOTAL_LABELS; idx++) {
        float prob = prob_map[i * idata->width * TFLITE_DEEPLAB_TOTAL_LABELS
            + j * TFLITE_DEEPLAB_TOTAL_LABELS + idx];
        if (prob > max_prob) {
          max_prob = prob;
          max_idx = idx;
        }
      }
      if (max_prob > DETECTION_THRESHOLD) {
        idata->segment_map[i][j] = max_idx;
      }
    }
  }
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
is_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  image_segments *idata = *pdata;
  const size_t size = idata->width * idata->height * RGBA_CHANNEL;
  GstMapInfo out_info;
  GstMemory *out_mem;
  gboolean status;

  /* init image segments if seg map is null */
  if (idata->segment_map == NULL) {
    if (!_init_modes (idata))
      return GST_FLOW_ERROR;
  }

  g_assert (outbuf);
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

  memset (out_info.data, 0, size);

  if (idata->mode == MODE_TFLITE_DEEPLAB) {
    g_assert (config->info.info[0].type == _NNS_FLOAT32);
    g_assert (config->info.info[0].dimension[0] == TFLITE_DEEPLAB_TOTAL_LABELS);
    set_label_index (idata, input->data);
  }

  set_color_according_to_label (idata, &out_info);

  gst_memory_unmap (out_mem, &out_info);

  if (gst_buffer_get_size (outbuf) == 0)
    gst_buffer_append_memory (outbuf, out_mem);

  return GST_FLOW_OK;
}

static gchar decoder_subplugin_image_segment[] = "image_segment";

/** @brief Image Segmentation tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef imageSegment = {
  .modename = decoder_subplugin_image_segment,
  .init = is_init,
  .exit = is_exit,
  .setOption = is_setOption,
  .getOutCaps = is_getOutCaps,
  .getTransformSize = is_getTransformSize,
  .decode = is_decode
};

/** @brief Initialize this object for tensordec-plugin */
void
init_is (void)
{
  nnstreamer_decoder_probe (&imageSegment);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_is (void)
{
  nnstreamer_decoder_exit (imageSegment.modename);
}
