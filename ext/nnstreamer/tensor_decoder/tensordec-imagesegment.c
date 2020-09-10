/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "image segment"
 * Copyright (C) 2019 Jihoon Lee <ulla4571@gmail.com>
 * Copyright (C) 2019 niklasjang <niklasjang@gmail.com>
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
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
 *          Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * option1: Decoder mode of image segmentation
 *          Available : tflite-deeplab
 *          Available : snpe-deeplab
 *          Available : snpe-depth
 *
 * option2: Maximum number of class labels (except background), default is 20 (Pascal)
 *
 * expected models
 * - tflite-deeplab : deeplabv3_257_mv_gpu.tflite (designed for embedded devices)
 * - snpe-deeplab   : deeplabv3_mnv2_pascal_train_aug.dlc (converted from a TF model)
 * - snpe-depth     : any snpe models (.dlc) producing grayscale images
 *
 * expected input dims
 * - tflite-deeplab : #labels x width x height (float32, label probability)
 *                    (e.g., 21 x 257 x 257)
 * - snpe-deeplab   : width x height x 1 (float32, label index)
 *                    (e.g., 513 x 513 x 1)
 * - snpe-depth     : 1 x width x height (float32, grayscale)
 *                    (e.g., 1 x 320 x 240)
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
#include <nnstreamer_log.h>
#include "tensordecutil.h"

#if defined(__aarch64__)
#include <arm_neon.h>

#define NEON64_ENABLED
#define GRAYSCALE_HEX (0x00010101)
#define ALPHA_HEX     (0xFF000000)
#endif

#define DEFAULT_LABELS  (20)
#define RGBA_CHANNEL    (4)
#define MAX_RGB         (255)

void init_is (void) __attribute__ ((constructor));
void fini_is (void) __attribute__ ((destructor));

const static float DETECTION_THRESHOLD = 0.5f;

/**
 * @brief There can be different schemes for image segmentation
 */
typedef enum
{
  MODE_TFLITE_DEEPLAB = 0,
  MODE_SNPE_DEEPLAB = 1,
  MODE_SNPE_DEPTH = 2,
  MODE_UNKNOWN,
} image_segment_modes;

/**
 * @brief List of image-segmentation decoding schemes in string
 */
static const char *is_modes[] = {
  [MODE_TFLITE_DEEPLAB] = "tflite-deeplab",
  [MODE_SNPE_DEEPLAB] = "snpe-deeplab",
  [MODE_SNPE_DEPTH] = "snpe-depth",
  NULL,
};

/**
 * @brief Data structure for image segmentation info
 */
typedef struct
{
  image_segment_modes mode; /**< The image segmentation decoding mode */
  float *segment_map;       /**< The image segmentated map */

  guint max_labels;         /**< Maximum number of labels */
  guint *color_map;         /**< The RGBA color map (up to max labels) */

  guint width;              /**< Input video width */
  guint height;             /**< Input video height */

  GRand *rand;              /**< random value generator */
  guint rgb_modifier;       /**< rgb modifier according to # labels */
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

  idata->rand = g_rand_new ();
  idata->mode = MODE_UNKNOWN;
  idata->width = 0;
  idata->height = 0;
  idata->max_labels = DEFAULT_LABELS;
  idata->segment_map = NULL;
  idata->color_map = NULL;
  idata->rgb_modifier = 0;

  return TRUE;
}

/** @brief Free the allocated resources */
static void
_free_resources (image_segments * idata)
{
  g_free (idata->segment_map);
  g_free (idata->color_map);
  g_rand_free (idata->rand);

  idata->segment_map = NULL;
  idata->color_map = NULL;
  idata->rand = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
is_exit (void **pdata)
{
  image_segments *idata = *pdata;

  _free_resources (idata);

  g_free (*pdata);
  *pdata = NULL;
}

/** @brief fill rgba color map */
static void
_fill_color_map (image_segments * idata)
{
  guint i;

  idata->color_map[0] = 0; /* background */

#if defined (NEON64_ENABLED)
  idata->rgb_modifier = 0xFFFFFF / (idata->max_labels + 1);
  for (i = 1; i <= idata->max_labels; i++) {
    /* colors should be the same with neon calculations */
    idata->color_map[i] = idata->rgb_modifier * i;
    ((guint8 *)&idata->color_map[i])[3] = '\xff'; /* alpha */
  }
#else
  for (i = 1; i <= idata->max_labels; i++) {
    /* any color value would be acceptable */
    idata->color_map[i] = g_rand_int_range (idata->rand, 0x101010, 0xFFFFFF);
    ((guint8 *)&idata->color_map[i])[3] = '\xff'; /* alpha */
  }
#endif
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
  } else if (op_num == 1) {
    guint64 max_labels_64 = g_ascii_strtoll (param, NULL, 10);
    if (max_labels_64 != 0 && max_labels_64 <= UINT_MAX)
      idata->max_labels = (guint) max_labels_64;
  }

  GST_WARNING ("mode-option-\"%d\" is not definded.", op_num);
  return TRUE;
}

/** @brief Initialize image_segments per mode */
static gboolean
_init_modes (image_segments * idata)
{
  if (idata->mode == MODE_TFLITE_DEEPLAB) {
    /* init image segments if seg map is null */
    if (idata->segment_map == NULL)
      idata->segment_map = g_new0 (float, idata->height * idata->width);

    if (idata->color_map == NULL) {
      idata->color_map = g_new (guint, idata->max_labels + 1);
      _fill_color_map (idata);
    }

    return TRUE;
  } else if (idata->mode == MODE_SNPE_DEEPLAB) {
    if (idata->color_map == NULL) {
      idata->color_map = g_new (guint, idata->max_labels + 1);
      _fill_color_map (idata);
    }
    return TRUE;
  } else if (idata->mode == MODE_SNPE_DEPTH) {
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
  GstCaps *caps;
  char *str;

  g_return_val_if_fail (config != NULL, NULL);
  GST_INFO ("Num Tensors = %d", config->info.num_tensors);
  g_return_val_if_fail (config->info.num_tensors >= 1, NULL);

  if (idata->mode == MODE_SNPE_DEEPLAB) {
    idata->width = config->info.info[0].dimension[0];
    idata->height = config->info.info[0].dimension[1];
  } else {
    idata->width = config->info.info[0].dimension[1];
    idata->height = config->info.info[0].dimension[2];
  }

  str = g_strdup_printf ("video/x-raw, format = RGBA, "
      "width = %u, height = %u", idata->width, idata->height);
  caps = gst_caps_from_string (str);
  setFramerateFromConfig (caps, config);
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

/** @brief Set color according to each pixel's label (RGBA) */
static void
set_color_according_to_label (image_segments * idata, GstMapInfo * out_info)
{
  float *input = idata->segment_map;
  uint32_t *output = (uint32_t *) out_info->data;
  guint num_pixels = idata->height * idata->width;
  guint label_idx, idx = 0;

#if defined (NEON64_ENABLED)
  float32x4_t v_src_float;

  uint32x4_t v_src_uint;
  uint32x4_t v_magic;
  uint32x4_t v_mask;
  uint32x4_t v_alpha;
  uint32x4_t v_zero;

  guint num_lanes = 4;

  v_magic = vdupq_n_u32 (idata->rgb_modifier);
  v_alpha = vdupq_n_u32 (ALPHA_HEX);
  v_zero = vdupq_n_u32 (0);

  for (idx = 0; idx < num_pixels; idx += num_lanes) {
    /* load float32 vector */
    v_src_float = vld1q_f32 (input);
    input += num_lanes;

    /* convert float32 vector to uint32 vector */
    v_src_uint = vcvtq_u32_f32 (v_src_float);

    /* multiply by magic number to fill RGB values */
    v_src_uint = vmulq_u32 (v_src_uint, v_magic);

    /* check whether the label is zero (i.e., background) */
    v_mask = vceqq_u32 (v_src_uint, v_zero);
    v_mask = vbslq_u32 (v_mask, v_zero, v_alpha);

    /* set the alpha value unless it's background */
    v_src_uint = vorrq_u32 (v_src_uint, v_mask);

    /* store uint32 vector */
    vst1q_u32 (output, v_src_uint);
    output += num_lanes;
  }

  if (num_pixels == idx)
    return;

  /* handle remaining data */
  input = (float *) idata->segment_map;
  output = (uint32_t *) out_info->data;
  idx -= num_lanes;
#endif
  for (; idx < num_pixels; idx++) {
    label_idx = (guint) input[idx];

    /* If out-of-range, don't draw it */
    if (G_UNLIKELY (label_idx > idata->max_labels))
      continue;

    output[idx] = idata->color_map[label_idx];
  }
}

/** @brief Find the maximum grayscale value */
static float
find_max_grayscale (image_segments * idata)
{
  float *input = idata->segment_map;
  float gray_max = 0.0;
  guint num_pixels = idata->height * idata->width;
  guint idx = 0;

#if defined (NEON64_ENABLED)
  float32x4_t v_src, v_max;
  guint num_lanes = 4;

  v_max = vdupq_n_f32 (0);

  /* find the maximum value per lane */
  for (idx = 0; idx < num_pixels; idx += num_lanes) {
    v_src = vld1q_f32 (input);
    input += num_lanes;

    v_max = vmaxq_f32 (v_src, v_max);
  }

  /* find the maximum value among all lanes */
  gray_max = MAX (gray_max, vgetq_lane_f32 (v_max, 0));
  gray_max = MAX (gray_max, vgetq_lane_f32 (v_max, 1));
  gray_max = MAX (gray_max, vgetq_lane_f32 (v_max, 2));
  gray_max = MAX (gray_max, vgetq_lane_f32 (v_max, 3));

  if (num_pixels == idx)
    return gray_max;

  /* handle remaining data */
  input = idata->segment_map;
  idx -= num_lanes;
#endif
  for (; idx < num_pixels; idx++)
    gray_max = MAX (gray_max, input [idx]);

  return gray_max;
}

/** @brief Set color with grayscale value */
static void
set_color_grayscale (image_segments * idata, GstMapInfo * out_info)
{
  float *input = idata->segment_map;
  uint32_t *output = (uint32_t *) out_info->data;
  float max_grayscale;
  guint num_pixels = idata->height * idata->width;
  guint grayscale;
  guint idx = 0;

  /* find the maximum grayscale value */
  max_grayscale = find_max_grayscale (idata);
  if (G_UNLIKELY (max_grayscale == 0.0))
    return;

#if defined (NEON64_ENABLED)
  {
    float32x4_t v_src_float;
    float32x4_t v_max_gray;
    float32x4_t v_max_rgb;

    uint32x4_t v_src_uint;
    uint32x4_t v_magic;
    uint32x4_t v_alpha;

    guint num_lanes = 4;

    v_max_gray = vdupq_n_f32 (max_grayscale);
    v_max_rgb = vdupq_n_f32 (MAX_RGB);
    v_magic = vdupq_n_u32 (GRAYSCALE_HEX);
    v_alpha = vdupq_n_u32 (ALPHA_HEX);

    for (idx = 0; idx < num_pixels; idx += num_lanes) {
      /* load float32 vector */
      v_src_float = vld1q_f32 (input);
      input += num_lanes;

      /* normalized_gray = (gray / max_gray) x max_rgb */
      v_src_float = vdivq_f32 (v_src_float, v_max_gray);
      v_src_float = vmulq_f32 (v_src_float, v_max_rgb);

      /* convert float32 vector to uint32 vector */
      v_src_uint = vcvtq_u32_f32 (v_src_float);

      /* multiply by magic number to fill the same RGB values */
      v_src_uint = vmulq_u32 (v_src_uint, v_magic);
      v_src_uint = vaddq_u32 (v_src_uint, v_alpha);

      /* store uint32 vector */
      vst1q_u32 (output, v_src_uint);
      output += num_lanes;
    }

    if (num_pixels == idx)
      return;

    /* handle remaining data */
    input = idata->segment_map;
    output = (uint32_t *) out_info->data;
    idx -= num_lanes;
  }
#endif
  for (; idx < num_pixels; idx++) {
    /* normalize grayscale values to RGB_MAX */
    grayscale = (guint) ((input[idx] / max_grayscale) * MAX_RGB);

    /* Should be less than 256 */
    if (G_UNLIKELY (grayscale > MAX_RGB))
      continue;

    grayscale = grayscale | (grayscale << 8) | (grayscale << 16) | 0xFF000000;
    output[idx] = grayscale;
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
  guint total_labels = idata->max_labels + 1;

  memset (idata->segment_map, '\x00',
      idata->width * idata->height * sizeof (float));

  for (i = 0; i < idata->height; i++) {
    for (j = 0; j < idata->width; j++) {
      max_idx = 0;
      max_prob = prob_map[i * idata->width * total_labels
        + j * total_labels];
      for (idx = 1; idx < total_labels; idx++) {
        float prob = prob_map[i * idata->width * total_labels
          + j * total_labels + idx];
        if (prob > max_prob) {
          max_prob = prob;
          max_idx = idx;
        }
      }
      if (max_prob > DETECTION_THRESHOLD) {
        idata->segment_map[i * idata->width + j] = (float) max_idx;
      } /* otherwise, regarded as background */
    }
  }
}

/** @brief set color to output buffer depending on each mode */
static void
set_color (image_segments * idata, void *data, GstMapInfo * out_info)
{
  /* tflite-deeplab needs to perform extra post-processing to set labels */
  if (idata->mode == MODE_TFLITE_DEEPLAB) {
    set_label_index (idata, data);
    set_color_according_to_label (idata, out_info);
    return;
  }

  /* snpe-deeplab already has labeled data as input */
  idata->segment_map = data;

  if (idata->mode == MODE_SNPE_DEEPLAB)
    set_color_according_to_label (idata, out_info);
  else if (idata->mode == MODE_SNPE_DEPTH)
    set_color_grayscale (idata, out_info);

  idata->segment_map = NULL;
}

/** @brief sanity check for each mode */
static gboolean
check_sanity (image_segments * idata, const GstTensorsConfig * config)
{
  if (idata->mode == MODE_TFLITE_DEEPLAB) {
    return (config->info.info[0].type == _NNS_FLOAT32) &&
           (config->info.info[0].dimension[0] == idata->max_labels + 1);
  } else if (idata->mode == MODE_SNPE_DEEPLAB) {
    return (config->info.info[0].type == _NNS_FLOAT32);
  } else if (idata->mode == MODE_SNPE_DEPTH) {
    return (config->info.info[0].type == _NNS_FLOAT32) &&
           (config->info.info[0].dimension[0] == 1);
  }

  return FALSE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
is_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  image_segments *idata = *pdata;
  const size_t size = idata->width * idata->height * RGBA_CHANNEL;
  gboolean need_output_alloc;
  GstMapInfo out_info;
  GstMemory *out_mem;

  if (FALSE == _init_modes (idata) || outbuf == NULL)
    return GST_FLOW_ERROR;

  need_output_alloc = (gst_buffer_get_size (outbuf) == 0);
  if (TRUE == need_output_alloc) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  if (FALSE == gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    ml_loge ("Cannot map output memory / tensordec-imagesegment.\n");
    goto error_free;
  }

  memset (out_info.data, '\x00', size);

  if (FALSE == check_sanity (idata, config)) {
    ml_loge ("Invalid input data format detected.\n");
    goto error_unmap;
  }

  set_color (idata, input->data, &out_info);

  gst_memory_unmap (out_mem, &out_info);

  if (TRUE == need_output_alloc)
    gst_buffer_append_memory (outbuf, out_mem);

  return GST_FLOW_OK;

error_unmap:
  gst_memory_unmap (out_mem, &out_info);
error_free:
  if (TRUE == need_output_alloc)
    gst_allocator_free (NULL, out_mem);

  return GST_FLOW_ERROR;
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
