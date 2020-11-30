/**
 * NNStreamer OpenCV Custom Filter Example: Scaler
 *
 * Copyright (C) 2018 Sangjung Woo <sangjung.woo@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnstreamer_customfilter_opencv_scaler.cc
 * @date  22 Oct 2018
 * @brief  OpenCV Custom NNStreamer Filter Example: Scaler
 * @author  Sangjung Woo <sangjung.woo@samsung.com>
 * @bug  No known bugs
 * @see  nnstreamer_customfilter_example_scaler_allocator.c
 *
 * This example scales an input tensor of [N][input_h][input_w][M]
 * to an ouput tensor of [N][output_h][output_w][M].
 *
 * The custom property is to be given as, "custom=[new-x]x[new-y]", where new-x and new-y are unsigned integers.
 * E.g., custom=640x480
 *
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <glib.h>
#include <nnstreamer_plugin_api.h>
#include <tensor_filter_custom.h>

/**
 * @brief Private data structure
 */
typedef struct _pt_data {
  uint32_t in_height;
  uint32_t in_width;
  uint32_t out_height;
  uint32_t out_width;
} pt_data;

/**
 * @brief init callback of tensor_filter custom
 */
static void *
pt_init (const GstTensorFilterProperties *prop)
{
  pt_data *data = g_new0 (pt_data, 1);
  g_assert (data != NULL);

  data->out_width = 0;
  data->out_height = 0;

  /* In case that custom property is given */
  if (prop->custom_properties && strlen (prop->custom_properties) > 0) {
    const char s[7] = "xX:_/ ";
    gchar **strv = g_strsplit_set (prop->custom_properties, s, 3);
    if (strv[0] != NULL) {
      data->out_width = (uint32_t)g_ascii_strtoll (strv[0], NULL, 10);
    } else {
      data->out_width = 0;
    }
    if (strv[1] != NULL) {
      data->out_height = (uint32_t)g_ascii_strtoll (strv[1], NULL, 10);
    } else {
      data->out_height = 0;
    }
    g_strfreev (strv);
  }

  return data;
}

/**
 * @brief exit callback of tensor_filter custom
 */
static void
pt_exit (void *private_data, const GstTensorFilterProperties *prop)
{
  pt_data *pdata = static_cast<pt_data *> (private_data);
  g_assert (pdata);
  g_free (pdata);
}

/**
 * @brief setInputDimension callback of tensor_filter custom
 */
static int
set_inputDim (void *private_data, const GstTensorFilterProperties *prop,
    const GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  pt_data *pdata = static_cast<pt_data *> (private_data);

  g_assert (pdata);
  g_assert (in_info);
  g_assert (out_info);

  out_info->num_tensors = 1;

  for (int i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    out_info->info[0].dimension[i] = in_info->info[0].dimension[i];

  /* Save width and height of an input tensor */
  pdata->in_width = in_info->info[0].dimension[1];
  pdata->in_height = in_info->info[0].dimension[2];

  /* Update output dimension [1] and [2] with new-x, new-y */
  if (pdata->out_width > 0)
    out_info->info[0].dimension[1] = pdata->out_width;
  if (pdata->out_height > 0)
    out_info->info[0].dimension[2] = pdata->out_height;

  out_info->info[0].type = in_info->info[0].type;
  return 0;
}

/**
 * @brief invoke-alloc callback of tensor_filter custom
 */
static int
pt_allocate_invoke (void *private_data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  pt_data *pdata = static_cast<pt_data *> (private_data);
  size_t in_size, out_size;
  void *buffer;
  cv::Mat img_src, img_dst;

  g_assert (pdata);
  g_assert (input);
  g_assert (output);

  in_size = gst_tensor_info_get_size (&prop->input_meta.info[0]);
  out_size = gst_tensor_info_get_size (&prop->output_meta.info[0]);
  buffer = g_malloc (in_size);
  g_assert (buffer != NULL);
  output[0].data = g_malloc (out_size);
  g_assert (output[0].data != NULL);

  /* Get Mat object from input tensor */
  memcpy (buffer, input[0].data, in_size);
  img_src = cv::Mat (pdata->in_height, pdata->in_width, CV_8UC3, buffer);
#if CV_MAJOR_VERSION >= 3
  cv::cvtColor (img_src, img_src, cv::COLOR_BGR2RGB);
#else
  cv::cvtColor (img_src, img_src, CV_BGR2RGB);
#endif

/* Scale from the shape of input tensor to that of output tensor
 * which is given as custom property */
#if CV_MAJOR_VERSION >= 3
  cv::resize (img_src, img_dst, cv::Size (pdata->out_width, pdata->out_height),
      0, 0, cv::INTER_NEAREST);
#else
  cv::resize (img_src, img_dst, cv::Size (pdata->out_width, pdata->out_height),
      0, 0, CV_INTER_NN);
#endif

/* Convert Mat object to output tensor */
#if CV_MAJOR_VERSION >= 3
  cv::cvtColor (img_dst, img_dst, cv::COLOR_RGB2BGR);
#else
  cv::cvtColor (img_dst, img_dst, CV_RGB2BGR);
#endif
  memcpy (output[0].data, img_dst.data, out_size);

  g_free (buffer);

  return 0;
}

/**
 * @brief destroy notify callback of tensor_filter custom
 */
static void
pt_destroy_notify (void *data)
{
  g_assert (data);
  g_free (data);
}

/**
 * @brief tensor_filter custom subplugin definition
 */
static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = pt_init,
  .exitfunc = pt_exit,
  .getInputDim = NULL,
  .getOutputDim = NULL,
  .setInputDim = set_inputDim,
  .invoke = NULL,
  .allocate_invoke = pt_allocate_invoke,
  .destroy_notify = pt_destroy_notify,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
