/**
 * NNStreamer OpenCV Custom Filter Example: Average
 * Copyright (C) 2018 Sangjung Woo <sangjung.woo@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnstreamer_customfilter_opencv_average.cc
 * @date  25 Oct 2018
 * @brief  OpenCV Custom NNStreamer Filter Example: Average
 * @author  Sangjung Woo <sangjung.woo@samsung.com>
 * @bug  No known bugs
 * @see  nnstreamer_customfilter_example_average.c
 *
 * This example calculates the average value of input tensor for
 * each channel (i.e. R, G & B). The shape of the input tensor is
 * [N][y][x][M] and that of the output tensor is [N][1][1][M].
 */

#include <opencv2/opencv.hpp>

#include <glib.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api.h>

/**
 * @brief _pt_data Internal data structure
 */
typedef struct _pt_data
{
  uint32_t in_height;  /***< height of input tensor */
  uint32_t in_width;   /***< width of input tensor */
  uint32_t in_channel; /***< channel of input tensor */
} pt_data;

/**
 * @brief pt_init
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  pt_data *pdata = g_new0 (pt_data, 1);
  g_assert (pdata != NULL);

  return pdata;
}

/**
 * @brief pt_exit
 */
static void
pt_exit (void *private_data, const GstTensorFilterProperties * prop)
{
  pt_data *pdata = static_cast<pt_data *> (private_data);
  g_assert (pdata);
  g_free (pdata);
}

/**
 * @brief set_inputDim
 */
static int
set_inputDim (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  int i;
  pt_data *pdata = static_cast<pt_data *> (private_data);

  g_assert (pdata);
  g_assert (in_info);
  g_assert (out_info);

  out_info->num_tensors = 1;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    out_info->info[0].dimension[i] = in_info->info[0].dimension[i];

  /* Save width, height and channel size of an input tensor */
  pdata->in_width = in_info->info[0].dimension[1];
  pdata->in_height = in_info->info[0].dimension[2];
  pdata->in_channel = in_info->info[0].dimension[0];

  /* Update output dimension [1] and [2] with new-x, new-y */
  out_info->info[0].dimension[1] = 1;
  out_info->info[0].dimension[2] = 1;

  out_info->info[0].type = in_info->info[0].type;
  return 0;
}

/**
 * @brief pt_invoke
 */
static int
pt_invoke (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  pt_data *pdata = static_cast<pt_data *> (private_data);
  size_t in_size;
  uint8_t *optr;
  cv::Mat img_src;
  std::vector<cv::Mat> channels;
  cv::Scalar mean_result;
  void *buffer;

  g_assert (pdata);
  g_assert (input);
  g_assert (output);

  in_size = gst_tensor_info_get_size (&prop->input_meta.info[0]);
  buffer = g_malloc (in_size);
  g_assert (buffer != NULL);

  /* Get Mat object from input tensor */
  memcpy (buffer, input[0].data, in_size);
  img_src = cv::Mat (pdata->in_height, pdata->in_width, CV_8UC3, buffer);

  /* Get the channel info from Mat object */
  cv::split(img_src, channels);

  /* Calculate an average of each channel */
  optr = static_cast<uint8_t *> (output[0].data);
  for (uint32_t i = 0; i < pdata->in_channel; ++i) {
    mean_result = cv::mean(channels[i]);
    *optr = static_cast<uint8_t> (mean_result[0]);
    optr++;
  }

  g_assert (input[0].data != output[0].data);
  g_free(buffer);

  return 0;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = pt_init,
  .exitfunc = pt_exit,
  .getInputDim = NULL,
  .getOutputDim = NULL,
  .setInputDim = set_inputDim,
  .invoke = pt_invoke,
  .allocate_invoke = NULL,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
