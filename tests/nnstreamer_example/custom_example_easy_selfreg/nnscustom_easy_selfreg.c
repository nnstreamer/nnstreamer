/**
 * NNStreamer custom-easy filter that registers itself when its library is loaded
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnscustom_easy_selfreg.c
 * @date  06 Oct 2026
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief  Custom-easy filter registered from the constructor of its shared library.
 * @bug  No known bugs
 *
 * tensor_filter loads this library by its model name, "libnnscustom_easy_selfreg",
 * from the custom filter path. The model takes four uint8 values and adds one to each.
 */

#include <glib.h>
#include <tensor_filter_custom_easy.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

void init_nnscustom_easy_selfreg (void) __attribute__((constructor));

/**
 * @brief Invoke callback: each output byte is the input byte plus one.
 */
static int
selfreg_invoke (void *data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  gsize i;

  UNUSED (data);
  UNUSED (prop);

  for (i = 0; i < input[0].size; i++)
    ((guint8 *) output[0].data)[i] = ((const guint8 *) input[0].data)[i] + 1;
  return 0;
}

/**
 * @brief Register the model while the library is being loaded.
 */
void
init_nnscustom_easy_selfreg (void)
{
  GstTensorsInfo info;

  gst_tensors_info_init (&info);
  info.num_tensors = 1;
  info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("4:1:1:1", info.info[0].dimension);

  NNS_custom_easy_register ("libnnscustom_easy_selfreg", selfreg_invoke, NULL,
      &info, &info);
  gst_tensors_info_free (&info);
}
