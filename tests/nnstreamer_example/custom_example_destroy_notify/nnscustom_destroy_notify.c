/**
 * NNStreamer custom filter that allocates its output and releases it with destroy_notify
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnscustom_destroy_notify.c
 * @date  30 Oct 2026
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief  A pass-through custom filter with allocate_invoke and destroy_notify.
 * @bug  No known bugs
 *
 * 'nnscustom_destroy_notify_count' counts the destroy_notify calls, so that a test case
 * can tell whether tensor_filter has handed the output it allocated back to the library.
 */

#include <string.h>
#include <glib.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

gint nnscustom_destroy_notify_count = 0; /**< destroy_notify calls so far */

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void *
dn_init (const GstTensorFilterProperties * prop)
{
  UNUSED (prop);

  return g_malloc0 (1);
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void
dn_exit (void *_data, const GstTensorFilterProperties * prop)
{
  UNUSED (prop);

  g_free (_data);
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
dn_setInputDim (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  UNUSED (_data);
  UNUSED (prop);

  gst_tensors_info_copy (out_info, in_info);
  return 0;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 * @details Allocates each output and copies the input of the same index into it.
 */
static int
dn_allocate_invoke (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  unsigned int i;

  UNUSED (_data);

  if (prop->input_meta.num_tensors != prop->output_meta.num_tensors)
    return -1;

  for (i = 0; i < prop->input_meta.num_tensors; i++) {
    if (input[i].size != output[i].size)
      return -1;

    output[i].data = g_malloc (output[i].size);
    memcpy (output[i].data, input[i].data, input[i].size);
  }

  return 0;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void
dn_destroy_notify (void *data)
{
  g_atomic_int_inc (&nnscustom_destroy_notify_count);
  g_free (data);
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = dn_init,
  .exitfunc = dn_exit,
  .setInputDim = dn_setInputDim,
  .allocate_invoke = dn_allocate_invoke,
  .destroy_notify = dn_destroy_notify,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
