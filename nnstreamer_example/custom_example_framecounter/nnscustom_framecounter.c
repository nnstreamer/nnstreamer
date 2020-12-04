/**
 * "Frame Counter"
 * NNStreamer Custom Filter for Multistream Synchronization Test
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnscustom_framecounter.c
 * @date  08 Nov 2018
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief  Custom filter that gives you the frame number as 1:1:1:1 uint32 tensor
 *
 * Input: ANY other/tensor or other/tensors
 * Output: 1:1:1:1 uint32 other/tensor
 *
 * If Input dimension is 1:1:1:1 uint32, single-tensor,
 * The output will copy the same value, ignoring the internal counter.
 *
 * This accepts custom properties from tensor_filter.
 * - custom=delay-300
 *     Adds 300ms of sleep for each invoke
 * - custom=stderr
 *     Do fprintf(stderr) of each frame invoke
 * - custom=stdout
 *     Do fprintf(stdout) of each frame invoke
 *
 * You may combine those with deliminator, ":"
 * - custom=stderr:delay-1000
 *     Do fprintf(stderr) and add 1000ms sleep for each invoke.
 *
 * @bug  No known bugs
 */


#include <tensor_filter_custom.h>
#include <stdio.h>
#include <assert.h>
#include <glib.h>
#include <string.h>
#include <stdlib.h>

static uint32_t maxid = 0;

/**
 * @brief nnstreamer custom filter private data for Frame Counter
 */
typedef struct _pt_data
{
  uint32_t id;
  uint32_t counter; /***< This counts the frame number from 0 */
  int copy; /***< Set 1 if input is 1:1:1:1 uint32 */
  int inputn; /***< # tensors in input */
  FILE *outf; /***< Where do you want to print? NULL if no print outs */
  uint32_t delay; /***< Delay per invoke in us */
} pt_data;

/**
 * @brief internal function to configure the filter from custom properties
 * @param[in] prop The properties of tensor_filter
 * @param[out] data The internal data structure of this custom filter.
 */
static void
configure (const GstTensorFilterProperties * prop, pt_data * data)
{
  if (prop->custom_properties != NULL) {
    gchar **str_ops;
    int counter = 0;

    str_ops = g_strsplit (prop->custom_properties, ":", -1);

    while (str_ops[counter]) {
      gchar *parsed = str_ops[counter];

      if (0 == strncmp (parsed, "stdout", 6)) {
        data->outf = stdout;
      } else if (0 == strncmp (parsed, "stderr", 6)) {
        data->outf = stderr;
      } else if (0 == strncmp (parsed, "delay-", 6)) {
        parsed = parsed + 6;
        if (*parsed)
          data->delay = (uint32_t) atoi (parsed);
      }

      counter++;
    }
    g_strfreev (str_ops);
  }
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));
  assert (data);
  maxid = maxid + 1;
  data->id = maxid;
  data->counter = 0U;
  data->copy = 0;
  data->inputn = 0;
  data->outf = NULL;
  data->delay = 0;
  configure (prop, data);
  return data;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void
pt_exit (void *_data, const GstTensorFilterProperties * prop)
{
  pt_data *data = _data;
  assert (data);
  if (data->outf)
    fprintf (data->outf, "[%u] %u\n", data->id, data->counter); /* The last counter value */
  free (data);
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
set_inputDim (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  int i;
  pt_data *data = _data;

  configure (prop, data);

  out_info->num_tensors = 1;
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    out_info->info[0].dimension[i] = 1;
  out_info->info[0].type = _NNS_UINT32;

  data->inputn = in_info->num_tensors;
  data->copy = 1;
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (in_info->info[0].dimension[i] != 1) {
      data->copy = 0;
      break;
    }
    if (in_info->info[0].type != _NNS_UINT32)
      data->copy = 0;
  }

  if (data->outf)
    fprintf (data->outf, "[%u] Returning dim.\n", data->id);
  return 0;
}

/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
invoke (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  pt_data *data = _data;
  uint32_t *counter = (uint32_t *) output[0].data;

  if (data->copy == 1) {
    uint32_t *incoming;
    incoming = (uint32_t *) input[0].data;
    *counter = *incoming;
  } else {
    *counter = data->counter;
  }
  data->counter = data->counter + 1;

  if (data->outf) {
    if (data->inputn > 1) {
      fprintf (data->outf, "[%u] Counter: %u / Output: %u <- %u\n", data->id, data->counter, *counter, *((uint32_t *) input[1].data));  /* The last counter value */
    } else {
      fprintf (data->outf, "[%u] Counter: %u / Output: %u\n", data->id, data->counter, *counter);       /* The last counter value */
    }
  }

  if (data->delay > 0)
    g_usleep (data->delay * 1000U);
  return 0;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = pt_init,
  .exitfunc = pt_exit,
  .setInputDim = set_inputDim,
  .invoke = invoke,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
