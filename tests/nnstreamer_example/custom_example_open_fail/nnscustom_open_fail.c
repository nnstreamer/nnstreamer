/**
 * NNStreamer custom filter to test the open error paths
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnscustom_open_fail.c
 * @date  18 Sep 2026
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief  Custom filters that tensor_filter refuses after it has loaded the library.
 * @bug  No known bugs
 *
 * This source is built once per error path of tensor_filter_custom. The build names
 * the variant in NNSCUSTOM_OPEN_FAIL_VARIANT and drops one callback with:
 * - NNSCUSTOM_OPEN_FAIL_NO_INIT: 'initfunc' is missing.
 * - NNSCUSTOM_OPEN_FAIL_NO_DIM: the dimension callbacks are missing.
 * - NNSCUSTOM_OPEN_FAIL_NO_INVOKE: the invoke callbacks are missing.
 * - NNSCUSTOM_OPEN_FAIL_BOTH_INVOKE: both invoke callbacks are given.
 * - NNSCUSTOM_OPEN_FAIL_NO_EXIT: 'exitfunc' is missing, which the sub-plugin accepts.
 * With none of them, it is a valid pass-through filter, so that a test case can tell
 * a refused open from a working one.
 *
 * 'nnscustom_open_fail_init_count_<variant>' and '..._exit_count_<variant>' count the
 * initfunc and exitfunc calls, so that a test case can tell whether a refused open has
 * released what initfunc returned. They are named after the variant because a custom
 * filter is loaded into the global symbol scope, where equally named symbols of the
 * variants would interpose each other.
 */

#include <stdlib.h>
#include <string.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#define OF_PASTE2(a, b) a##b
#define OF_PASTE(a, b) OF_PASTE2 (a, b)

#define OF_INIT_COUNT \
  OF_PASTE (nnscustom_open_fail_init_count_, NNSCUSTOM_OPEN_FAIL_VARIANT)
#define OF_EXIT_COUNT \
  OF_PASTE (nnscustom_open_fail_exit_count_, NNSCUSTOM_OPEN_FAIL_VARIANT)

unsigned int OF_INIT_COUNT = 0; /**< initfunc calls so far */
unsigned int OF_EXIT_COUNT = 0; /**< exitfunc calls so far */

#ifdef NNSCUSTOM_OPEN_FAIL_NO_EXIT
static int of_private_data; /**< what initfunc returns when nothing can release it */
#endif

#ifndef NNSCUSTOM_OPEN_FAIL_NO_INIT
/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void *
of_init (const GstTensorFilterProperties * prop)
{
  UNUSED (prop);

  OF_INIT_COUNT++;

#ifdef NNSCUSTOM_OPEN_FAIL_NO_EXIT
  /* Nothing would release a heap block: this variant has no exitfunc. */
  return &of_private_data;
#else
  return malloc (16);
#endif
}
#endif

#ifndef NNSCUSTOM_OPEN_FAIL_NO_EXIT
/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static void
of_exit (void *_data, const GstTensorFilterProperties * prop)
{
  UNUSED (prop);

  OF_EXIT_COUNT++;
  free (_data);
}
#endif

#ifndef NNSCUSTOM_OPEN_FAIL_NO_DIM
/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
of_setInputDim (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  UNUSED (_data);
  UNUSED (prop);

  gst_tensors_info_copy (out_info, in_info);
  return 0;
}
#endif

#ifndef NNSCUSTOM_OPEN_FAIL_NO_INVOKE
/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
of_invoke (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  unsigned int i;

  UNUSED (_data);

  for (i = 0; i < prop->output_meta.num_tensors; i++)
    memcpy (output[i].data, input[i].data, input[i].size);

  return 0;
}
#endif

#ifdef NNSCUSTOM_OPEN_FAIL_BOTH_INVOKE
/**
 * @brief nnstreamer custom filter standard vmethod
 * Refer tensor_filter_custom.h
 */
static int
of_allocate_invoke (void *_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  UNUSED (_data);
  UNUSED (prop);
  UNUSED (input);
  UNUSED (output);

  return 0;
}
#endif

static NNStreamer_custom_class NNStreamer_custom_body = {
#ifndef NNSCUSTOM_OPEN_FAIL_NO_INIT
  .initfunc = of_init,
#endif
#ifndef NNSCUSTOM_OPEN_FAIL_NO_EXIT
  .exitfunc = of_exit,
#endif
#ifndef NNSCUSTOM_OPEN_FAIL_NO_DIM
  .setInputDim = of_setInputDim,
#endif
#ifndef NNSCUSTOM_OPEN_FAIL_NO_INVOKE
  .invoke = of_invoke,
#endif
#ifdef NNSCUSTOM_OPEN_FAIL_BOTH_INVOKE
  .allocate_invoke = of_allocate_invoke,
#endif
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
