/**
 * GStreamer Tensor_Filter, Tensorflow-Lite Module
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_filter_tensorflow_lite.c
 * @date	24 May 2018
 * @brief	Tensorflow-lite module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow-lite) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include "tensor_filter.h"
#include "tensor_filter_tensorflow_lite_core.h"
#include <glib.h>
#include <string.h>

/**
 * @brief custom option enumeration for tensorflow-lite.
 */
enum
{
  TFLITE_OPTION_NNAPI = 0,
  TFLITE_OPTION_MAX
};

/**
 * @brief custom option string for tensorflow-lite.
 */
const char *tflite_options[TFLITE_OPTION_MAX] = {
  "NNAPI",
};

/**
 * @brief delimiter characters.
 */
static const char *tokens = "=:,";

/**
 * @brief a pointer to custom option string.
 * @note the rule should be custom=OPTION1:VALUE1,...,OPTIONn:VALUEn
 */
static char *custom_options;

/**
 * @brief internal data of tensorflow lite
 */
struct _Tflite_data
{
  void *tflite_private_data;
};
typedef struct _Tflite_data tflite_data;

/**
  * @brief parse custom options for tensorflow lite
  * @param filter : tensor_filter instance
  * @param id : Option ID
  * @return 0 or 1 if successfully parsed.
  *        -1 if parsing a value of a given option failed.
  */
static int
tflite_get_parser_boolean (const GstTensorFilter * filter, int id)
{
  int ret = -1;

  if (filter->prop.custom_properties == NULL)
    return ret;

  if (id < 0 || id >= TFLITE_OPTION_MAX) {
    GST_WARNING ("Invalid option id. (%d)\n", id);
    return ret;
  }

  custom_options = g_strdup (filter->prop.custom_properties);
  if (custom_options == NULL) {
    GST_WARNING ("failed to dump custom property string\n");
    return ret;
  }

  char *str = strtok (custom_options, tokens);
  while (str != NULL) {
    if (strcmp (str, tflite_options[id]) == 0) {
      str = strtok (NULL, tokens);
      break;
    }

    /** TODO. check other options here if required. */

    str = strtok (NULL, tokens);
    if (str == NULL)
      break;

    str = strtok (NULL, tokens);
  }

  if (str == NULL) {
    GST_WARNING ("Invalid custom option.\n");
    goto out;
  }

  switch (id) {
    case TFLITE_OPTION_NNAPI:
      if (strcmp (str, "TRUE") == 0)
        ret = 1;
      else if (strcmp (str, "FALSE") == 0)
        ret = 0;
      else
        GST_WARNING ("Invalid custom value. (%s)\n", str);
      break;
  /** TODO. add additional cases if required. */
    default:
      break;
  }

out:
  free (custom_options);
  return ret;
}

/**
 * @brief Free privateData and move on.
 */
static void
tflite_close (const GstTensorFilter * filter, void **private_data)
{
  tflite_data *tf;
  tf = *private_data;
  tflite_core_delete (tf->tflite_private_data);
  g_free (tf);
  *private_data = NULL;
  g_assert (filter->privateData == NULL);
}

/**
 * @brief Load tensorflow lite modelfile
 * @param filter : tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
tflite_loadModelFile (const GstTensorFilter * filter, void **private_data)
{
  tflite_data *tf;
  if (filter->privateData != NULL) {
    /** @todo : Check the integrity of filter->data and filter->model_file, nnfw */
    tf = *private_data;
    if (strcmp (filter->prop.model_file,
            tflite_core_getModelPath (tf->tflite_private_data))) {
      tflite_close (filter, private_data);
    } else {
      return 1;
    }
  }
  tf = g_new0 (tflite_data, 1); /** initialize tf Fill Zero! */
  *private_data = tf;

  int use_nnapi = tflite_get_parser_boolean (filter, TFLITE_OPTION_NNAPI);
  if (use_nnapi < 0) {
    GST_INFO ("The supported option is custom=NNAPI:[TRUE|FALSE].\n");
    GST_INFO ("In default, it uses CPU fallback path instead of NNAPI.\n");
    use_nnapi = 0;
  }

  GST_LOG ("NNAPI=%s\n", use_nnapi ? "TRUE" : "FALSE");

  tf->tflite_private_data =
      tflite_core_new (filter->prop.model_file, use_nnapi);
  if (tf->tflite_private_data) {
    if (tflite_core_init (tf->tflite_private_data))
      return -2;
    return 0;
  } else {
    return -1;
  }
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param filter : tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 */
static int
tflite_open (const GstTensorFilter * filter, void **private_data)
{
  return tflite_loadModelFile (filter, private_data);
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
tflite_invoke (const GstTensorFilter * filter, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  int retval;
  tflite_data *tf;
  tf = *private_data;
  g_assert (filter->privateData && *private_data == filter->privateData);
  retval = tflite_core_invoke (tf->tflite_private_data, input, output);
  g_assert (retval == 0);
  return retval;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static int
tflite_getInputDim (const GstTensorFilter * filter, void **private_data,
    GstTensorsInfo * info)
{
  tflite_data *tf;
  tf = *private_data;
  g_assert (filter->privateData && *private_data == filter->privateData);
  int ret = tflite_core_getInputDim (tf->tflite_private_data, info);
  return ret;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static int
tflite_getOutputDim (const GstTensorFilter * filter, void **private_data,
    GstTensorsInfo * info)
{
  tflite_data *tf;
  tf = *private_data;
  g_assert (filter->privateData && *private_data == filter->privateData);
  int ret = tflite_core_getOutputDim (tf->tflite_private_data, info);
  return ret;
}

GstTensorFilterFramework NNS_support_tensorflow_lite = {
  .name = "tensorflow-lite",
  .allow_in_place = FALSE,      /** @todo: support this to optimize performance later. */
  .allocate_in_invoke = FALSE,
  .invoke_NN = tflite_invoke,
  .getInputDimension = tflite_getInputDim,
  .getOutputDimension = tflite_getOutputDim,
  .open = tflite_open,
  .close = tflite_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
__attribute__ ((constructor))
     void init_filter_tflite (void)
{
  tensor_filter_probe (&NNS_support_tensorflow_lite);
}

/** @brief Destruct the subplugin */
__attribute__ ((destructor))
     void fini_filter_tflite (void)
{
  tensor_filter_exit (NNS_support_tensorflow_lite.name);
}
