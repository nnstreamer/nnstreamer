/**
 * A Tensor Filter Extension for NCSDK Ver.2 (Intel Movidius Neural Compute Stick)
 * Copyright (C) 2019 Wook Song <wook16.song@samsung.com>
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
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
 * @file    tensor_filter_movidius_ncsdk2.c
 * @date    13 May 2019
 * @brief   NCSDK2 module for tensor_filter gstreamer plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Wook16.song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (Intel Movidius NCSDK2) for tensor_filter.
 */

#include <fcntl.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <mvnc2/mvnc.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_filter.h>
#include <sys/types.h>
#include <sys/stat.h>

static const gchar *mvncsdk2_accl_support[] = {
  ACCL_AUTO_STR,
  ACCL_DEFAULT_STR,
  ACCL_NPU_STR,
  ACCL_NPU_MOVIDIUS_STR,
  NULL
};

void init_filter_mvncsdk2 (void) __attribute__ ((constructor));
void fini_filter_mvncsdk2 (void) __attribute__ ((destructor));

enum _private_constants
{
  NNS_MVNCSDK2_SUPPORT_MAX_NUMS_DEVICES = 8,
  NNS_MVNCSDK2_SUPPORT_API_MAJOR_VER = 2,
  NNS_MVNCSDK2_API_VER_ARRAY_SIZE = NC_VERSION_MAX_SIZE,
  NNS_MVNCSDK2_MAX_NUM_ELEM_IN_FIFO = 2,
  NNS_MVNCSDK2_MAX_NUM_TENOSORS_SUPPORTED = 1,
};

static const char NNS_MVNCSDK2_NAME_INPUT_FIFO[] = "INPUT_FIFO";
static const char NNS_MVNCSDK2_NAME_OUTPUT_FIFO[] = "OUTPUT_FIFO";

/**
 * @brief internal data of mvncsdk2
 */
typedef struct _mvncsdk2_data
{
  /* Variables of the data types from mvnc.h */
  struct ncDeviceHandle_t *handle_device; /** handle for Intel Movidius device */
  struct ncGraphHandle_t *handle_graph; /** handle for graph (model) */
  struct ncTensorDescriptor_t tensor_desc_input; /** description of input tensor */
  struct ncTensorDescriptor_t tensor_desc_output; /** description of output tenser */
  struct ncFifoHandle_t *handle_fifo_input; /** handle for input fifo (buffer) */
  struct ncFifoHandle_t *handle_fifo_output; /** handle for output fifo (buffer) */
  /* Normal variables */
  gint32 idx_device;  /** index of device to use (Q. is it necessary?) */
} mvncsdk2_data;

/**
 * @brief Free privateData and move on.
 * @param prop : property of tensor_filter instance
 * @param private_data : movidius-ncsdk2 plugin's private data
 */
static void
_mvncsdk2_close (const GstTensorFilterProperties * prop, void **private_data)
{
  mvncsdk2_data *pdata = *private_data;

  if (pdata != NULL) {
    ncGraphDestroy (&(pdata->handle_graph));
    ncFifoDestroy (&(pdata->handle_fifo_output));
    ncFifoDestroy (&(pdata->handle_fifo_input));
    ncDeviceDestroy (&(pdata->handle_device));

    g_free (pdata);
    *private_data = NULL;
  }
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop : property of tensor_filter instance
 * @param private_data : movidius-ncsdk2 plugin's private data
 * @return 0 if OK. -1 if error.
 */
static int
_mvncsdk2_open (const GstTensorFilterProperties * prop, void **private_data)
{
  /* Variables of the data types from mvnc.h */
  struct ncDeviceHandle_t *handle_device = NULL;
  struct ncGraphHandle_t *handle_graph = NULL;
  struct ncTensorDescriptor_t tensor_desc_input;
  struct ncTensorDescriptor_t tensor_desc_output;
  struct ncFifoHandle_t *handle_fifo_input;
  struct ncFifoHandle_t *handle_fifo_output;
  ncStatus_t ret_code;
  /* Normal variables */
  GMappedFile *file_model = NULL;
  mvncsdk2_data *pdata = NULL;
  gint32 sdk_ver[NNS_MVNCSDK2_API_VER_ARRAY_SIZE];
  guint32 size_sdk_ver = sizeof (sdk_ver);
  gint32 idx_dev = -1;
  guint32 len_model_file;
  void *buf_model_file;
  guint32 len;
  gint32 i;

  /* 0. Check the API version */
  ret_code = ncGlobalGetOption (NC_RO_API_VERSION, sdk_ver, &size_sdk_ver);
  if ((ret_code == NC_OK) && (sdk_ver[0] != NNS_MVNCSDK2_SUPPORT_API_MAJOR_VER)) {
    g_printerr
        ("The major version number of the MVNCSDK API should be %d, not %d\n",
        NNS_MVNCSDK2_SUPPORT_API_MAJOR_VER, sdk_ver[0]);
    return -1;
  } else if (ret_code != NC_OK) {
    g_printerr
        ("Failed to get the information about the version of the MVNCSDK API\n");
    return -1;
  }

  /**
   * 1. Initialize device handle:
   *  we do not know how many devices are plugged and used. Therefore,
   *  let's try all the possible device indices (currently,
   *  0 ~ NNS_MVNCSDK2_SUPPORT_MAX_NUMS_DEVICES) here.
   */
  for (i = 0; i < NNS_MVNCSDK2_SUPPORT_MAX_NUMS_DEVICES; ++i) {
    ret_code = ncDeviceCreate (i, &handle_device);
    if (ret_code == NC_OK) {
      idx_dev = i;
      break;
    } else {
      GST_WARNING ("Failed to create device handle at index %d: "
          "%d is returned\n", i, ret_code);
    }
  }
  if ((ret_code != NC_OK) && (i == NNS_MVNCSDK2_SUPPORT_MAX_NUMS_DEVICES)) {
    g_printerr ("Cannot create device handle: no available device found\n");
    return -1;
  }

  /**
   * 2. Initialize graph (model) handle
   */
  ret_code = ncGraphCreate (prop->model_files[0], &handle_graph);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot create graph handle for \"%s\"\n",
        prop->model_files[0]);
    goto err_destroy_device_h;
  }

  /**
   * 3. Open device using device handle
   */
  ret_code = ncDeviceOpen (handle_device);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot open device at index %d\n", idx_dev);
    goto err_destroy_graph_h;
  }

  /**
   * 4. Send model (graph) to the device
   */
  file_model = g_mapped_file_new (prop->model_files[0], FALSE, NULL);
  if (file_model == NULL) {
    g_printerr ("Failed to g_mapped_file_new for the model file, \"%s\"\n",
        prop->model_files[0]);
    goto err_destroy_graph_h;
  }

  /* Warning: conversion unsigned long to unsigned int */
  len_model_file = g_mapped_file_get_length (file_model);
  buf_model_file = (void *) g_mapped_file_get_contents (file_model);

  /* Actually, send the model to the device */
  ret_code = ncGraphAllocate (handle_device, handle_graph, buf_model_file,
      len_model_file);
  /** After allocating, we do not need file_model any more */
  g_mapped_file_unref (file_model);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot send the model file to the device\n");
    goto err_destroy_graph_h;
  }

  /**
   * 5. Get the tensor desciptions for input and output form allocated model
   */
  len = sizeof (tensor_desc_input);
  ret_code =
      ncGraphGetOption (handle_graph, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS,
      &tensor_desc_input, &len);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot get the tensor description for input\n");
    goto err_destroy_graph_h;
  }

  len = sizeof (tensor_desc_output);
  ret_code =
      ncGraphGetOption (handle_graph, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS,
      &tensor_desc_output, &len);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot get the tensor description for output\n");
    goto err_destroy_graph_h;
  }
  /**
   * 6. Create fifo handles for input and output tensors
   */
  ret_code = ncFifoCreate (NNS_MVNCSDK2_NAME_INPUT_FIFO,
      NC_FIFO_HOST_WO, &handle_fifo_input);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot create FIFO handle for input tensor\n");
    goto err_destroy_graph_h;
  }

  ret_code = ncFifoCreate (NNS_MVNCSDK2_NAME_OUTPUT_FIFO,
      NC_FIFO_HOST_RO, &handle_fifo_output);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot create FIFO handle for output tensor\n");
    goto err_destroy_graph_h;
  }

  /**
   * 7. Allocate fifos for input and output tensors
   */
  ret_code = ncFifoAllocate (handle_fifo_input, handle_device,
      &tensor_desc_input, NNS_MVNCSDK2_MAX_NUM_ELEM_IN_FIFO);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot allocate FIFO in the device for input tensor\n");
    goto err_destroy_graph_h;
  }

  ret_code = ncFifoAllocate (handle_fifo_output, handle_device,
      &tensor_desc_output, NNS_MVNCSDK2_MAX_NUM_ELEM_IN_FIFO);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot allocate FIFO in the device for output tensor\n");
    ncFifoDestroy (&handle_fifo_input);
    goto err_destroy_graph_h;
  }

  /**
   * 8. Create private data and fill it
   */
  pdata = g_try_new0 (mvncsdk2_data, 1);
  if (pdata == NULL) {
    g_printerr ("Cannot allocate memory for private data structure\n");
    goto err_destroy_fifo_h;
  }

  pdata->handle_device = handle_device;
  pdata->handle_graph = handle_graph;
  pdata->handle_fifo_input = handle_fifo_input;
  pdata->handle_fifo_output = handle_fifo_output;
  pdata->tensor_desc_input = tensor_desc_input;
  pdata->tensor_desc_output = tensor_desc_output;
  *private_data = pdata;

  return 0;

err_destroy_fifo_h:
  ncFifoDestroy (&handle_fifo_input);
  ncFifoDestroy (&handle_fifo_output);
err_destroy_graph_h:
  ncGraphDestroy (&handle_graph);
err_destroy_device_h:
  ncDeviceDestroy (&handle_device);

  g_printerr ("Failed to initialize %s tensor_filter framework", prop->fwname);

  return -1;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop : property of tensor_filter instance
 * @param private_data : movidius-ncsdk2 plugin's private data
 * @param[in] input : The array of input tensors
 * @param[out] output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
_mvncsdk2_invoke (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  ncStatus_t ret_code;
  mvncsdk2_data *pdata = *private_data;
  guint32 buf_size;

  g_return_val_if_fail (prop->input_configured, -1);
  if (prop->input_meta.num_tensors != NNS_MVNCSDK2_MAX_NUM_TENOSORS_SUPPORTED) {
    ml_loge ("The number of input tensor should be one: "
        "The MVNCSDK API supports single tensor input and output only");
    goto err_destroy;
  }

  /* Warning: conversion unsigned long to unsigned int */
  buf_size = (guint32) input->size;
  ret_code = ncFifoWriteElem (pdata->handle_fifo_input, input->data,
      &buf_size, 0);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot write input data to the FIFO buffer in the device");
    goto err_destroy;
  }

  ret_code = ncGraphQueueInference (pdata->handle_graph,
      &(pdata->handle_fifo_input), NNS_MVNCSDK2_MAX_NUM_TENOSORS_SUPPORTED,
      &(pdata->handle_fifo_output), NNS_MVNCSDK2_MAX_NUM_TENOSORS_SUPPORTED);
  if (ret_code != NC_OK) {
    g_printerr ("Failed to run inference using the device\n");
    goto err_destroy;
  }

  /* Warning: conversion unsigned long to unsigned int */
  buf_size = (guint32) output->size;
  ret_code = ncFifoReadElem (pdata->handle_fifo_output, output->data, &buf_size,
      NULL);
  if (ret_code != NC_OK) {
    g_printerr ("Cannot fetch inference results from the device\n");
    goto err_destroy;
  }

  return 0;

err_destroy:
  /**
   * When this invoke callback is returned with -1, the whole pipeline is
   * immediately terminated by g_assert() without any unref() or free()
   * invocations. Until we fix this issue, the invoke callback calls the close()
   * itself before returning.
   */
  _mvncsdk2_close (prop, private_data);

  g_printerr ("Failed to call the invoke callback for the tensor_filter"
      "framework, %s", prop->fwname);

  return -1;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop : property of tensor_filter instance
 * @param private_data : movidius-ncsdk2 plugin's private data
 * @param[out] info : The dimesions and types of input tensors
 */
static int
_mvncsdk2_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  mvncsdk2_data *pdata = *private_data;
  struct ncTensorDescriptor_t *nc_input_desc = &(pdata->tensor_desc_input);
  GstTensorInfo *nns_input_tensor_info;

  /** MVNCSDK only supports one tensor at a time */
  info->num_tensors = NNS_MVNCSDK2_MAX_NUM_TENOSORS_SUPPORTED;
  nns_input_tensor_info =
      &(info->info[NNS_MVNCSDK2_MAX_NUM_TENOSORS_SUPPORTED - 1]);
  /**
   * MVNCSDK only supports data types of FP32 and FP16. If the data type of
   * input tensor is set to FP32, NCSDK automatically convert it to FP16 as
   * needed
   */
  nns_input_tensor_info->type = _NNS_FLOAT32;

  nns_input_tensor_info->dimension[0] = nc_input_desc->c;
  nns_input_tensor_info->dimension[1] = nc_input_desc->w;
  nns_input_tensor_info->dimension[2] = nc_input_desc->h;
  nns_input_tensor_info->dimension[3] = nc_input_desc->n;

  return 0;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop : property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info : The dimesions and types of output tensors
 */
static int
_mvncsdk2_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  mvncsdk2_data *pdata = *private_data;
  struct ncTensorDescriptor_t *nc_output_desc = &(pdata->tensor_desc_output);
  GstTensorInfo *nns_output_info;

  /** MVNCSDK only supports one tensor at a time */
  info->num_tensors = NNS_MVNCSDK2_MAX_NUM_TENOSORS_SUPPORTED;
  nns_output_info = &(info->info[NNS_MVNCSDK2_MAX_NUM_TENOSORS_SUPPORTED - 1]);
  /**
   * MVNCSDK only supports data types of FP32 and FP16. If the data type of
   * input tensor is set to FP32, NCSDK automatically convert it to FP16 as
   * needed
   */
  nns_output_info->type = _NNS_FLOAT32;

  nns_output_info->dimension[0] = nc_output_desc->c;
  nns_output_info->dimension[1] = nc_output_desc->w;
  nns_output_info->dimension[2] = nc_output_desc->h;
  nns_output_info->dimension[3] = nc_output_desc->n;

  return 0;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] hw backend accelerator hardware
 * @return 0 if supported. -errno if not supported.
 */
static int
_mvncsdk2_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (mvncsdk2_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}

static gchar filter_subplugin_movidius_ncsdk2[] = "movidius-ncsdk2";

static GstTensorFilterFramework NNS_support_movidius_ncsdk2 = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = _mvncsdk2_open,
  .close = _mvncsdk2_close,
  .checkAvailability = _mvncsdk2_checkAvailability,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_mvncsdk2 (void)
{
  NNS_support_movidius_ncsdk2.name = filter_subplugin_movidius_ncsdk2;
  NNS_support_movidius_ncsdk2.allow_in_place = FALSE;
  NNS_support_movidius_ncsdk2.allocate_in_invoke = FALSE;
  NNS_support_movidius_ncsdk2.verify_model_path = FALSE;
  NNS_support_movidius_ncsdk2.invoke_NN = _mvncsdk2_invoke;
  NNS_support_movidius_ncsdk2.getInputDimension = _mvncsdk2_getInputDim;
  NNS_support_movidius_ncsdk2.getOutputDimension = _mvncsdk2_getOutputDim;

  nnstreamer_filter_probe (&NNS_support_movidius_ncsdk2);
}

/** @brief Destruct the subplugin */
void
fini_filter_mvncsdk2 (void)
{
  nnstreamer_filter_exit (NNS_support_movidius_ncsdk2.name);
}
