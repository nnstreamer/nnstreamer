/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 *
 * GStreamer Tensor_Filter (vivante Code)
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 * Copyright (C) 2019 Geunsik Lim <geunsik.lim@samsung.com>
 *
 */

/**
 * @file	tensor_filter_subplugin.c
 * @date	28 Jan 2020
 * @brief	NNStreamer tensor-filter subplugin template
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *       	Geunsik Lim <geunsik.lim@samsung.com>
 * @bug		No known bugs
 * @note    Example of Pipeline description.
 * const char description[] = "videotestsrc ! videoconvert ! videoscale ! "
 * "video/x-raw,format=RGB,width=299,height=299,framerate=(fraction)30/1 ! "
 * "tensor_converter ! "
 * "tensor_filter framework=vivante "
 *   "model=/opt/vivante/model/inception_v3.nb,/opt/vivante/model/libinception_v3.so"
 *   "input=3:299:299:1 inputtype=UINT8 inputname=data "
 *   "output=1001:1:1:1 outputtype=UINT16 outputname=prob "
 * "file_sink location=vivante.out.bin";
 *
 * @memo
 * 1. Common APIs for Vivante NPU
 * 1.1. Mandatory APIs:
 * vnn_CreateNeuralNetwork() to create a neural network from .nb file
 * vsi_nn_VerifyGraph() to verify the generated graph
 * vsi_nn_RunGraph() to process the graph
 * vsi_nn_GetTensor() to get meta data from input/output tensor
 * vsi_nn_CopyDataToTensor to copy the input buffer to the input tensor
 * vsi_nn_CopyTensorToBuffer() to copy the output tensor to the output buffer
 * vnn_ReleaseNeuralNetwork() to release the neural network
 * 1.2. Optional APIs:
 * vnn_PreProcessNeuralNetwork() to do a pre-process input data (e.g., image)
 * - Refer to https://github.com/nnsuite/nnstreamer/tree/master/gst/nnstreamer/tensor_transform
 * vnn_PostProcessNeuralNetwork() to display Top5 and save output tensor data
 * - The inceptionv3 model does not execute post-process tasks.
 * vsi_nn_DumpGraphNodeOutputs() to dump the graph for a debugging
 *
 * 2. Packaging, Build & run test, and Custom property
 * 2.1. Distribution: Tizen packaging
 * ubuntu$ git clone https://github.sec.samsung.net/AIP/vivante-nnstreamer-filter.git
 * ubuntu$ time gbs build -A armv7l --clean --include-all
 *
 * 2.2. Development: Build and run source code on real target
 * You may run the below script to build and run on the real target boards manually.
 * target# ./tests/nnstreamer_filter_vivante/build_run.sh
 *
 * 2.3 Custom properties for Vivante subplugin
 * CP = Exp,CP | Exp | NULL
 * Exp = postprocess | pp      # Both enables post-processing feature of Vivante if possible.
 *
 * Usage with Tizen C-API
 * Pipeline
 * ... ! tensor_filter framework=vivante model=a.nb,b.so custom=pp ! ...
 * Single
 * ... (open "handle")
 * ml_single_set_property (handle, "custom", "pp");
 * ... (invoke with "handle") x N
 * ... (close "handle")
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <glib.h>
#include <errno.h>

#include <nnstreamer_plugin_api_filter.h>

#include <dlfcn.h>

/** A neural network headers that are provided by the vivante/acuity tool. */
#include <ovx/vsi_nn_pub.h>

/** A macro for debugging all nodes of a graph */
#define DEBUG_MODE	0

/** A macro for verifying a corrcteness of the Vivante model (*.nb) */
#define EVAL_MODE	0

/** Fetch an address of a Vivante API with the dlsym() library call*/
#define vivante_api_fetch_dlsym(pdata, pname, symstr, errlabel) do { \
    char *error; \
    dlerror (); \
    g_assert (pdata->handle); \
    pdata->pname = dlsym(pdata->handle, symstr); \
    if ((error = dlerror()) != NULL) { \
      g_printerr("Cannot load %s: %s\n", symstr, error); \
      goto errlabel; \
    } \
} while (0)

/** Call a specified API among the Vivante APIs described in 'pdata' */
#define call(name, err, ...) ( (pdata->name) ? \
    pdata->name (__VA_ARGS__) : \
    err)

void init_filter_vivante (void) __attribute__ ((constructor));
void fini_filter_vivante (void) __attribute__ ((destructor));

/**
 * @brief If you need to store session or model data,
 *        this is what you want to fill in.
 */
typedef struct
{
  gchar *model_path; /**< The model nb file */
  gchar *so_path; /**< The so file for the .nb file */
  GstTensorsInfo input_tensor; /**< The description of input tensor */
  GstTensorsInfo output_tensor; /**< The description of output tensor */

  vsi_nn_graph_t *graph; /**< Graph data structure for handling Vivant neural network */
  void* handle; /**< Variables for handling the dlopen library call. */
  vsi_status (*result_vsi_nn_CopyDataToTensor)(vsi_nn_graph_t *, vsi_nn_tensor_t *, uint8_t *);
  void (*result_vnn_ReleaseNeuralNetwork)(vsi_nn_graph_t *);
  vsi_nn_graph_t * (*result_vnn_CreateNeuralNetwork)(const char *);
  vsi_status (*result_vsi_nn_RunGraph)(vsi_nn_graph_t *);

  int postProcess;
  vsi_status (*postProcessFunc)(vsi_nn_graph_t *graph);

#if EVAL_MODE
  vsi_status (*result_vnn_PostProcessNeuralNetwork)(vsi_nn_graph_t *);
#endif
#if DEBUG_MODE
  vsi_status (*result_vsi_nn_VerifyGraph)(vsi_nn_graph_t *);
  void (*result_nn_DumpGraphNodeOutputs)(vsi_nn_graph_t *, const char *, uint32_t *, uint32_t, vsi_bool, vsi_nn_dim_fmt_e);
#endif
} vivante_pdata;

/** @brief Parse a custom property and check if post-process is enabled */
static int
parseCustomProperty (const char *value, int * postProcess)
{
  int i = 0;
  gchar ** strv;

  /* default value */
  *postProcess = 0;

  if (!value || !value[0])
    return 0;

  strv = g_strsplit_set (value, ", :", -1);

  do {
    if (!g_ascii_strcasecmp (strv[i], "postprocess") ||
        !g_ascii_strcasecmp (strv[i], "pp")) {
      *postProcess = 1;
    }
    i++;
  } while (strv[i]);

  g_strfreev (strv);
  return 0;
}

/**
 * @brief Get post-processed output data
 * @details We assume that post-process does NOT alter output dimensions.
 */
static int
doPostProcess (vivante_pdata *pdata)
{
  vsi_status status;

  if (pdata->postProcess && pdata->postProcessFunc) {
    status = call (postProcessFunc, -EINVAL, pdata->graph);
    if (status == VSI_SUCCESS)
      return 0;
    return VSI_FAILURE;
  }
  return -EINVAL;
}

unsigned int
convert_tensortype (unsigned tensor_type);

static void
vivante_close (const GstTensorFilterProperties * prop,
    void **private_data);

/**
 *  * @brief Configure private_data
 *   */
static int
allocateData (const GstTensorFilterProperties * prop, void **private_data)
{
  vivante_pdata *pdata;

  if (*private_data != NULL) {
    /* Already opened */
    pdata = (vivante_pdata *) *private_data;

    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      printf("Model path (.nb) is not given.");
      return -1;
    }
    if (!prop->model_files[1] || prop->model_files[1][0] == '\0') {
      printf("Shared library path (.so) is not given.");
      return -1;
    }
    if (pdata->model_path && g_strcmp0 (prop->model_files[0],
          pdata->model_path) == 0) {
      return 0; /* Already opened with same model file. Skip ops */
    }
    if (pdata->so_path && g_strcmp0 (prop->model_files[1],
          pdata->so_path) == 0) {
      return 0; /* Already opened with same so file. Skip ops */
    }
    vivante_close (prop, private_data); /* Close before opening one. */
  }

  *private_data = pdata = g_new0 (vivante_pdata, 1);
  if (pdata == NULL) {
      printf("Failed to allocate memory for vivante tensor_filer.");
    return -1;
  }

  return 0;
}

/**
 * @brief Convert a tesor type from vivant format to nnstreamer format
 * Note that you must refer to the "linux_sdk/acuity-ovxlib-dev/include/vsi_nn_types.h" file.
 */
unsigned int
convert_tensortype (unsigned tensor_type)
{
  switch (tensor_type) {
  case VSI_NN_TYPE_INT8:
    return _NNS_INT8;
  case VSI_NN_TYPE_UINT8:
    return _NNS_UINT8;
  case VSI_NN_TYPE_INT16:
    return _NNS_INT16;
  case VSI_NN_TYPE_UINT16:
    return _NNS_UINT16;
  /** Note that the current nnstreamer (tensor_typedef.h) does not support FLOAT16.
   *  Let's use UINT16 as a workaround.
   */
  case VSI_NN_TYPE_FLOAT16:
    return _NNS_UINT16;
  case VSI_NN_TYPE_FLOAT32:
    return _NNS_FLOAT32;
  default:
    /** @todo Support other types */
    break;
  }
  return _NNS_END;
}

/**
 * @brief Open the vivante tensor filter
 * @note
 * 1. Read multi files (e.g., inception_v3.nb and libinception_v3.so)
 */
static int
vivante_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int ret = allocateData (prop, private_data);
  unsigned int i,j,k;
  vivante_pdata *pdata = (vivante_pdata *) *private_data;

  if (ret < 0)
    return ret;

  if (*private_data == NULL)
    return -ENOMEM;

  pdata->model_path = g_strdup (prop->model_files[0]);
  pdata->so_path    = g_strdup (prop->model_files[1]);

  ret = parseCustomProperty (prop->custom_properties, &pdata->postProcess);
  if (ret < 0) {
    g_free (pdata->model_path);
    g_free (pdata->so_path);
    g_free (pdata);
    return ret;
  }

  /** Create the neural network with .nb (a network binary of Vivante) */
  pdata->handle = dlopen(pdata->so_path, RTLD_NOW);
  if (!pdata->handle) {
    printf ("vivante_open: dlopen cannot load the shared library (.so).\n");
    g_free (pdata->model_path);
    g_free (pdata->so_path);
    g_free (pdata);
    return -EINVAL;
  }

  vivante_api_fetch_dlsym (pdata, result_vsi_nn_CopyDataToTensor,
      "vsi_nn_CopyDataToTensor", error_dlsym);
  vivante_api_fetch_dlsym (pdata, result_vnn_ReleaseNeuralNetwork,
      "vnn_ReleaseNeuralNetwork", error_dlsym);
  vivante_api_fetch_dlsym (pdata, result_vsi_nn_RunGraph,
      "vsi_nn_RunGraph", error_dlsym);

  if (pdata->postProcess) {
    vivante_api_fetch_dlsym (pdata, postProcessFunc,
        "vnn_PostProcessNeuralNetwork", error_dlsym);
  }

  vivante_api_fetch_dlsym (pdata, result_vnn_CreateNeuralNetwork,
      "vnn_CreateNeuralNetwork", error_dlsym);
  pdata->graph = call (result_vnn_CreateNeuralNetwork, NULL,
      pdata->model_path);

#if EVAL_MODE
  vivante_api_fetch_dlsym (pdata, result_vnn_PostProcessNeuralNetwork,
      "vnn_PostProcessNeuralNetwork", error_dlsym);
#endif
#if DEBUG_MODE
  vivante_api_fetch_dlsym (pdata, result_vsi_nn_VerifyGraph,
      "vsi_nn_VerifyGraph", error_dlsym);
  vivante_api_fetch_dlsym (pdata, result_nn_DumpGraphNodeOutputs,
      "vsi_nn_DumpGraphNodeOutputs", error_dlsym);
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"
#pragma GCC diagnostic ignored "-Wnested-externs"
  gst_tensors_info_init (&pdata->input_tensor);
  gst_tensors_info_init (&pdata->output_tensor);

  /** Note that we must use vsi_nn_GetTensor() to get a meta data
   * (e.g., input tensor and outout tensor).
   * ./linux_sdk/acuity-ovxlib-dev/lib/libovxlib.so
   * ./linux_sdk/acuity-ovxlib-dev/include/vsi_nn_graph.h
   * ./linux_sdk/acuity-ovxlib-dev/include/vsi_nn_tensor.h
   */

#if DEBUG_MODE
  printf ("[DEBUG] input_tensors_num :%d  \n", pdata->graph->input.num);
  printf ("[DEBUG] output_tensors_num:%d \n", pdata->graph->output.num);
#endif

  /** Get the meta data from the input tensor. */
  for (i = 0; i < pdata->graph->input.num; i++) {
    vsi_nn_tensor_t *i_tensor = vsi_nn_GetTensor(pdata->graph,
        pdata->graph->input.tensors[i]);
    if (i_tensor == NULL)
      return -1;

#if DEBUG_MODE
    printf ("[DEBUG] input_dim_num[%d]:%d\n", i, i_tensor->attr.dim_num);
#endif
    for (j = 0; j < i_tensor->attr.dim_num; ++j) {
      /** dimension structure: channel, width, height, number */
      pdata->input_tensor.info[i].dimension[j] = i_tensor->attr.size[j];
    }
    for (k = j; k < NNS_TENSOR_RANK_LIMIT; ++k) {
      pdata->input_tensor.info[i].dimension[k] = 1;
    }

    /** Get an input data type: VSI_NN_TYPE_UINT8 (u8) in case of inceptionv3 */
    pdata->input_tensor.info[i].type =
        convert_tensortype(i_tensor->attr.dtype.vx_type);
    asprintf (&pdata->input_tensor.info[i].name, "%i",
        pdata->graph->input.tensors[i]); /** dummy name */
    pdata->input_tensor.num_tensors = pdata->graph->input.num; /** number of tensors */
  }

  /** Get the meta data from the output tensor. */
  for (i = 0; i < pdata->graph->output.num; i++) {
    vsi_nn_tensor_t *o_tensor = NULL;
    o_tensor = vsi_nn_GetTensor(pdata->graph, pdata->graph->output.tensors[i]);
    if (o_tensor == NULL)
      return -1;

#if DEBUG_MODE
    printf ("[DEBUG] output_dim_num[%d]:%d\n", i, o_tensor->attr.dim_num);
#endif
    for (j = 0; j < o_tensor->attr.dim_num; ++j) {
      /** dimension structure: channel, width, height, number */
      pdata->output_tensor.info[i].dimension[j] = o_tensor->attr.size[j];
    }
    for (k = j; k < NNS_TENSOR_RANK_LIMIT; ++k) {
      pdata->output_tensor.info[i].dimension[k] = 1;
    }

    /** Get an output data type: VSI_NN_TYPE_FLOAT16 (f16) in case of inceptionv3 */
    pdata->output_tensor.info[i].type =
        convert_tensortype(o_tensor->attr.dtype.vx_type);
    asprintf (&pdata->output_tensor.info[i].name, "%i",
        pdata->graph->output.tensors[i]); /** dummy name */
    pdata->output_tensor.num_tensors = pdata->graph->output.num; /** number of tensors */
  }

#pragma GCC diagnostic pop
  return ret;
error_dlsym:
  dlclose (pdata->handle);
  pdata->handle = NULL;
  return -EINVAL;
}

/**
 * @brief Close the vivante tensor filter
 * @note Close what you have opened/allocated with vivante_open
 *  Release a buffer image.
 */
static void
vivante_close (const GstTensorFilterProperties * prop, void **private_data)
{
  vivante_pdata *pdata = *private_data;

  call(result_vnn_ReleaseNeuralNetwork, NULL, pdata->graph);

  dlclose(pdata->handle);

  g_free (pdata->model_path);
  pdata->model_path = NULL;
  g_free (pdata->so_path);
  pdata->so_path = NULL;

  g_free (pdata);
  *private_data = NULL;
}

/**
 * @brief The standard tensor_filter callback
 * @note Call your framework/hardware with the given input
 *  1. Do a pre-process for the image data (Skipped)
 *  2. Verify the model data (graph).
 *  3. Process a network model (graph)
 *  4. Optionally, dump all node outputs (e.g., ./network_dump) for a debugging.
 *  5. Do prost-process to output data.
 */
static int
vivante_invoke (const GstTensorFilterProperties * prop,
    void **private_data, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  int ret = 0;
  int i;
  vivante_pdata *pdata = *private_data;
  vsi_status status = VSI_FAILURE;

  g_assert (*private_data);

#if 1
  /**
   * In order to develop a NNStreamer application, Please utilize the 'tensor_transform" element.
   * Copy input data to a tensor using vsi_nn_CopyDataToTensor instead of
   * vnn_PreProcessNeuralNetwork() to finalize a pre-process of input data (e.g., image)
   * Note that the vnn_PreProcessNeuralNetwork API requires jpeg-9a library by default.
   */
  for (i = 0; i < pdata->graph->input.num; i++) {
    vsi_nn_tensor_t *tensor = NULL;
    tensor = vsi_nn_GetTensor(pdata->graph, pdata->graph->input.tensors[i]);

   /** Copy an input buffer to an input tensor */
    status = call(result_vsi_nn_CopyDataToTensor, VSI_FAILURE, pdata->graph,
        tensor, input[i].data);
  }
#endif

#if DEBUG_MODE
  /**
   * Verify a graph. Note that vnn_VerifyGraph() calls vsi_nn_VerifyGraph()
   * that is provided by OVXLIB_API.
   */
  status = call (result_vsi_nn_VerifyGraph, VSI_FAILURE, pdata->graph);
#endif

  /**
   * Process a graph. Note that vnn_ProcessGraph() calls vsi_nn_RunGraph()
   * that is provided by OVXLIB_API.
   */
  status = call (result_vsi_nn_RunGraph, VSI_FAILURE, pdata->graph);


#if DEBUG_MODE
  /** Dump all node outputs */
  g_print("Saving debug file (e.g., ./network_dump)\n");

  call (result_nn_DumpGraphNodeOutputs, NULL, pdata->graph, "./network_dump", NULL, 0, TRUE, 0);
#endif

  if (pdata->postProcess)
    ret = doPostProcess (pdata);
  if (ret < 0) {
    g_printerr("PostProcess Failure\n");
    return ret;
  }

#if EVAL_MODE
  /** In case of Inceptionv3, the major goal of of the post-process is as follows.
   *  a. Show the Top5 result
   *  b. Save all output tensor data to txt file.
   */
  status = call (result_vnn_PostProcessNeuralNetwork, VSI_FAILURE,
      pdata->graph);
#endif

  #define _DUMP_FILE_LENGTH 1028
  #define _DUMP_SHAPE_LENGTH 128

  for (i = 0; i < pdata->graph->output.num; i++) {
    vsi_nn_tensor_t *out_tensor = vsi_nn_GetTensor(pdata->graph, pdata->graph->output.tensors[i]);

#if DEBUG_MODE
    char filename[_DUMP_FILE_LENGTH] = {0};
    char shape[_DUMP_SHAPE_LENGTH] = {0};
    vsi_nn_ShapeToString(out_tensor->attr.size, out_tensor->attr.dim_num,
        shape, _DUMP_SHAPE_LENGTH, FALSE );
    snprintf(filename, _DUMP_FILE_LENGTH, "nnstreamer_output%u_%s.dat", i, shape);
    vsi_nn_SaveTensorToBinary(pdata->graph, out_tensor, filename);
#endif

    /** Copy an output tensor to an output buffer */
    vsi_nn_CopyTensorToBuffer(pdata->graph, out_tensor, output[i].data);
  }
  if (status == VSI_FAILURE)
    return -EINVAL;

  return 0;
}


/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
vivante_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  vivante_pdata *pdata = (vivante_pdata *) *private_data;

  if (!pdata)
    return -1;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"
#pragma GCC diagnostic ignored "-Wnested-externs"
  gst_tensors_info_copy (info, &pdata->input_tensor);
#pragma GCC diagnostic pop
  return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
vivante_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  vivante_pdata *pdata = (vivante_pdata *) *private_data;

  if (!pdata)
    return -1;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"
#pragma GCC diagnostic ignored "-Wnested-externs"
  gst_tensors_info_copy (info, &pdata->output_tensor);
#pragma GCC diagnostic pop

  return 0;
}

static gchar filter_subplugin_vivante[] = "vivante";

static GstTensorFilterFramework NNS_support_vivante = {
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
#endif
  .open = vivante_open,
  .close = vivante_close,
};

/**@brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_vivante (void)
{
  NNS_support_vivante.name = filter_subplugin_vivante;
  NNS_support_vivante.allow_in_place = FALSE;
  NNS_support_vivante.allocate_in_invoke = FALSE;
  NNS_support_vivante.run_without_model = FALSE;
  NNS_support_vivante.invoke_NN = vivante_invoke;
  NNS_support_vivante.getInputDimension = vivante_getInputDim;
  NNS_support_vivante.getOutputDimension = vivante_getOutputDim;
  nnstreamer_filter_probe (&NNS_support_vivante);
}

/** @brief Destruct the subplugin */
void
fini_filter_vivante (void)
{
  nnstreamer_filter_exit (NNS_support_vivante.name);
}
