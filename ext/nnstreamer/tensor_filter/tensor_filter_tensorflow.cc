/**
 * GStreamer Tensor_Filter, Tensorflow Module
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2018 HyoungJoo Ahn <hello.ahn@samsung.com>
 * Copyright (C) 2018 Jijoong Moon <jjioong.moon@samsung.com>
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
 * @file   tensor_filter_tensorflow.cc
 * @date   02 Aug 2018
 * @brief  Tensorflow module for tensor_filter gstreamer plugin
 * @see    http://github.com/nnsuite/nnstreamer
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow) for tensor_filter.
 */

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>

#include <tensorflow/c/c_api.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief	Internal data structure for tensorflow
 */
typedef struct
{
  TF_DataType type;
  int rank;
  std::vector < std::int64_t > dims;
} tf_tensor_info_s;

/**
 * @brief	ring cache structure
 */
class TFCore
{
public:
  /**
   * member functions.
   */
  TFCore (const char * _model_path);
   ~TFCore ();

  int init (const GstTensorFilterProperties * prop);
  int loadModel ();
  const char *getModelPath ();

  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int run (const GstTensorMemory * input, GstTensorMemory * output);

  static std::map < void *, TF_Tensor * >outputTensorMap;

private:

  char *model_path;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors from user input */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors from user input */

  std::vector < tf_tensor_info_s > input_tensor_info; /* hold information for TF */

  TF_Graph *graph;
  TF_Session *session;

  tensor_type getTensorTypeFromTF (TF_DataType tfType);
  TF_DataType getTensorTypeToTF (tensor_type tType);
  int validateTensor (const GstTensorsInfo * tensorInfo, int is_input);
};

void init_filter_tf (void) __attribute__ ((constructor));
void fini_filter_tf (void) __attribute__ ((destructor));

std::map <void*, TF_Tensor*> TFCore::outputTensorMap;

/**
 * @brief	TFCore creator
 * @param	_model_path	: the logical path to '{model_name}.pb' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFCore::TFCore (const char *_model_path)
{
  g_assert (_model_path != NULL);
  model_path = g_strdup (_model_path);
  graph = nullptr;
  session = nullptr;

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
}

/**
 * @brief	TFCore Destructor
 * @return	Nothing
 */
TFCore::~TFCore ()
{
  if (graph != nullptr)
    TF_DeleteGraph (graph);

  if (session != nullptr) {
    TF_Status* status = TF_NewStatus ();

    TF_CloseSession (session, status);
    if (TF_GetCode (status) != TF_OK) {
      g_critical ("Error during session close!! - [Code: %d] %s",
        TF_GetCode (status), TF_Message (status));
    }

    TF_DeleteSession (session, status);
    if (TF_GetCode (status) != TF_OK) {
      g_critical ("Error during session delete!! - [Code: %d] %s",
        TF_GetCode (status), TF_Message (status));
    }
    TF_DeleteStatus (status);
  }

  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
  g_free (model_path);
}

/**
 * @brief	initialize the object with tensorflow model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initialization of output tensor is failed.
 */
int
TFCore::init (const GstTensorFilterProperties * prop)
{
  if (loadModel ()) {
    g_critical ("Failed to load model");
    return -1;
  }

  if (validateTensor (&prop->input_meta, 1)) {
    g_critical ("Failed to validate input tensor");
    return -2;
  }

  if (validateTensor (&prop->output_meta, 0)) {
    g_critical ("Failed to validate output tensor");
    return -3;
  }

  gst_tensors_info_copy (&inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&outputTensorMeta, &prop->output_meta);

  return 0;
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char *
TFCore::getModelPath ()
{
  return model_path;
}

/**
 * @brief	the definition of a deallocator method
 */
static void
DeallocateBuffer (void* data, size_t t) {
  std::free (data);
}

/**
 * @brief	load the tf model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 *        -1 if the pb file is not regular.
 *        -2 if the pb file is not loaded.
 *        -3 if importing graph is failed.
 *        -4 if the Tensorflow session is not created.
 */
int
TFCore::loadModel ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  gsize file_size;
  gchar *content = nullptr;
  GError *file_error = nullptr;

  g_assert (model_path != nullptr);

  if (!g_file_test (model_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("the file of model_path (%s) is not valid (not regular)\n", model_path);
    return -1;
  }

  if (!g_file_get_contents (model_path, &content, &file_size, &file_error)) {
    g_critical ("Error reading model file!! - %s", file_error->message);
    g_clear_error (&file_error);
    return -2;
  }

  TF_Buffer* buffer = TF_NewBuffer ();
  buffer->data = content;
  buffer->length = file_size;
  buffer->data_deallocator = DeallocateBuffer;

  graph = TF_NewGraph ();
  g_assert (graph != nullptr);

  TF_Status* status = TF_NewStatus ();
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions ();

  TF_GraphImportGraphDef (graph, buffer, opts, status);
  TF_DeleteImportGraphDefOptions (opts);
  TF_DeleteBuffer (buffer);

  if (TF_GetCode (status) != TF_OK) {
    g_critical ("Error deleting graph!! - [Code: %d] %s",
      TF_GetCode (status), TF_Message (status));
    TF_DeleteStatus (status);
    TF_DeleteGraph (graph);
    return -3;
  }

  TF_SessionOptions* options = TF_NewSessionOptions ();
  session = TF_NewSession (graph, options, status);
  TF_DeleteSessionOptions (options);

  if (TF_GetCode (status) != TF_OK) {
    g_critical ("Error creating Session!! - [Code: %d] %s",
      TF_GetCode (status), TF_Message (status));
    TF_DeleteStatus (status);
    TF_DeleteGraph (graph);
    return -4;
  }
  TF_DeleteStatus (status);

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tfType	: the defined type of Tensorflow
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
TFCore::getTensorTypeFromTF (TF_DataType tfType)
{
  switch (tfType) {
    case TF_INT32:
      return _NNS_INT32;
    case TF_UINT32:
      return _NNS_UINT32;
    case TF_INT16:
      return _NNS_INT16;
    case TF_UINT16:
      return _NNS_UINT16;
    case TF_INT8:
      return _NNS_INT8;
    case TF_UINT8:
      return _NNS_UINT8;
    case TF_INT64:
      return _NNS_INT64;
    case TF_UINT64:
      return _NNS_UINT64;
    case TF_FLOAT:
      return _NNS_FLOAT32;
    case TF_DOUBLE:
      return _NNS_FLOAT64;
    default:
      /** @todo Support other types */
      break;
  }

  return _NNS_END;
}

/**
 * @brief	return the data type of the tensor for Tensorflow
 * @param tType	: the defined type of NNStreamer
 * @return the enum of defined tensorflow::TF_DataType
 */
TF_DataType
TFCore::getTensorTypeToTF (tensor_type tType)
{
  switch (tType) {
    case _NNS_INT32:
      return TF_INT32;
    case _NNS_UINT32:
      return TF_UINT32;
    case _NNS_INT16:
      return TF_INT16;
    case _NNS_UINT16:
      return TF_UINT16;
    case _NNS_INT8:
      return TF_INT8;
    case _NNS_UINT8:
      return TF_UINT8;
    case _NNS_INT64:
      return TF_INT64;
    case _NNS_UINT64:
      return TF_UINT64;
    case _NNS_FLOAT32:
      return TF_FLOAT;
    case _NNS_FLOAT64:
      return TF_DOUBLE;
    default:
      /** @todo Support other types */
      break;
  }

  /* there is no flag for INVALID */
  return TF_VARIANT;
}

/**
 * @brief validate the src tensor info with graph
 * @param	tensorInfo : the tensors' info which user inserted
 * @param is_input : check is it input tensor or not to save the original shape
 * @note Compare user inserted tensor information with information from loaded graphs
 * @return 0 if OK. non-zero if error.
 *        -1 if getting rank of tensor is failed from the graph.
 *        -2 if getting shape of tensor is failed from the graph.
 */
int
TFCore::validateTensor (const GstTensorsInfo * tensorInfo, int is_input)
{
  for (unsigned int i = 0; i < tensorInfo->num_tensors; i++) {
    /* set the name of tensor */
    TF_Operation *op = TF_GraphOperationByName (graph, tensorInfo->info[i].name);

    g_assert (op != nullptr);

    const int num_outputs = TF_OperationNumOutputs (op);
    g_assert (num_outputs == 1); /* an in/output tensor has only one output for now */

    TF_Status *status = TF_NewStatus ();
    const TF_Output output = {op, 0};
    const TF_DataType type = TF_OperationOutputType (output);
    const int num_dims = TF_GraphGetTensorNumDims (graph, output, status);
    tf_tensor_info_s info_s;

    if (TF_GetCode (status) != TF_OK) {
      g_critical ("Error Tensor validation!! - [Code: %d] %s",
        TF_GetCode (status), TF_Message (status));
      TF_DeleteStatus (status);
      return -1;
    }

    if (type != TF_STRING) {
      g_assert (tensorInfo->info[i].type == getTensorTypeFromTF (type));
    }
    info_s.type = type;

    if (num_dims == -1) { /* in case of unknown shape */
      info_s.rank = 0;
    } else {
      g_assert (num_dims > 0);
      info_s.rank = num_dims;

      std::vector<std::int64_t> dims (num_dims);

      TF_GraphGetTensorShape (graph, output, dims.data (), num_dims, status);
      if (TF_GetCode (status) != TF_OK) {
        g_critical ("Error Tensor validation!! - [Code: %d] %s",
          TF_GetCode (status), TF_Message (status));
        TF_DeleteStatus (status);
        return -2;
      }

      /* check the validity of dimension */
      for (int d = 0; d < num_dims; ++d) {
        info_s.dims.push_back (
          static_cast<int64_t> (tensorInfo->info[i].dimension[num_dims - d - 1])
        );
        if (dims[d] < 0) {
          continue;
        }
        g_assert (tensorInfo->info[i].dimension[num_dims - d - 1] == dims[d]);
      }
    }
    if (is_input) {
      /* save the original shape of the tensor */
      input_tensor_info.push_back (info_s);
    }
    TF_DeleteStatus (status);
  }
  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::getInputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &inputTensorMeta);
  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::getOutputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &outputTensorMeta);
  return 0;
}

/**
 * @brief	the definition of a deallocator method
 */
static void
DeallocateInputTensor (void *data, size_t len, void *arg)
{
  tf_tensor_info_s *info_s = (tf_tensor_info_s *) arg;

  if (info_s && info_s->type == TF_STRING) {
    /* free encoded string */
    g_free (data);
  }

  return;
}

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 *        -1 if encoding STRING is failed.
 *        -2 if running session is failed.
 */
int
TFCore::run (const GstTensorMemory * input, GstTensorMemory * output)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  std::vector<TF_Output> input_ops;
  std::vector<TF_Tensor*> input_tensors;
  std::vector<TF_Output> output_ops;
  std::vector<TF_Tensor*> output_tensors;
  TF_Status* status = TF_NewStatus ();
  int ret = 0;

  /* create input tensor for the graph from `input` */
  for (unsigned int i = 0; i < inputTensorMeta.num_tensors; i++) {
    TF_Tensor* in_tensor = nullptr;
    TF_Output input_op = {
      TF_GraphOperationByName (graph, inputTensorMeta.info[i].name), 0
      };
    g_assert (input_op.oper != nullptr);
    input_ops.push_back (input_op);

    if (input_tensor_info[i].type == TF_STRING){
      size_t encoded_size = TF_StringEncodedSize (input[i].size);
      size_t total_size = 8 + encoded_size;

      char *input_encoded = (char*) g_malloc0 (total_size);
      if (input_encoded == NULL) {
        g_critical ("Failed to allocate memory for input tensor.");
        ret = -1;
        goto failed;
      }

      TF_StringEncode (
        (char *)input[i].data,
        input[i].size,
        input_encoded+8,
        encoded_size,
        status); /* fills the rest of tensor data */
      if (TF_GetCode (status) != TF_OK) {
        g_critical ("Error String Encoding!! - [Code: %d] %s",
          TF_GetCode (status), TF_Message (status));
        g_free (input_encoded);
        ret = -1;
        goto failed;
      }
      in_tensor = TF_NewTensor (
        input_tensor_info[i].type,
        NULL,
        0,
        input_encoded,
        total_size,
        &DeallocateInputTensor,
        &input_tensor_info[i]);
    } else {
      in_tensor = TF_NewTensor (
          input_tensor_info[i].type,
          input_tensor_info[i].dims.data (),
          input_tensor_info[i].rank,
          input[i].data,
          input[i].size,
          DeallocateInputTensor,
          &input_tensor_info[i]);
    }
    input_tensors.push_back (in_tensor);
  }

  /* create output tensor for the graph from `output` */
  for (unsigned int i = 0; i < outputTensorMeta.num_tensors; i++) {
    TF_Output output_op = {
      TF_GraphOperationByName (graph, outputTensorMeta.info[i].name), 0
      };
    g_assert (output_op.oper != nullptr);
    output_ops.push_back (output_op);

    TF_Tensor* out_tensor = nullptr;
    output_tensors.push_back (out_tensor);
  }

  TF_SessionRun (session,
                nullptr,
                input_ops.data (), input_tensors.data (),
                inputTensorMeta.num_tensors,
                output_ops.data (), output_tensors.data (),
                outputTensorMeta.num_tensors,
                nullptr, 0,
                nullptr,
                status
                );

  if (TF_GetCode (status) != TF_OK) {
    g_critical ("Error Running Session!! - [Code: %d] %s",
      TF_GetCode (status), TF_Message (status));
    ret = -2;
    goto failed;
  }

  for (unsigned int i = 0; i < outputTensorMeta.num_tensors; i++) {
    output[i].data = TF_TensorData (output_tensors[i]);
    outputTensorMap.insert (std::make_pair (output[i].data, output_tensors[i]));
  }

failed:
  for (unsigned int i = 0; i < input_tensors.size(); i++) {
    TF_DeleteTensor (input_tensors[i]);
  }

  TF_DeleteStatus (status);

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Run() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  return ret;
}

/**
 * @brief Free privateData and move on.
 */
static void
tf_close (const GstTensorFilterProperties * prop, void **private_data)
{
  TFCore *core = static_cast<TFCore *>(*private_data);

  if (!core)
    return;

  delete core;
  *private_data = NULL;
}

/**
 * @brief Load tensorflow modelfile
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
tf_loadModelFile (const GstTensorFilterProperties * prop, void **private_data)
{
  TFCore *core;
  const gchar *model_file;

  if (prop->num_models != 1)
    return -1;

  core = static_cast<TFCore *>(*private_data);
  model_file = prop->model_files[0];

  if (core != NULL) {
    if (g_strcmp0 (model_file, core->getModelPath ()) == 0)
      return 1; /* skipped */

    tf_close (prop, private_data);
  }

  core = new TFCore (model_file);
  if (core == NULL) {
    g_printerr ("Failed to allocate memory for filter subplugin: tensorflow\n");
    return -1;
  }

  if (core->init (prop) != 0) {
    *private_data = NULL;
    delete core;

    g_printerr ("failed to initailize the object: tensorflow\n");
    return -2;
  }

  *private_data = core;

  return 0;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 */
static int
tf_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int status = tf_loadModelFile (prop, private_data);

  return status;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 */
static int
tf_run (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  TFCore *core = static_cast<TFCore *>(*private_data);
  g_return_val_if_fail (core && input && output, -EINVAL);

  return core->run (input, output);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static int
tf_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  TFCore *core = static_cast<TFCore *>(*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getInputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop: property of tensor_filter instance
 * @param private_data : tensorflow plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static int
tf_getOutputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  TFCore *core = static_cast<TFCore *>(*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getOutputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param data : The data element.
 * @param private_data : tensorflow plugin's private data
 */
static void
tf_destroyNotify (void **private_data, void *data)
{
  TF_DeleteTensor ( (TFCore::outputTensorMap.find (data))->second);
  TFCore::outputTensorMap.erase (data);
}

static gchar filter_subplugin_tensorflow[] = "tensorflow";

static GstTensorFilterFramework NNS_support_tensorflow = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = tf_open,
  .close = tf_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_tf (void)
{
  NNS_support_tensorflow.name = filter_subplugin_tensorflow;
  NNS_support_tensorflow.allow_in_place = FALSE;      /** @todo: support this to optimize performance later. */
  NNS_support_tensorflow.allocate_in_invoke = TRUE;
  NNS_support_tensorflow.run_without_model = FALSE;
  NNS_support_tensorflow.verify_model_path = FALSE;
  NNS_support_tensorflow.invoke_NN = tf_run;
  NNS_support_tensorflow.getInputDimension = tf_getInputDim;
  NNS_support_tensorflow.getOutputDimension = tf_getOutputDim;
  NNS_support_tensorflow.destroyNotify = tf_destroyNotify;

  nnstreamer_filter_probe (&NNS_support_tensorflow);
}

/** @brief Destruct the subplugin */
void
fini_filter_tf (void)
{
  nnstreamer_filter_exit (NNS_support_tensorflow.name);
}
