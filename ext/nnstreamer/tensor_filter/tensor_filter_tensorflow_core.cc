/**
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
 * @file   tensor_filter_tensorflow_core.cc
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @date   08/02/2018
 * @brief  connection with tensorflow libraries.
 *
 * @bug     No known bugs.
 */

#include <nnstreamer_plugin_api.h>
#include "tensor_filter_tensorflow_core.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

std::map <void*, TF_Tensor*> TFCore::outputTensorMap;

/**
 * @brief	TFCore creator
 * @param	_model_path	: the logical path to '{model_name}.pb' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFCore::TFCore (const char *_model_path)
{
  model_path = _model_path;

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
}

/**
 * @brief	TFCore Destructor
 * @return	Nothing
 */
TFCore::~TFCore ()
{
  TF_DeleteGraph (graph);

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

  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
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

  g_assert (graph != nullptr);
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

  return TF_VARIANT; // there is no flag for INVALID
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
  for (int i = 0; i < tensorInfo->num_tensors; i++) {
    // set the name of tensor
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
    }
    else {
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

      // check the validity of dimension
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
DeallocateTensor (void* data, std::size_t, void*) {
  /* do nothing, the data will be free at the last of pipeline */
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
  char *input_encoded = nullptr;

  // create input tensor for the graph from `input`
  for (int i = 0; i < inputTensorMeta.num_tensors; i++) {
    TF_Tensor* in_tensor = nullptr;
    TF_Output input_op = {
      TF_GraphOperationByName (graph, inputTensorMeta.info[i].name), 0
      };
    g_assert (input_op.oper != nullptr);
    input_ops.push_back (input_op);

    if (input_tensor_info[i].type == TF_STRING){
      size_t encoded_size = TF_StringEncodedSize (input[i].size);
      size_t total_size = 8 + encoded_size;
      input_encoded = (char*) malloc (total_size);
      for (int j =0; j < 8; ++j) {
          input_encoded[j] = 0;
      }
      TF_StringEncode (
        (char *)input[i].data,
        input[i].size,
        input_encoded+8,
        encoded_size,
        status); // fills the rest of tensor data
      if (TF_GetCode (status) != TF_OK) {
        g_critical ("Error String Encoding!! - [Code: %d] %s",
          TF_GetCode (status), TF_Message (status));
        TF_DeleteStatus (status);
        return -1;
      }
      in_tensor = TF_NewTensor (
        input_tensor_info[i].type,
        NULL,
        0,
        input_encoded,
        total_size,
        &DeallocateTensor,
        nullptr);
    }
    else {
      in_tensor = TF_NewTensor (
          input_tensor_info[i].type,
          input_tensor_info[i].dims.data (),
          input_tensor_info[i].rank,
          input[i].data,
          input[i].size,
          DeallocateTensor, /* no deallocator */
          nullptr);
    }
    input_tensors.push_back (in_tensor);
  }

  // create output tensor for the graph from `output`
  for (int i = 0; i < outputTensorMeta.num_tensors; i++) {
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

  for (int i = 0; i < inputTensorMeta.num_tensors; i++) {
    TF_DeleteTensor (input_tensors[i]);
    if (input_tensor_info[i].type == TF_STRING && input_encoded){
      free (input_encoded);
    }
  }

  if (TF_GetCode (status) != TF_OK) {
    g_critical ("Error Running Session!! - [Code: %d] %s",
      TF_GetCode (status), TF_Message (status));
    TF_DeleteStatus (status);
    return -2;
  }

  for (int i = 0; i < outputTensorMeta.num_tensors; i++) {
    output[i].data = TF_TensorData (output_tensors[i]);
    outputTensorMap.insert (std::make_pair (output[i].data, output_tensors[i]));
  }
  TF_DeleteStatus (status);

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Run() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  return 0;
}

void *
tf_core_new (const char *_model_path)
{
  return new TFCore (_model_path);
}

/**
 * @brief	delete the TFCore class.
 * @param	tf	: the class object
 * @return	Nothing
 */
void
tf_core_delete (void * tf)
{
  TFCore *c = (TFCore *) tf;
  delete c;
}

/**
 * @brief	initialize the object with tf model
 * @param	tf	: the class object
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_init (void * tf, const GstTensorFilterProperties * prop,
  const gboolean tf_mem_optmz)
{
  TFCore *c = (TFCore *) tf;
  return c->init (prop);
}

/**
 * @brief	get model path
 * @param	tf	: the class object
 * @return	model path
 */
const char *
tf_core_getModelPath (void * tf)
{
  TFCore *c = (TFCore *) tf;
  return c->getModelPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of model
 * @param	tf	the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_getInputDim (void * tf, GstTensorsInfo * info)
{
  TFCore *c = (TFCore *) tf;
  return c->getInputTensorDim (info);
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tf	the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_getOutputDim (void * tf, GstTensorsInfo * info)
{
  TFCore *c = (TFCore *) tf;
  return c->getOutputTensorDim (info);
}

/**
 * @brief	run the model
 * @param	tf	: the class object
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_run (void * tf, const GstTensorMemory * input, GstTensorMemory * output)
{
  TFCore *c = (TFCore *) tf;
  return c->run (input, output);
}

/**
 * @brief	the destroy notify method for tensorflow. it will free the output tensor
 * @param[in] data : the data element destroyed at the pipeline
 */
void
tf_core_destroyNotify (void * data)
{
  TF_DeleteTensor ( (TFCore::outputTensorMap.find (data))->second);
  TFCore::outputTensorMap.erase (data);
}
