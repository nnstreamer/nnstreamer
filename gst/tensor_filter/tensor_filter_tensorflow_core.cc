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

#include "tensor_filter_tensorflow_core.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

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
  if (setTensorProp (&inputTensorMeta, &prop->input_meta)) {
    GST_ERROR ("Failed to initialize input tensor\n");
    return -2;
  }
  if (setTensorProp (&outputTensorMeta, &prop->output_meta)) {
    GST_ERROR ("Failed to initialize output tensor\n");
    return -3;
  }
  if (loadModel ()) {
    GST_ERROR ("Failed to load model\n");
    return -1;
  }
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
 * @brief	load the tf model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 *        -1 if the pb file is not loaded.
 *        -2 if the input properties is different with model.
 *        -3 if the Tensorflow session is not created.
 */
int
TFCore::loadModel ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(Env::Default(), model_path, &graph_def);
  if (!load_graph_status.ok()) {
    GST_ERROR ("Failed to load compute graph at '%s'", model_path);
    return -1;
  }

  /* get input tensor */
  std::vector<const NodeDef*> placeholders;
  for (const NodeDef& node : graph_def.node()) {
    if (node.op() == "Placeholder") {
      placeholders.push_back(&node);
    }
  }

  if (placeholders.empty()) {
    GST_WARNING ("No inputs spotted.");
  } else {
    GST_INFO ("Found possible inputs: %ld", placeholders.size());
    if (inputTensorValidation(placeholders)) {
      GST_ERROR ("Input Tensor Information is not valid");
      return -2;
    }
  }

  /* get session */
  Status new_session_status = NewSession(SessionOptions(), &session);
  Status session_create_status = session->Create(graph_def);
  if (!new_session_status.ok() || !session_create_status.ok()) {
    GST_ERROR ("Create Tensorflow Session was Failed");
    return -3;
  }
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
TFCore::getTensorTypeFromTF (DataType tfType)
{
  switch (tfType) {
    case DT_INT32:
      return _NNS_INT32;
    case DT_UINT32:
      return _NNS_UINT32;
    case DT_INT16:
      return _NNS_INT16;
    case DT_UINT16:
      return _NNS_UINT16;
    case DT_INT8:
      return _NNS_INT8;
    case DT_UINT8:
      return _NNS_UINT8;
    case DT_INT64:
      return _NNS_INT64;
    case DT_UINT64:
      return _NNS_UINT64;
    case DT_FLOAT:
      return _NNS_FLOAT32;
    case DT_DOUBLE:
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
 * @return the enum of defined tensorflow::DataType
 */
DataType
TFCore::getTensorTypeToTF (tensor_type tType)
{
  switch (tType) {
    case _NNS_INT32:
      return DT_INT32;
    case _NNS_UINT32:
      return DT_UINT32;
    case _NNS_INT16:
      return DT_INT16;
    case _NNS_UINT16:
      return DT_UINT16;
    case _NNS_INT8:
      return DT_INT8;
    case _NNS_UINT8:
      return DT_UINT8;
    case _NNS_INT64:
      return DT_INT64;
    case _NNS_UINT64:
      return DT_UINT64;
    case _NNS_FLOAT32:
      return DT_FLOAT;
    case _NNS_FLOAT64:
      return DT_DOUBLE;
    default:
      /** @todo Support other types */
      break;
  }

  return DT_INVALID;
}

/**
 * @brief	check the inserted information about input tensor with model
 * @return 0 if OK. non-zero if error.
 *        -1 if the number of input tensors is not matched.
 *        -2 if the name of input tensors is not matched.
 *        -3 if the type of input tensors is not matched.
 *        -4 if the dimension of input tensors is not matched.
 *        -5 if the rank of input tensors exceeds our capacity NNS_TENSOR_RANK_LIMIT.
 */
int
TFCore::inputTensorValidation (std::vector<const NodeDef*> placeholders)
{
  if (inputTensorMeta.num_tensors != placeholders.size()){
    GST_ERROR ("Input Tensor is not valid: the number of input tensor is different\n");
    return -1;
  }
  int length = placeholders.size();
  for (int i = 0; i < length; i++) {
    const NodeDef* node = placeholders[i];
    string shape_description = "None";
    if (node->attr().count("shape")) {
      TensorShapeProto shape_proto = node->attr().at("shape").shape();
      Status shape_status = PartialTensorShape::IsValidShape(shape_proto);
      if (shape_status.ok()) {
        shape_description = PartialTensorShape(shape_proto).DebugString();
      } else {
        shape_description = shape_status.error_message();
      }
    }
    char chars[] = "[]";
    for (unsigned int i = 0; i < strlen(chars); ++i)
    {
      shape_description.erase (
        std::remove(
          shape_description.begin(),
          shape_description.end(),
          chars[i]
        ),
        shape_description.end()
      );
    }

    DataType dtype = DT_INVALID;
    if (node->attr().count("dtype")) {
      dtype = node->attr().at("dtype").type();
    }

    if (strcmp (inputTensorMeta.info[i].name, node->name().c_str())){
      GST_ERROR ("Input Tensor is not valid: the name of input tensor is different\n");
      return -2;
    }
    if (inputTensorMeta.info[i].type != getTensorTypeFromTF(dtype)){
      GST_ERROR ("Input Tensor is not valid: the type of input tensor is different\n");
      return -3;
    }

    gchar **str_dims;
    str_dims = g_strsplit (shape_description.c_str(), ",", -1);
    inputTensorRank[i] = g_strv_length (str_dims);
    if (inputTensorRank[i] > NNS_TENSOR_RANK_LIMIT){
      GST_ERROR ("The Rank of Input Tensor is not affordable. It's over our capacity.\n");
      return -5;
    }
    for (int j = 0; j < inputTensorRank[i]; j++) {
      if (!strcmp (str_dims[j], "?"))
        continue;

      if (inputTensorMeta.info[i].dimension[inputTensorRank[i] - j - 1] != atoi (str_dims[j])){
        GST_ERROR ("Input Tensor is not valid: the dim of input tensor is different\n");
        return -4;
      }
    }
  }
  return 0;
}

/**
 * @brief extract and store the information of src tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::setTensorProp (GstTensorsInfo * dest, const GstTensorsInfo * src)
{
  dest->num_tensors = src->num_tensors;
  for (int i = 0; i < src->num_tensors; i++) {
    dest->info[i].name = src->info[i].name;
    dest->info[i].type = src->info[i].type;
    for (int j = 0; j < NNS_TENSOR_RANK_LIMIT; j++) {
      dest->info[i].dimension[j] = src->info[i].dimension[j];
    }
  }
  return 0;
}

/**
 * @brief	return the number of Input Tensors.
 * @return	the number of Input Tensors.
 */
int
TFCore::getInputTensorSize ()
{
  return inputTensorMeta.num_tensors;
}

/**
 * @brief	return the number of Output Tensors.
 * @return	the number of Output Tensors
 */
int
TFCore::getOutputTensorSize ()
{
  return outputTensorMeta.num_tensors;
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
  info->num_tensors = inputTensorMeta.num_tensors;
  memcpy (info->info, inputTensorMeta.info,
      sizeof (GstTensorInfo) * inputTensorMeta.num_tensors);
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
  info->num_tensors = outputTensorMeta.num_tensors;
  memcpy (info->info, outputTensorMeta.info,
      sizeof (GstTensorInfo) * outputTensorMeta.num_tensors);
  return 0;
}

#define copyInputWithType(type) \
  inputTensor.flat<type>()(j) = ((type*)input->data)[j];

#define copyOutputWithType(type) \
  for(int j = 0; j < n; j++) \
    ((type *)output[i].data)[j] = outputs[i].flat<type>()(j); \

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::run (const GstTensorMemory * input, GstTensorMemory * output)
{
  std::vector<std::pair<string, Tensor>> input_feeds;
  std::vector<string> output_tensor_names;
  std::vector<Tensor> outputs;

  for (int i = 0; i < inputTensorMeta.num_tensors; i++) {
    TensorShape ts = TensorShape({});
    for (int j = inputTensorRank[i] - 1; j >= 0; j--){
      ts.AddDim(inputTensorMeta.info[i].dimension[j]);
    }
    Tensor inputTensor(
      getTensorTypeToTF(input->type),
      ts
    );
    int len = input->size / tensor_element_size[input->type];

    for (int j = 0; j < len; j++) {
      switch (input->type) {
        case _NNS_INT32:
          copyInputWithType (int32);
          break;
        case _NNS_UINT32:
          copyInputWithType (uint32);
          break;
        case _NNS_INT16:
          copyInputWithType (int16);
          break;
        case _NNS_UINT16:
          copyInputWithType (uint16);
          break;
        case _NNS_INT8:
          copyInputWithType (int8);
          break;
        case _NNS_UINT8:
          copyInputWithType (uint8);
          break;
        case _NNS_INT64:
          copyInputWithType (int64);
          break;
        case _NNS_UINT64:
          copyInputWithType (uint64);
          break;
        case _NNS_FLOAT32:
          copyInputWithType (float);
          break;
        case _NNS_FLOAT64:
          copyInputWithType (double);
          break;
        default:
          /** @todo Support other types */
          break;
      }
    }
    input_feeds.push_back({inputTensorMeta.info[i].name, inputTensor});
  }

  for (int i = 0; i < outputTensorMeta.num_tensors; i++) {
    output_tensor_names.push_back(outputTensorMeta.info[i].name);
  }

  Status run_status =
      session->Run(input_feeds, output_tensor_names, {}, &outputs);


  for (int i = 0; i < outputTensorMeta.num_tensors; i++) {
    output[i].type = getTensorTypeFromTF(outputs[i].dtype());
    output[i].size = tensor_element_size[output[i].type];
    for (int j = 0; j < NNS_TENSOR_RANK_LIMIT; j++)
      output[i].size *= outputTensorMeta.info[i].dimension[j];

    int n = output[i].size / tensor_element_size[output[i].type];

    switch (output[i].type) {
      case _NNS_INT32:{
        copyOutputWithType (int32);
        break;
      }
      case _NNS_UINT32:{
        copyOutputWithType (uint32);
        break;
      }
      case _NNS_INT16:{
        copyOutputWithType (int16);
        break;
      }
      case _NNS_UINT16:{
        copyOutputWithType (uint16);
        break;
      }
      case _NNS_INT8:{
        copyOutputWithType (int8);
        break;
      }
      case _NNS_UINT8:{
        copyOutputWithType (uint8);
        break;
      }
      case _NNS_INT64:{
        copyOutputWithType (int64);
        break;
      }
      case _NNS_UINT64:{
        copyOutputWithType (uint64);
        break;
      }
      case _NNS_FLOAT32:{
        copyOutputWithType (float);
        break;
      }
      case _NNS_FLOAT64:{
        copyOutputWithType (double);
        break;
      }
      default:
        /** @todo Support other types */
        break;
    }
  }

  return 0;
}

extern void *
tf_core_new (const char *_model_path)
{
  return new TFCore (_model_path);
}

/**
 * @brief	delete the TFCore class.
 * @param	tf	: the class object
 * @return	Nothing
 */
extern void
tf_core_delete (void *tf)
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
tf_core_init (void *tf, const GstTensorFilterProperties * prop)
{
  TFCore *c = (TFCore *) tf;
  int ret = c->init (prop);
  return ret;
}

/**
 * @brief	get model path
 * @param	tf	: the class object
 * @return	model path
 */
extern const char *
tf_core_getModelPath (void *tf)
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
tf_core_getInputDim (void *tf, GstTensorsInfo * info)
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
tf_core_getOutputDim (void *tf, GstTensorsInfo * info)
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
tf_core_run (void *tf, const GstTensorMemory * input, GstTensorMemory * output)
{
  TFCore *c = (TFCore *) tf;
  return c->run (input, output);
}
