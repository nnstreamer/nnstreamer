/**
 * GStreamer Tensor_Filter, OpenVino (DLDT) Module
 * Copyright (C) 2019 Wook Song <wook16.song@samsung.com>
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
 * @file    tensor_filter_openvino.cc
 * @date    23 Dec 2019
 * @brief   Tensor_filter subplugin for OpenVino (DLDT).
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (OpenVino) for tensor_filter.
 */

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>
#ifdef __OPENVINO_CPU_EXT__
#include <ext_list.hpp>
#endif /* __OPENVINO_CPU_EXT__ */
#include <inference_engine.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include "tensor_filter_openvino.hh"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_filter_openvino (void) __attribute__ ((constructor));
void fini_filter_openvino (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

std::map<accl_hw, std::string> TensorFilterOpenvino::_nnsAcclHwToOVDevMap = {
  { ACCL_CPU, "CPU" },
  { ACCL_NPU, "MYRIAD" },
  { ACCL_NPU_MOVIDIUS, "MYRIAD" },
};

const std::string TensorFilterOpenvino::extBin = ".bin";
const std::string TensorFilterOpenvino::extXml = ".xml";

std::mutex TensorFilterOpenvino::_ieCoreLock;
bool TensorFilterOpenvino::_cpuExtAdded = false;

/**
 * @brief Get the Inference Engine core shared by every instance in the process
 * @return the shared core
 *
 * Destroying a core unloads the device plugins it has loaded, and the CPU
 * plugin of OpenVino 2019R3 leaves a worker thread of its own running in its
 * code when it is unloaded, which takes the process down. The core is,
 * therefore, created once and never destroyed. The caller must hold
 * _ieCoreLock while using it.
 */
InferenceEngine::Core &
TensorFilterOpenvino::getIECore ()
{
  static InferenceEngine::Core *core = new InferenceEngine::Core ();

  return *core;
}

/**
 * @brief Convert the string representing the tensor data type to _nns_tensor_type
 * @param type a std::string representing the tensor data type in InferenceEngine
 * @return _nns_tensor_type corresponding to the tensor data type in InferenceEngine if OK, otherwise _NNS_END
 */
tensor_type
TensorFilterOpenvino::convertFromIETypeStr (std::string type)
{
  if (type[0] == 'U') {
    if (type[1] == '8')
      return _NNS_UINT8;
    else
      return _NNS_UINT16;
  } else if (type[0] == 'I') {
    if (type[1] == '1')
      return _NNS_INT16;
    else if (type[1] == '3')
      return _NNS_INT32;
    else
      return _NNS_INT8;
  } else if (type[0] == 'F') {
    if (type[2] == '3')
      return _NNS_FLOAT32;
    else
      return _NNS_END;
  } else {
    return _NNS_END;
  }
}

/**
 * @brief Convert a tensor container in NNS to a tensor container in IE
 * @param tensorDesc the class that defines a Tensor description to be converted from a GstTensorMemory
 * @param gstTensor the container of a tensor in NNS to be converted to a tensor container in IE
 * @return a pointer to the Blob which is a container of a tensor in IE if OK, otherwise nullptr
 */
InferenceEngine::Blob::Ptr
TensorFilterOpenvino::convertGstTensorMemoryToBlobPtr (const InferenceEngine::TensorDesc tensorDesc,
    const GstTensorMemory *gstTensor, const tensor_type gstType)
{
  switch (gstType) {
    case _NNS_UINT8:
      return InferenceEngine::Blob::Ptr (new InferenceEngine::TBlob<uint8_t> (
          tensorDesc, (uint8_t *) gstTensor->data, gstTensor->size));
    case _NNS_UINT16:
      return InferenceEngine::Blob::Ptr (new InferenceEngine::TBlob<uint16_t> (
          tensorDesc, (uint16_t *) gstTensor->data, gstTensor->size));
    case _NNS_INT8:
      return InferenceEngine::Blob::Ptr (new InferenceEngine::TBlob<int8_t> (
          tensorDesc, (int8_t *) gstTensor->data, gstTensor->size));
    case _NNS_INT16:
      return InferenceEngine::Blob::Ptr (new InferenceEngine::TBlob<int16_t> (
          tensorDesc, (int16_t *) gstTensor->data, gstTensor->size));
    case _NNS_INT32:
      return InferenceEngine::Blob::Ptr (new InferenceEngine::TBlob<int32_t> (
          tensorDesc, (int32_t *) gstTensor->data, gstTensor->size));
    case _NNS_FLOAT32:
      return InferenceEngine::Blob::Ptr (new InferenceEngine::TBlob<float> (
          tensorDesc, (float *) gstTensor->data, gstTensor->size));
    default:
      return nullptr;
  }
}

/**
 * @brief Check the given hw is supported by the fw or not
 * @param devsVector a reference of a vector of the available device names (the return of GetAvailableDevices () of the core)
 * @param hw a user-given acceleration device of which the data type is accl_hw
 * @return TRUE if supported
 */
inline bool
TensorFilterOpenvino::isAcclDevSupported (std::vector<std::string> &devsVector, accl_hw hw)
{
  std::vector<std::string>::iterator it;

  it = std::find (devsVector.begin (), devsVector.end (), _nnsAcclHwToOVDevMap[hw]);

  if (it == devsVector.end ()) {
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Resolve the name to be used as the key of an Inference Engine blob map
 * @param modelNames the tensor names declared by the given model
 * @param info the tensor info configured for this instance
 * @param nth the index of the tensor
 * @param[out] name the resolved name
 * @return true if a name is resolved, false if there is none to use
 *
 * A name given by the user has the priority. The framework, however, keeps the
 * tensor info of the user intact when it is compatible with the model, and the
 * comparison it uses ignores the names. A user who configures the dimensions
 * and the types only, therefore, leaves the names unset, in which case the
 * names of the model are the ones to use.
 */
bool
TensorFilterOpenvino::getBlobName (const std::vector<std::string> &modelNames,
    const GstTensorInfo *info, guint nth, std::string &name)
{
  if (info->name != NULL) {
    name = info->name;
    return true;
  }

  if (nth >= modelNames.size ())
    return false;

  name = modelNames[nth];

  return true;
}

/**
 * @brief Get a path where the model file in XML format is located
 * @return a std::string of the path
 */
std::string
TensorFilterOpenvino::getPathModelXml ()
{
  return this->_pathModelXml;
}

/**
 * @brief Get a path where the model file in bin format is located
 * @return a std::string of the path
 */
std::string
TensorFilterOpenvino::getPathModelBin ()
{
  return this->_pathModelBin;
}

/**
 * @brief TensorFilterOpenvino constructor
 * @param pathModelXml the path of the given model in a XML format
 * @param pathModelBin the path of the given model in a Bin format
 * @return  Nothing
 */
TensorFilterOpenvino::TensorFilterOpenvino (std::string pathModelXml, std::string pathModelBin)
{
  guint i = 0;

  this->_pathModelXml = pathModelXml;
  this->_pathModelBin = pathModelBin;
  (this->_networkReaderCNN).ReadNetwork (this->_pathModelXml);
  (this->_networkReaderCNN).ReadWeights (this->_pathModelBin);
  this->_networkCNN = (this->_networkReaderCNN).getNetwork ();
  this->_inputsDataMap = (this->_networkCNN).getInputsInfo ();
  this->_outputsDataMap = (this->_networkCNN).getOutputsInfo ();

  /** The names and the descriptors of the tensors are what the model tells,
   *  and invoke () needs them whether the dimension callbacks run or not. */
  for (auto &eachInput : this->_inputsDataMap) {
    if (i >= NNS_TENSOR_SIZE_LIMIT)
      break;
    this->_inputTensorNames.push_back (eachInput.second->name ());
    this->_inputTensorDescs[i++] = eachInput.second->getTensorDesc ();
  }

  i = 0;
  for (auto &eachOutput : this->_outputsDataMap) {
    if (i >= NNS_TENSOR_SIZE_LIMIT)
      break;
    this->_outputTensorNames.push_back (eachOutput.second->getName ());
    this->_outputTensorDescs[i++] = eachOutput.second->getTensorDesc ();
  }

  this->_isLoaded = false;
  this->_hw = ACCL_NONE;
}

/**
 * @brief TensorFilterOpenvino Destructor
 * @return  Nothing
 */
TensorFilterOpenvino::~TensorFilterOpenvino ()
{
  ;
}

/**
 * @brief Load the given neural network into the target device
 * @param hw a user-given acceleration device to use
 * @return 0 (TensorFilterOpenvino::RetSuccess) if OK, negative values if error
 */
int
TensorFilterOpenvino::loadModel (accl_hw hw)
{
  std::string targetDevice;
  std::vector<std::string> strVector;

  if (this->_isLoaded) {
    /** @todo Can OpenVino support to replace the loaded model with a new one? */
    ml_loge ("The model file is already loaded onto the device.");
    return RetEBusy;
  }

  std::lock_guard<std::mutex> lock (_ieCoreLock);
  InferenceEngine::Core &ieCore = getIECore ();

  strVector = ieCore.GetAvailableDevices ();
  if (strVector.size () == 0) {
    ml_loge ("No devices found for the OpenVino toolkit; "
             "check your plugin is installed, and the device is also connected.");
    return RetENoDev;
  }

  if (!TensorFilterOpenvino::isAcclDevSupported (strVector, hw)) {
    ml_loge ("Failed to find the device (%s) or its plugin (%s)",
        get_accl_hw_str (hw), _nnsAcclHwToOVDevMap[hw].c_str ());
    return RetEInval;
  }

#ifdef __OPENVINO_CPU_EXT__
  if (hw == ACCL_CPU && !_cpuExtAdded) {
    ieCore.AddExtension (
        std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions> (),
        _nnsAcclHwToOVDevMap[hw]);
    _cpuExtAdded = true;
  }
#endif
  /** @todo Catch the IE exception */
  this->_executableNet
      = ieCore.LoadNetwork (this->_networkCNN, _nnsAcclHwToOVDevMap[hw]);
  this->_hw = hw;
  this->_isLoaded = true;
  this->_inferRequest = this->_executableNet.CreateInferRequest ();

  return RetSuccess;
}

/**
 * @brief Check the neural network model is loaded.
 */
bool
TensorFilterOpenvino::isModelLoaded ()
{
  return _isLoaded;
}

/**
 * @brief	Get the information about the dimensions of input tensors from the given model
 * @param[out] info metadata containing the dimensions and types information of the input tensors
 * @return 0 (TensorFilterOpenvino::RetSuccess) if OK, negative values if error
 */
int
TensorFilterOpenvino::getInputTensorDim (GstTensorsInfo *info)
{
  InferenceEngine::InputsDataMap *inputsDataMap = &(this->_inputsDataMap);
  InferenceEngine::InputsDataMap::iterator inputDataMapIter;
  GstTensorInfo *_info;
  int ret, i, j;

  gst_tensors_info_init (info);

  info->num_tensors = (uint32_t) inputsDataMap->size ();
  if (info->num_tensors > NNS_TENSOR_SIZE_LIMIT) {
    ml_loge ("The number of input tenosrs in the model "
             "exceeds more than NNS_TENSOR_SIZE_LIMIT, %s",
        NNS_TENSOR_SIZE_LIMIT_STR);
    ret = RetEOverFlow;
    goto failed;
  }

  for (inputDataMapIter = inputsDataMap->begin (), i = 0;
      inputDataMapIter != inputsDataMap->end (); ++inputDataMapIter, ++i) {
    InferenceEngine::SizeVector::reverse_iterator sizeVecRIter;
    InferenceEngine::TensorDesc eachInputTensorDesc;
    InferenceEngine::InputInfo::Ptr eachInputInfo;
    InferenceEngine::SizeVector dimsSizeVec;
    std::string ieTensorTypeStr;
    tensor_type nnsTensorType;

    eachInputInfo = inputDataMapIter->second;
    eachInputTensorDesc = eachInputInfo->getTensorDesc ();
    dimsSizeVec = eachInputTensorDesc.getDims ();
    if (dimsSizeVec.size () > NNS_TENSOR_RANK_LIMIT) {
      ml_loge ("The ranks of dimensions of InputTensor[%d] in the model "
               "exceeds NNS_TENSOR_RANK_LIMIT, %u",
          i, NNS_TENSOR_RANK_LIMIT);
      ret = RetEOverFlow;
      goto failed;
    }

    _info = gst_tensors_info_get_nth_info (info, i);

    for (sizeVecRIter = dimsSizeVec.rbegin (), j = 0;
        sizeVecRIter != dimsSizeVec.rend (); ++sizeVecRIter, ++j) {
      _info->dimension[j] = (*sizeVecRIter != 0 ? *sizeVecRIter : 1);
    }

    ieTensorTypeStr = eachInputInfo->getPrecision ().name ();
    nnsTensorType = TensorFilterOpenvino::convertFromIETypeStr (ieTensorTypeStr);
    if (nnsTensorType == _NNS_END) {
      ml_loge ("The type of tensor elements, %s, "
               "in the model is not supported",
          ieTensorTypeStr.c_str ());
      ret = RetEInval;
      goto failed;
    }

    _info->type = nnsTensorType;
    _info->name = g_strdup (eachInputInfo->name ().c_str ());
    this->_inputTensorDescs[i] = eachInputTensorDesc;
  }

  return TensorFilterOpenvino::RetSuccess;

failed:
  ml_loge ("Failed to get dimension information about input tensor");

  return ret;
}

/**
 * @brief	Get the information about the dimensions of output tensors from the given model
 * @param[out] info metadata containing the dimensions and types information of the output tensors
 * @return 0 (TensorFilterOpenvino::RetSuccess) if OK, negative values if error
 */
int
TensorFilterOpenvino::getOutputTensorDim (GstTensorsInfo *info)
{
  InferenceEngine::OutputsDataMap *outputsDataMap = &(this->_outputsDataMap);
  InferenceEngine::OutputsDataMap::iterator outputDataMapIter;
  GstTensorInfo *_info;
  int ret, i, j;

  gst_tensors_info_init (info);

  info->num_tensors = (uint32_t) outputsDataMap->size ();
  if (info->num_tensors > NNS_TENSOR_SIZE_LIMIT) {
    ml_loge ("The number of output tenosrs in the model "
             "exceeds more than NNS_TENSOR_SIZE_LIMIT, %s",
        NNS_TENSOR_SIZE_LIMIT_STR);
    ret = RetEOverFlow;
    goto failed;
  }

  for (outputDataMapIter = outputsDataMap->begin (), i = 0;
      outputDataMapIter != outputsDataMap->end (); ++outputDataMapIter, ++i) {
    InferenceEngine::SizeVector::reverse_iterator sizeVecRIter;
    InferenceEngine::TensorDesc eachOutputTensorDesc;
    InferenceEngine::SizeVector dimsSizeVec;
    InferenceEngine::DataPtr eachOutputInfo;
    std::string ieTensorTypeStr;
    tensor_type nnsTensorType;

    eachOutputInfo = outputDataMapIter->second;
    eachOutputTensorDesc = eachOutputInfo->getTensorDesc ();
    dimsSizeVec = eachOutputTensorDesc.getDims ();
    if (dimsSizeVec.size () > NNS_TENSOR_RANK_LIMIT) {
      ml_loge ("The ranks of dimensions of OutputTensor[%d] in the model "
               "exceeds NNS_TENSOR_RANK_LIMIT, %u",
          i, NNS_TENSOR_RANK_LIMIT);
      ret = RetEOverFlow;
      goto failed;
    }

    _info = gst_tensors_info_get_nth_info (info, i);

    for (sizeVecRIter = dimsSizeVec.rbegin (), j = 0;
        sizeVecRIter != dimsSizeVec.rend (); ++sizeVecRIter, ++j) {
      _info->dimension[j] = (*sizeVecRIter != 0 ? *sizeVecRIter : 1);
    }

    ieTensorTypeStr = eachOutputInfo->getPrecision ().name ();
    nnsTensorType = TensorFilterOpenvino::convertFromIETypeStr (ieTensorTypeStr);
    if (nnsTensorType == _NNS_END) {
      ml_loge ("The type of tensor elements, %s, "
               "in the model is not supported",
          ieTensorTypeStr.c_str ());
      ret = RetEInval;
      goto failed;
    }

    _info->type = nnsTensorType;
    _info->name = g_strdup (eachOutputInfo->getName ().c_str ());
    this->_outputTensorDescs[i] = eachOutputTensorDesc;
  }

  return TensorFilterOpenvino::RetSuccess;

failed:
  ml_loge ("Failed to get dimension information about output tensor");

  return ret;
}

/**
 * @brief Do inference using Inference Engine of the OpenVino framework
 * @param inInfo the tensor info configured for the input tensors
 * @param outInfo the tensor info configured for the output tensors
 * @param[in] input the array of input tensors
 * @param[out] output the array of output tensors
 * @return RetSuccess if OK. non-zero if error
 */
int
TensorFilterOpenvino::invoke (const GstTensorsInfo *inInfo,
    const GstTensorsInfo *outInfo, const GstTensorMemory *input, GstTensorMemory *output)
{
  InferenceEngine::BlobMap inBlobMap;
  InferenceEngine::BlobMap outBlobMap;
  GstTensorInfo *info;
  guint num_tensors;
  guint i;

  num_tensors = inInfo->num_tensors;
  for (i = 0; i < num_tensors; ++i) {
    std::string name;

    info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) inInfo, i);
    if (!getBlobName (this->_inputTensorNames, info, i, name)) {
      ml_loge ("Failed to get the name of the input tensor: %u", i);
      return RetEInval;
    }

    InferenceEngine::Blob::Ptr blob = convertGstTensorMemoryToBlobPtr (
        this->_inputTensorDescs[i], &(input[i]), info->type);
    if (blob == nullptr) {
      ml_loge ("Failed to create a blob for the input tensor: %u", i);
      return RetEInval;
    }
    inBlobMap.insert (make_pair (name, blob));
  }
  this->_inferRequest.SetInput (inBlobMap);

  num_tensors = outInfo->num_tensors;
  for (i = 0; i < num_tensors; ++i) {
    std::string name;

    info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) outInfo, i);
    if (!getBlobName (this->_outputTensorNames, info, i, name)) {
      ml_loge ("Failed to get the name of the output tensor: %u", i);
      return RetEInval;
    }

    InferenceEngine::Blob::Ptr blob = convertGstTensorMemoryToBlobPtr (
        this->_outputTensorDescs[i], &(output[i]), info->type);
    if (blob == nullptr) {
      ml_loge ("Failed to create a blob for the output tensor: %u", i);
      return RetEInval;
    }
    outBlobMap.insert (make_pair (name, blob));
  }
  this->_inferRequest.SetOutput (outBlobMap);

  this->_inferRequest.Infer ();

  return RetSuccess;
}


namespace nnstreamer
{
namespace tensorfilter_openvino
{

/**
 * @brief The tensor_filter sub-plugin running a model with TensorFilterOpenvino
 */
class openvino_subplugin final : public tensor_filter_subplugin
{
  public:
  static void init_filter_openvino ();
  static void fini_filter_openvino ();

  openvino_subplugin ();
  ~openvino_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);

  private:
  static accl_hw getAccelerator (const GstTensorFilterProperties *prop);
  static void getModelPaths (const GstTensorFilterProperties *prop,
      std::string &pathXml, std::string &pathBin);

  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw;
  static openvino_subplugin *registeredRepresentation;

  std::unique_ptr<TensorFilterOpenvino> tfOv;
  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;
};

const char *openvino_subplugin::name = "openvino";
/** The first one is the accelerator for the default and the auto config */
const accl_hw openvino_subplugin::hw_list[] = { ACCL_NPU_MOVIDIUS, ACCL_NPU, ACCL_CPU };
const int openvino_subplugin::num_hw = G_N_ELEMENTS (openvino_subplugin::hw_list);
openvino_subplugin *openvino_subplugin::registeredRepresentation = nullptr;

/**
 * @brief Construct an empty instance
 */
openvino_subplugin::openvino_subplugin ()
    : tensor_filter_subplugin (), tfOv (nullptr)
{
  gst_tensors_info_init (&inputInfo);
  gst_tensors_info_init (&outputInfo);
}

/**
 * @brief Destruct the instance
 */
openvino_subplugin::~openvino_subplugin ()
{
  gst_tensors_info_free (&inputInfo);
  gst_tensors_info_free (&outputInfo);
}

/**
 * @brief Get a new empty instance of the sub-plugin
 */
tensor_filter_subplugin &
openvino_subplugin::getEmptyInstance ()
{
  return *(new openvino_subplugin ());
}

/**
 * @brief Get the accelerator to load the model onto
 * @param prop property of tensor_filter instance
 * @return the accelerator
 */
accl_hw
openvino_subplugin::getAccelerator (const GstTensorFilterProperties *prop)
{
  accl_hw hw;
  int i;

  if (prop->num_hw <= 0 || prop->hw_list == NULL)
    return hw_list[0];

  /** The model is loaded onto the first accelerator only */
  hw = prop->hw_list[0];
  if (hw == ACCL_AUTO || hw == ACCL_DEFAULT)
    return hw_list[0];

  for (i = 0; i < num_hw; ++i) {
    if (hw_list[i] == hw)
      return hw;
  }

  throw std::invalid_argument (std::string ("'") + get_accl_hw_str (hw)
                               + "' is not a valid value for the 'accelerator' property. "
                                 "An acceptable format is 'true:[cpu|npu.movidius]', "
                                 "and 'cpu' is only for the x86_64 architecture.");
}

/**
 * @brief Get the paths of the model files in the XML and the bin format
 * @param prop property of tensor_filter instance
 * @param[out] pathXml the path of the model file in the XML format
 * @param[out] pathBin the path of the model file in the bin format
 */
void
openvino_subplugin::getModelPaths (const GstTensorFilterProperties *prop,
    std::string &pathXml, std::string &pathBin)
{
  const std::string &extXml = TensorFilterOpenvino::extXml;
  const std::string &extBin = TensorFilterOpenvino::extBin;
  guint num_models_xml = 0;
  guint num_models_bin = 0;

  if (prop->num_models == 1) {
    std::string path_model_prefix (prop->model_files[0]);

    if (g_str_has_suffix (prop->model_files[0], extBin.c_str ())
        || g_str_has_suffix (prop->model_files[0], extXml.c_str ()))
      path_model_prefix.resize (path_model_prefix.length () - 4);

    pathXml = path_model_prefix + extXml;
    pathBin = path_model_prefix + extBin;
  } else {
    for (gint i = 0; i < prop->num_models; ++i) {
      if (g_str_has_suffix (prop->model_files[i], extXml.c_str ())) {
        num_models_xml++;
        pathXml = std::string (prop->model_files[i]);
      } else if (g_str_has_suffix (prop->model_files[i], extBin.c_str ())) {
        num_models_bin++;
        pathBin = std::string (prop->model_files[i]);
      }

      if (num_models_xml > 1)
        throw std::invalid_argument ("Too many model files in a XML format are provided.");
      else if (num_models_bin > 1)
        throw std::invalid_argument ("Too many model files in a BIN format are provided.");
    }
  }

  if (!g_file_test (pathXml.c_str (), G_FILE_TEST_IS_REGULAR))
    throw std::invalid_argument ("Failed to open the XML model file, " + pathXml);
  if (!g_file_test (pathBin.c_str (), G_FILE_TEST_IS_REGULAR))
    throw std::invalid_argument ("Failed to open the BIN model file, " + pathBin);
}

/**
 * @brief Load the given model onto the accelerator
 * @param prop property of tensor_filter instance
 */
void
openvino_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  std::string pathXml;
  std::string pathBin;
  accl_hw accelerator;
  int ret;

  accelerator = getAccelerator (prop);
#ifndef __OPENVINO_CPU_EXT__
  if (accelerator == ACCL_CPU)
    throw std::invalid_argument ("Accelerating via CPU is not supported on the current platform");
#endif

  getModelPaths (prop, pathXml, pathBin);

  tfOv.reset (new TensorFilterOpenvino (pathXml, pathBin));
  ret = tfOv->loadModel (accelerator);
  if (ret != TensorFilterOpenvino::RetSuccess)
    throw std::system_error (-ret, std::generic_category (), "Failed to load the given model");

  /** The framework keeps the tensor info it has once the model is opened, and
   *  takes the one of the model when it has none. */
  if (prop->input_meta.num_tensors > 0)
    gst_tensors_info_copy (&inputInfo, &prop->input_meta);
  else if ((ret = tfOv->getInputTensorDim (&inputInfo)) != TensorFilterOpenvino::RetSuccess)
    throw std::system_error (
        -ret, std::generic_category (), "Failed to get the input tensor info");

  if (prop->output_meta.num_tensors > 0)
    gst_tensors_info_copy (&outputInfo, &prop->output_meta);
  else if ((ret = tfOv->getOutputTensorDim (&outputInfo)) != TensorFilterOpenvino::RetSuccess)
    throw std::system_error (
        -ret, std::generic_category (), "Failed to get the output tensor info");
}

/**
 * @brief Run an inference with the loaded model
 * @param[in] input the array of input tensors
 * @param[out] output the array of output tensors
 */
void
openvino_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  int ret = tfOv->invoke (&inputInfo, &outputInfo, input, output);

  if (ret != TensorFilterOpenvino::RetSuccess)
    throw std::system_error (-ret, std::generic_category (), "Failed to invoke the model");
}

/**
 * @brief Get the information about the framework
 * @param[out] info the information of the framework
 */
void
openvino_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = FALSE;
  info.allocate_in_invoke = FALSE;
  info.run_without_model = FALSE;
  info.verify_model_path = FALSE;
  info.hw_list = hw_list;
  info.num_hw = num_hw;
  info.accl_auto = hw_list[0];
  info.accl_default = hw_list[0];
  info.statistics = nullptr;
}

/**
 * @brief Get the tensor info of the loaded model
 * @param ops the operation to do; only GET_IN_OUT_INFO is supported
 * @param[out] in_info the dimensions and types of the input tensors
 * @param[out] out_info the dimensions and types of the output tensors
 * @return 0 if OK, -ENOENT if the operation is not supported, other negative values if error
 */
int
openvino_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  int ret;

  if (ops != GET_IN_OUT_INFO)
    return -ENOENT;

  ret = tfOv->getInputTensorDim (&in_info);
  if (ret != TensorFilterOpenvino::RetSuccess)
    return ret;

  return tfOv->getOutputTensorDim (&out_info);
}

/**
 * @brief Register the sub-plugin
 */
void
openvino_subplugin::init_filter_openvino ()
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<openvino_subplugin> ();
}

/**
 * @brief Unregister the sub-plugin
 */
void
openvino_subplugin::fini_filter_openvino ()
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

} /* namespace tensorfilter_openvino */
} /* namespace nnstreamer */

/**
 * @brief Initialize this object for tensor_filter sub-plugin runtime register
 */
void
init_filter_openvino (void)
{
  nnstreamer::tensorfilter_openvino::openvino_subplugin::init_filter_openvino ();
}

/**
 * @brief Finalize the subplugin
 */
void
fini_filter_openvino (void)
{
  nnstreamer::tensorfilter_openvino::openvino_subplugin::fini_filter_openvino ();
}
