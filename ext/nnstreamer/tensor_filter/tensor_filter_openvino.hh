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
 * @file    tensor_filter_openvino.hh
 * @date    23 Dec 2019
 * @brief   Tensor_filter subplugin for OpenVino (DLDT).
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (OpenVino) for tensor_filter.
 *
 * @note This header file is only for internal use.
 *
 * To Packagers:
 *
 * This should not to be exposed with the development packages to the application developers.
 */

#ifndef __TENSOR_FILTER_OPENVINO_H__
#define __TENSOR_FILTER_OPENVINO_H__

#include <glib.h>
#include <nnstreamer_plugin_api_filter.h>
#ifdef __OPENVINO_CPU_EXT__
#include <ext_list.hpp>
#endif /* __OPENVINO_CPU_EXT__ */
#include <inference_engine.hpp>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Wrapper class for OpenVino.
 */
class TensorFilterOpenvino
{
  public:
  enum RetVal {
    RetSuccess = 0,
    RetEBusy = -EBUSY,
    RetEInval = -EINVAL,
    RetENoDev = -ENODEV,
    RetEOverFlow = -EOVERFLOW,
  };

  /** @brief Convert an IE tensor data type string to a _nns_tensor_type */
  static tensor_type convertFromIETypeStr (std::string type);
  /** @brief Convert a tensor container in NNS to a tensor container in IE */
  static InferenceEngine::Blob::Ptr convertGstTensorMemoryToBlobPtr (
      const InferenceEngine::TensorDesc tensorDesc,
      const GstTensorMemory *gstTensor, const tensor_type gstType);
  /** @brief Check the given hw has a matching device in devsVector */
  static bool isAcclDevSupported (std::vector<std::string> &devsVector, accl_hw hw);

  /** @brief Construct with the paths to the XML and bin model files */
  TensorFilterOpenvino (std::string path_model_xml, std::string path_model_bin);
  /** @brief Destruct the instance */
  ~TensorFilterOpenvino ();

  /**
   * @brief Load the given neural network into the target device
   * @todo Need to support other acceleration devices
   */
  int loadModel (accl_hw hw);
  /** @brief Check the neural network model is loaded */
  bool isModelLoaded ();
  /** @brief Get the dimensions of the input tensors of the given model */
  int getInputTensorDim (GstTensorsInfo *info);
  /** @brief Get the dimensions of the output tensors of the given model */
  int getOutputTensorDim (GstTensorsInfo *info);
  /** @brief Do inference using Inference Engine of the OpenVino framework */
  int invoke (const GstTensorFilterProperties *prop,
      const GstTensorMemory *input, GstTensorMemory *output);
  /** @brief Get the path where the model file in XML format is located */
  std::string getPathModelXml ();
  /** @brief Set the path where the model file in XML format is located */
  void setPathModelXml (std::string pathXml);
  /** @brief Get the path where the model file in bin format is located */
  std::string getPathModelBin ();
  /** @brief Set the path where the model file in bin format is located */
  void setPathModelBin (std::string pathBin);

  static const std::string extBin;
  static const std::string extXml;

  protected:
  InferenceEngine::InputsDataMap _inputsDataMap;
  InferenceEngine::OutputsDataMap _outputsDataMap;

  private:
  /** @brief Hidden default constructor; the model paths are mandatory */
  TensorFilterOpenvino ();

  InferenceEngine::Core _ieCore;
  InferenceEngine::CNNNetReader _networkReaderCNN;
  InferenceEngine::CNNNetwork _networkCNN;
  InferenceEngine::TensorDesc _inputTensorDescs[NNS_TENSOR_SIZE_LIMIT];
  InferenceEngine::TensorDesc _outputTensorDescs[NNS_TENSOR_SIZE_LIMIT];
  InferenceEngine::ExecutableNetwork _executableNet;
  InferenceEngine::InferRequest _inferRequest;
  static std::map<accl_hw, std::string> _nnsAcclHwToOVDevMap;

  std::string _pathModelXml;
  std::string _pathModelBin;
  bool _isLoaded;
  accl_hw _hw;
};

#endif /* __TENSOR_FILTER_OPENVINO_H__ */
