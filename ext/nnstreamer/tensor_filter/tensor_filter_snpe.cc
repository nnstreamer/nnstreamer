/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for SNPE
 * Copyright (C) 2020 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 */
/**
 * @file	tensor_filter_snpe.cc
 * @date	24 Apr 2020
 * @brief	NNStreamer tensor-filter sub-plugin for SNPE (Qualcomm Neural Processing SDK)
 * @see		http://github.com/nnstreamer/nnstreamer
 * @see		https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
 * @author	Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug		No known bugs except for NYI items
 * 
 * This is the per-NN-framework plugin (SNPE) for tensor_filter.
 * 
 * @todo This supports only ITensor for input. Do support IUserBuffer.
 * @todo This supports float32 input output only. Do support Tf8 using IUserBuffer.
 * @todo This supports only CPU runtime on linux-x86_64. Do support others.
 */

#include <iostream>
#include <string>

#include <nnstreamer_log.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <tensor_common.h>
#include <glib.h>

#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/RuntimeList.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlSystem/TensorMap.hpp>

namespace nnstreamer {
namespace tensor_filter_snpe {

extern "C" {
  void _init_filter_snpe (void) __attribute__ ((constructor));
  void _fini_filter_snpe (void) __attribute__ ((destructor));
}

class snpe_subplugin final : public tensor_filter_subplugin {
private:
  bool empty_model;
  char *model_path; /**< The model *.dlc file */
  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */

  zdl::DlSystem::RuntimeList runtime_list;
  std::unique_ptr<zdl::DlContainer::IDlContainer> container;
  std::unique_ptr<zdl::SNPE::SNPE> snpe;

  zdl::DlSystem::TensorMap input_tensor_map;
  zdl::DlSystem::TensorMap output_tensor_map;
  std::vector<std::unique_ptr <zdl::DlSystem::ITensor>> input_tensors;
  
  static const char *name;
  static snpe_subplugin *registeredRepresentation;

  void cleanup ();
  static void setTensorProp (GstTensorsInfo & tensor_meta,
      zdl::DlSystem::TensorMap & tensor_map);

public:
  static void init_filter_snpe ();
  static void fini_filter_snpe ();

  snpe_subplugin ();
  ~snpe_subplugin ();

  tensor_filter_subplugin & getEmptyInstance();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info,
      GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *snpe_subplugin::name = "snpe";

snpe_subplugin::snpe_subplugin () :
    tensor_filter_subplugin (),
    empty_model (true),
    model_path (nullptr),
    runtime_list (zdl::DlSystem::Runtime_t::CPU),
    container (nullptr),
    snpe (nullptr)
{
  inputInfo.num_tensors = 0;
  outputInfo.num_tensors = 0;
  input_tensors.reserve (NNS_TENSOR_RANK_LIMIT);
}

void snpe_subplugin::cleanup ()
{
  if (empty_model)
    return;
  
  if (container) {
    container = nullptr;
  }

  if (snpe) {
    snpe.reset ();
    snpe = nullptr;
  }

  if (model_path)
    delete model_path;

  runtime_list.clear ();
  input_tensors.clear ();
  input_tensor_map.clear ();
  output_tensor_map.clear ();

  model_path = nullptr;
  inputInfo.num_tensors = 0;
  outputInfo.num_tensors = 0;
  empty_model = true;
}

snpe_subplugin::~snpe_subplugin ()
{
  cleanup ();
}

tensor_filter_subplugin & snpe_subplugin::getEmptyInstance ()
{
  return *(new snpe_subplugin());
}

void snpe_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  nns_logi ("SNPE Version: %s",
    zdl::SNPE::SNPEFactory::getLibraryVersion ().asString ().c_str ());

  if (!empty_model) {
    /* Already opend */

    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      std::cerr << "Model path is not given." << std::endl;
      throw std::invalid_argument ("Model path is not given.");
    }

    cleanup ();
  }

  assert (model_path == nullptr);

  model_path = g_strdup (prop->model_files[0]);

  container = zdl::DlContainer::IDlContainer::open (model_path);
  
  zdl::SNPE::SNPEBuilder snpe_builder (container.get());
  snpe_builder.setOutputLayers ({});
  snpe_builder.setUseUserSuppliedBuffers (false);
  snpe_builder.setInitCacheMode (false);
  snpe_builder.setRuntimeProcessorOrder (runtime_list);

  snpe = snpe_builder.build ();
  if (snpe == nullptr) {
    nns_loge ("fail to build snpe");
  }

  const zdl::DlSystem::Optional<zdl::DlSystem::StringList> &strList_opt = 
      snpe->getInputTensorNames ();
 
  assert (strList_opt);

  const zdl::DlSystem::StringList &strList = *strList_opt;

  for (size_t i = 0; i < strList.size (); ++i) {
    const zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> &inputDims_opt = 
      snpe->getInputDimensions (strList.at (i));
    const zdl::DlSystem::TensorShape &input_shape = *inputDims_opt;

    input_tensors.emplace_back (zdl::SNPE::SNPEFactory::getTensorFactory ().createTensor (input_shape));
    input_tensor_map.add (strList.at (i), input_tensors[i].get ());
  }

  /* do execution for get info of output tensors */
  snpe->execute (input_tensor_map, output_tensor_map);

  setTensorProp (inputInfo, input_tensor_map);
  setTensorProp (outputInfo, output_tensor_map);

  output_tensor_map.clear ();

  empty_model = false;
}

void snpe_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  assert (!empty_model);
  assert (snpe);

  /* Configure inputs */
  for (unsigned int i = 0; i < inputInfo.num_tensors; ++i) { 
    float *finput = (float *) input[i].data;
    size_t fsize =  input_tensors[i].get ()->getSize ();
    std::copy (finput, finput + fsize, input_tensors[i].get ()->begin ());
  }

  output_tensor_map.clear ();
  snpe->execute (input_tensor_map, output_tensor_map);

  for (unsigned int i = 0; i < outputInfo.num_tensors; ++i) {
    zdl::DlSystem::ITensor *output_tensor = 
        output_tensor_map.getTensor (output_tensor_map.getTensorNames ().at (i));
    float *foutput = (float *) output[i].data;
    std::copy (output_tensor->cbegin (), output_tensor->cend (), foutput);
  }

}

void snpe_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
}

int snpe_subplugin::getModelInfo (model_info_ops ops,
    GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info),
        std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info),
        std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

int snpe_subplugin::eventHandler (event_ops ops,
    GstTensorFilterFrameworkEventData &data)
{
  return -ENOENT;
}

void snpe_subplugin::setTensorProp (GstTensorsInfo & tensor_meta,
    zdl::DlSystem::TensorMap & tensor_map)
{
  tensor_meta.num_tensors = tensor_map.size ();
  for (unsigned int i = 0; i < tensor_map.size (); ++i) {
    tensor_meta.info[i].name = g_strdup (tensor_map.getTensorNames ().at (i));
    tensor_meta.info[i].type = _NNS_FLOAT32;

    unsigned int rank = 
      tensor_map.getTensor (tensor_meta.info[i].name)->getShape ().rank ();
    for (unsigned int j = 0; j < rank; ++j) {
      tensor_meta.info[i].dimension[j] = 
        tensor_map.getTensor (tensor_meta.info[i].name)->getShape () [rank - j - 1];
    }
    for (unsigned int j = rank; j < NNS_TENSOR_RANK_LIMIT; ++j) {
      tensor_meta.info[i].dimension[j] = 1;
    }
  }
}

snpe_subplugin *snpe_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void snpe_subplugin::init_filter_snpe (void)
{
  registeredRepresentation =
      tensor_filter_subplugin::register_subplugin<snpe_subplugin> ();
}

/** @brief Destruct the subplugin */
void snpe_subplugin::fini_filter_snpe (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

void _init_filter_snpe ()
{
  if (nnstreamer_filter_find ("snap")) {
    nns_loge ("Cannot use SNPE and SNAP both. Won't register this SNPE subplugin.");
    return;
  }
  snpe_subplugin::init_filter_snpe ();
}

void _fini_filter_snpe ()
{
  snpe_subplugin::fini_filter_snpe ();
}

} /* namespace nnstreamer::tensor_filter_snpe */
} /* namespace nnstreamer */
