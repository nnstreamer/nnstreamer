/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2020 Samsung Electronics
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file tensor_filter_trix_engine.cc
 * @date 20 Jan 2020
 * @brief NNStreamer tensor-filter subplugin for TRIx devices
 * @see http://github.com/nnstreamer/nnstreamer
 * @author Dongju Chae <dongju.chae@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <tensor_filter_trix_engine.hh>

using namespace std;

namespace nnstreamer {

void init_filter_trix_engine (void) __attribute__ ((constructor));
void fini_filter_trix_engine (void) __attribute__ ((destructor));

TensorFilterTRIxEngine *TensorFilterTRIxEngine::registered = nullptr;
const char *TensorFilterTRIxEngine::name = "trix-engine";
const accl_hw TensorFilterTRIxEngine::hw_list[] = {ACCL_NPU_SR};
const int TensorFilterTRIxEngine::num_hw = 1;

/**
 * @brief Construct a new TRIx-Engine subplugin instance
 */
TensorFilterTRIxEngine::TensorFilterTRIxEngine ()
    : dev_type_ (NPUCOND_CONN_UNKNOWN),
      dev_ (nullptr),
      model_path_ (nullptr),
      model_meta_ (nullptr),
      model_id_ (0),
      trix_in_info_ (),
      trix_out_info_ () {
  gst_tensors_info_init (addressof (nns_in_info_));
  gst_tensors_info_init (addressof (nns_out_info_));
}

/**
 * @brief Destruct the TRIx-Engine subplugin instance
 */
TensorFilterTRIxEngine::~TensorFilterTRIxEngine () {
  g_free (model_path_);
  g_free (model_meta_);

  gst_tensors_info_free (std::addressof (nns_in_info_));
  gst_tensors_info_free (std::addressof (nns_out_info_));

  if (dev_ != nullptr) {
    unregisterNPUmodel_all (dev_);
    putNPUdevice (dev_);
  }
}

/**
 * @brief Method to get an empty object
 */
tensor_filter_subplugin &
TensorFilterTRIxEngine::getEmptyInstance () {
  return *(new TensorFilterTRIxEngine ());
}

/**
 * @brief Configure TRIx-Engine instance
 */
void
TensorFilterTRIxEngine::configure_instance (const GstTensorFilterProperties *prop) {
  if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
    nns_loge ("Unable to find a model filepath given\n");
    throw invalid_argument ("Unable to find a model filepath given");
  }

  model_path_ = g_strdup (prop->model_files[0]);
  model_meta_ = getNPUmodel_metadata (model_path_, false);
  if (model_meta_ == nullptr) {
    nns_loge ("Unable to extract the model metadata\n");
    throw runtime_error ("Unable to extract the model metadata");
  }

  int status = -ENOENT;
  for (int i = 0; i < prop->num_hw; i++) {
    /* TRIV2 alias for now */
    if (prop->hw_list[i] == ACCL_NPU_SR) {
      status = getNPUdeviceByTypeAny (&dev_, NPUCOND_TRIV2_CONN_SOCIP, 2);
      if (status == 0)
        break;
    }
  }

  if (status != 0) {
    nns_loge ("Unable to find a proper NPU device\n");
    throw runtime_error ("Unable to find a proper NPU device");
  }

  generic_buffer model_file;
  model_file.filepath = model_path_;
  model_file.size = model_meta_->size;
  model_file.type = BUFFER_FILE;

  if (registerNPUmodel (dev_, &model_file, &model_id_) != 0) {
    nns_loge ("Unable to register the model\n");
    throw runtime_error ("Unable to register the model");
  }

  /* check user-provided input tensor info */
  if (prop->input_meta.num_tensors == 0) {
    nns_in_info_.num_tensors = model_meta_->input_seg_num;
    for (uint32_t i = 0; i < nns_in_info_.num_tensors; i++) {
      nns_in_info_.info[i].type = _NNS_UINT8;
      for (uint32_t j = 0; j < NNS_TENSOR_RANK_LIMIT; j++)
        nns_in_info_.info[i].dimension[j] =
            model_meta_->input_seg_dims[i][NNS_TENSOR_RANK_LIMIT - j - 1];
    }
  } else {
    gst_tensors_info_copy (&nns_in_info_, &prop->input_meta);
  }

  /* check user-provided output tensor info */
  if (prop->output_meta.num_tensors == 0) {
    nns_out_info_.num_tensors = model_meta_->output_seg_num;
    for (uint32_t i = 0; i < nns_out_info_.num_tensors; i++) {
      nns_out_info_.info[i].type = _NNS_UINT8;
      for (uint32_t j = 0; j < NNS_TENSOR_RANK_LIMIT; j++)
        nns_out_info_.info[i].dimension[j] =
            model_meta_->output_seg_dims[i][NNS_TENSOR_RANK_LIMIT - j - 1];
    }
  } else {
    gst_tensors_info_copy (&nns_out_info_, &prop->output_meta);
  }

  set_data_info (prop);
}

/**
 * @brief Convert data layout (from NNStreamer to TRIx-Engine)
 */
data_layout
TensorFilterTRIxEngine::convert_data_layout (const tensor_layout &layout) {
  switch (layout) {
    case _NNS_LAYOUT_NHWC:
      return DATA_LAYOUT_NHWC;
    case _NNS_LAYOUT_NCHW:
      return DATA_LAYOUT_NCHW;
    default:
      return DATA_LAYOUT_MODEL;
  }
}

/**
 * @brief Convert data type (from NNStreamer to TRIx-Engine)
 */
data_type
TensorFilterTRIxEngine::convert_data_type (const tensor_type &type) {
  switch (type) {
    case _NNS_INT32:
      return DATA_TYPE_INT32;
    case _NNS_UINT32:
      return DATA_TYPE_UINT32;
    case _NNS_INT16:
      return DATA_TYPE_INT16;
    case _NNS_UINT16:
      return DATA_TYPE_UINT16;
    case _NNS_INT8:
      return DATA_TYPE_INT8;
    case _NNS_UINT8:
      return DATA_TYPE_UINT8;
    case _NNS_FLOAT64:
      return DATA_TYPE_FLOAT64;
    case _NNS_FLOAT32:
      return DATA_TYPE_FLOAT32;
    case _NNS_INT64:
      return DATA_TYPE_INT64;
    case _NNS_UINT64:
      return DATA_TYPE_UINT64;
    default:
      return DATA_TYPE_MODEL;
  }
}

/**
 * @brief Set data info of input/output tensors using metadata
 */
void
TensorFilterTRIxEngine::set_data_info (const GstTensorFilterProperties *prop) {
  const tensor_layout *input_layout = &(prop->input_layout[0]);
  const tensor_layout *output_layout = &(prop->output_layout[0]);

  trix_in_info_.num_info = model_meta_->input_seg_num;

  for (uint32_t idx = 0; idx < trix_in_info_.num_info; ++idx) {
    trix_in_info_.info[idx].layout = convert_data_layout (input_layout[idx]);
    trix_in_info_.info[idx].type = convert_data_type (nns_in_info_.info[idx].type);
  }

  trix_out_info_.num_info = model_meta_->output_seg_num;

  for (uint32_t idx = 0; idx < trix_out_info_.num_info; ++idx) {
    trix_out_info_.info[idx].layout = convert_data_layout (output_layout[idx]);
    trix_out_info_.info[idx].type = convert_data_type (nns_out_info_.info[idx].type);
  }
}

/**
 * @brief Feed the tensor data to input buffers before invoke()
 */
void
TensorFilterTRIxEngine::feed_input_data (const GstTensorMemory *input, input_buffers *input_buf) {
  input_buf->num_buffers = model_meta_->input_seg_num;

  for (uint32_t idx = 0; idx < input_buf->num_buffers; ++idx) {
    input_buf->bufs[idx].addr = input[idx].data;
    input_buf->bufs[idx].size = input[idx].size;
    input_buf->bufs[idx].type = BUFFER_MAPPED;
  }
}

/**
 * @brief Extract the tensor data from output buffers after invoke()
 */
void
TensorFilterTRIxEngine::extract_output_data (const output_buffers *output_buf,
                                             GstTensorMemory *output) {
  /* internal logic error */
  assert (output_buf->num_buffers == model_meta_->output_seg_num);

  for (uint32_t idx = 0; idx < output_buf->num_buffers; ++idx) {
    output[idx].data = output_buf->bufs[idx].addr;
    output[idx].size = output_buf->bufs[idx].size;
  }
}

/**
 * @brief Invoke TRIxEngine using input tensors
 */
void
TensorFilterTRIxEngine::invoke (const GstTensorMemory *input, GstTensorMemory *output) {
  int req_id;
  int status;

  status = createNPU_request (dev_, model_id_, &req_id);
  if (status != 0) {
    nns_loge ("Unable to create NPU request with model id (%u): %d", model_id_, status);
    return;
  }

  input_buffers input_buf;
  output_buffers output_buf;
  /* feed input data to npu-engine */
  feed_input_data (input, &input_buf);

  status =
      setNPU_requestData (dev_, req_id, &input_buf, &trix_in_info_, &output_buf, &trix_out_info_);
  if (status != 0) {
    nns_loge ("Unable to create NPU request for model %u", model_id_);
    return;
  }

  status = submitNPU_request (dev_, req_id);
  if (status != 0) {
    nns_loge ("Unable to submit NPU request with id (%u): %d", req_id, status);
    return;
  }
  /* extract output data from npu-engine */
  extract_output_data (&output_buf, output);

  status = removeNPU_request (dev_, req_id);
  if (status != 0) {
    nns_loge ("Unable to remove NPU request with id (%u): %d", req_id, status);
    return;
  }
}

/**
 * @brief Get TRIxEngine framework info.
 */
void
TensorFilterTRIxEngine::getFrameworkInfo (GstTensorFilterFrameworkInfo &info) {
  info.name = name;
  info.allow_in_place = FALSE;
  info.allocate_in_invoke = TRUE;
  info.run_without_model = FALSE;
  info.verify_model_path = TRUE;
  info.hw_list = hw_list;
  info.num_hw = num_hw;
}

/**
 * @brief Get TRIxEngine model info.
 */
int
TensorFilterTRIxEngine::getModelInfo (model_info_ops ops, GstTensorsInfo &in_info,
                                      GstTensorsInfo &out_info) {
  if (ops != GET_IN_OUT_INFO) {
    return -ENOENT;
  }

  gst_tensors_info_copy (addressof (in_info), addressof (nns_in_info_));
  gst_tensors_info_copy (addressof (out_info), addressof (nns_out_info_));
  return 0;
}

/**
 * @brief Method to handle the event
 */
int
TensorFilterTRIxEngine::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data) {
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

/**
 * @brief Register the subpkugin
 */
void
TensorFilterTRIxEngine::init_filter_trix_engine () {
  registered = tensor_filter_subplugin::register_subplugin<TensorFilterTRIxEngine> ();
}

/**
 * @brief Destruct the subplugin
 */
void
TensorFilterTRIxEngine::fini_filter_trix_engine () {
  /* internal logic error */
  assert (registered != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registered);
}

/**
 * @brief Subplugin initializer
 */
void
init_filter_trix_engine () {
  TensorFilterTRIxEngine::init_filter_trix_engine ();
}

/**
 * @brief Subplugin finalizer
 */
void
fini_filter_trix_engine () {
  TensorFilterTRIxEngine::fini_filter_trix_engine ();
}

} /* namespace nnstreamer */
