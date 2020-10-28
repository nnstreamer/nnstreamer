/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter, Mediapipe Module
 * Copyright (C) 2020 HyoungJoo Ahn <hello.ahn@samsung.com>
 */
/**
 * @file   tensor_filter_mediapipe.cc
 * @date   06 May 2020
 * @brief  Mediapipe module for tensor_filter gstreamer plugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @bug    No known bugs except for NYI items
 * @todo   If there is other input type (not video), or multiple inputs, current system cannot handle it.
 *
 * This is the per-NN-framework plugin (mediapipe) for tensor_filter.
 */
#include <iostream>
#include <stdexcept>
#include <string>

#include <nnstreamer_cppplugin_api_filter.hh>
#include <tensor_common.h>

#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/port/commandlineflags.h>
#include <mediapipe/framework/port/file_helpers.h>
#include <mediapipe/framework/port/opencv_highgui_inc.h>
#include <mediapipe/framework/port/opencv_imgproc_inc.h>
#include <mediapipe/framework/port/opencv_video_inc.h>
#include <mediapipe/framework/port/parse_text_proto.h>
#include <mediapipe/framework/port/status.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

using nnstreamer::tensor_filter_subplugin;

namespace nnstreamer
{
namespace tensorfilter_mediapipe
{

void _init_filter_mediapipe (void) __attribute__ ((constructor));
void _fini_filter_mediapipe (void) __attribute__ ((destructor));

/**
 * @brief tensor_filter_subplugin concrete class for mediapipe.
 */
class mediapipe_subplugin final : public tensor_filter_subplugin
{
  private:
  gchar *config_path;
  size_t frame_timestamp;

  GstTensorsInfo inputInfo; /**< The tensor info of input tensors */
  GstTensorsInfo outputInfo; /**< The tensor info of output tensors */

  mediapipe::CalculatorGraphConfig config; /**< about .pbtxt file */
  mediapipe::CalculatorGraph graph;

  int loadMediapipeGraph (const GstTensorFilterProperties *prop);
  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw = 0;
  static mediapipe_subplugin *registeredRepresentation;

  public:
  static void init_filter_mediapipe ();
  static void fini_filter_mediapipe ();
  static void inputPtrDeleter (uint8_t *a);

  mediapipe_subplugin ();
  ~mediapipe_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *mediapipe_subplugin::name = "mediapipe";
const accl_hw mediapipe_subplugin::hw_list[] = {};

/**
 * @brief	mediapipe_subplugin Constructor
 */
mediapipe_subplugin::mediapipe_subplugin ()
    : tensor_filter_subplugin (), config_path (nullptr)
{
  gst_tensors_info_init (&inputInfo);
  gst_tensors_info_init (&outputInfo);

  frame_timestamp = 0;
}

/**
 * @brief	mediapipe_subplugin Destructor
 */
mediapipe_subplugin::~mediapipe_subplugin ()
{
  mediapipe::Status status;

  g_free (config_path);

  for (unsigned int i = 0; i < inputInfo.num_tensors; i++) {
    status = graph.CloseInputStream (inputInfo.info[i].name);
    if (!status.ok ()) {
      std::cerr << "Failed to close input stream" << std::endl;
    }
  }

  gst_tensors_info_free (&inputInfo);
  gst_tensors_info_free (&outputInfo);

  status = graph.WaitUntilDone ();
  if (!status.ok ()) {
    std::cerr << "Failed to closing mediapipe graph" << std::endl;
  }
}

/**
 * @brief	return empty instance
 * @return an empty instance
 */
tensor_filter_subplugin &
mediapipe_subplugin::getEmptyInstance ()
{
  return *(new mediapipe_subplugin ());
}

/**
 * @brief	configure the instance before run it
 * @param prop: the properties of the instance.
 *              the properties are written with the construction of the pipeline
 */
void
mediapipe_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
    std::cerr << "Model path is not given." << std::endl;
    throw std::invalid_argument ("Model path is not given.");
  }
  assert (config_path == nullptr);
  config_path = g_strdup (prop->model_files[0]);

  gst_tensors_info_copy (&inputInfo, &prop->input_meta);
  gst_tensors_info_copy (&outputInfo, &prop->output_meta);

  if (loadMediapipeGraph (prop)) {
    std::cerr << "Failed to load mediapipe graph" << std::endl;
    throw std::runtime_error ("Failed to load mediapipe graph");
  }
}

/**
 * @brief	The mendatory delete function of mediapipe data pointer
 */
void
mediapipe_subplugin::inputPtrDeleter (uint8_t *a)
{
  /* do nothing */
  return;
}

/**
 * @brief	load the mp model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 *        -1 if the pbtxt file is not regular.
 *        -2 if the pbtxt file is not loaded properly.
 *        -3 if importing pbtxt file is failed at mediapipe.
 *        -4 if init graph is failed.
 */
int
mediapipe_subplugin::loadMediapipeGraph (const GstTensorFilterProperties *prop)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  gsize file_size;
  gchar *content = nullptr;
  GError *file_error = nullptr;
  mediapipe::Status status;

  assert (config_path != nullptr);

  if (!g_file_test (config_path, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("the file of config_path (%s) is not valid (not regular)\n", config_path);
    return -1;
  }

  if (!g_file_get_contents (config_path, &content, &file_size, &file_error)) {
    ml_loge ("Error reading pbtxt file!! - %s", file_error->message);
    g_clear_error (&file_error);
    return -2;
  }

  std::string calculator_graph_config_contents;
  status = mediapipe::file::GetContents (config_path, &calculator_graph_config_contents);
  if (!status.ok ()) {
    ml_loge ("Failed to GetContents of pbtxt file: %s", config_path);
    return -3;
  }
  config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig> (
      calculator_graph_config_contents);

  status = graph.Initialize (config);
  if (!status.ok ()) {
    ml_loge ("Failed to Initialize Mediapipe Graph with the config file: %s", config_path);
    return -4;
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Mediapipe is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	run the mediapipe graph
 * @todo there are few points which can help to improve the performance.
 */
void
mediapipe_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  int input_width = inputInfo.info[0].dimension[1];
  int input_height = inputInfo.info[0].dimension[2];
  int input_channels = inputInfo.info[0].dimension[0];
  int input_widthStep = input_width * input_channels;
  mediapipe::Status status;

  /* TODO to make it better, start the graph at init or previous step */
  mediapipe::OutputStreamPoller poller
      = graph.AddOutputStreamPoller (outputInfo.info[0].name).ValueOrDie ();
  status = graph.StartRun ({});
  if (!status.ok ()) {
    std::cerr << "Fail to start mediapipe graph" << std::endl;
    throw std::runtime_error ("Fail to start mediapipe graph");
  }

  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame> (
      mediapipe::ImageFormat::SRGB, input_width, input_height, input_widthStep,
      (uint8_t *)input->data, inputPtrDeleter /* do nothing */
      );

  // Send image packet
  status = graph.AddPacketToInputStream (inputInfo.info[0].name,
      mediapipe::Adopt (input_frame.release ()).At (mediapipe::Timestamp (frame_timestamp++)));
  if (!status.ok ()) {
    std::cerr << "Failed to add input packet" << std::endl;
    throw std::runtime_error ("Failed to add input packet");
  }

  // Get the graph result packet, or stop if that fails.
  mediapipe::Packet packet;
  if (!poller.Next (&packet)) {
    std::cerr << "Failed to get output packet from mediapipe graph" << std::endl;
    throw std::runtime_error ("Failed to get output packet from mediapipe graph");
  }
  auto &output_frame = packet.Get<mediapipe::ImageFrame> ();
  cv::Mat output_frame_mat = mediapipe::formats::MatView (&output_frame);

  /* TODO remove memcpy if it's possible */
  memcpy (output->data, output_frame_mat.data, output->size);

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Run () is finished: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
}

/**
 * @brief describe the subplugin's setting
 */
void
mediapipe_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = FALSE;
  info.allocate_in_invoke = FALSE;
  info.run_without_model = FALSE;
  info.verify_model_path = TRUE;
  info.hw_list = hw_list;
  info.num_hw = num_hw;
}

/**
 * @brief there is no model info available from the mediapipe.
 *        For this reason, the acquired properties gotten at configuration will be filled.
 * @return 0 if OK. non-zero if error.
 */
int
mediapipe_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops != GET_IN_OUT_INFO) {
    return -ENOENT;
  }

  gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
  gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
  return 0;
}

/**
 * @brief tensor-filter-subplugin eventHandler API.
 */
int
mediapipe_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  return -ENOENT;
}

mediapipe_subplugin *mediapipe_subplugin::registeredRepresentation = nullptr;

/**
 * @brief Initialize this object for tensor_filter subplugin runtime register
 */
void
mediapipe_subplugin::init_filter_mediapipe (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<mediapipe_subplugin> ();
}

/**
 * @brief Register mediapipe subplugin
 */
void
_init_filter_mediapipe ()
{
  mediapipe_subplugin::init_filter_mediapipe ();
}

/**
 * @brief Destruct the subplugin
 */
void
mediapipe_subplugin::fini_filter_mediapipe (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief fin
 */
void
_fini_filter_mediapipe ()
{
  mediapipe_subplugin::fini_filter_mediapipe ();
}

} /* namespace nnstreamer::tensorfilter_mediapipe */
} /* namespace nnstreamer */
