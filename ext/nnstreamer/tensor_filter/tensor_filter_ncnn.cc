/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    tensor_filter_ncnn.cc
 * @date    13 Sep 2022
 * @brief   NNStreamer tensor-filter sub-plugin for Tencent NCNN
 * @author  Sungbin Jo <goranmoomin@daum.net>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     Nothing works.
 *
 * This is the NCNN plugin for tensor_filter.
 */

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>
#include <thread>

#include <ncnn/net.h>

namespace nnstreamer
{
namespace tensorfilter_ncnn
{

G_BEGIN_DECLS

void init_filter_ncnn (void) __attribute__ ((constructor));
void fini_filter_ncnn (void) __attribute__ ((destructor));

G_END_DECLS

/**
 * @brief Class for NCNN subplugin.
 */
class ncnn_subplugin final : public tensor_filter_subplugin
{
  private:
  bool empty_model;
  gchar *model_path;
  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;

  static const char *name;
  static ncnn_subplugin *registeredRepresentation;

  ncnn::Net net;
  std::vector<ncnn::Mat> input_mats;
  std::vector<ncnn::Mat> output_mats;

  public:
  static void init_filter_ncnn ();
  static void fini_filter_ncnn ();

  ncnn_subplugin ();
  ~ncnn_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *ncnn_subplugin::name = "ncnn";
/**
 * @brief Construct a new ncnn subplugin::ncnn subplugin object
 */
ncnn_subplugin::ncnn_subplugin ()
    : tensor_filter_subplugin (), model_path (nullptr)
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));
  ml_logw ("Hello world from %s", __func__);
}

/**
 * @brief Destroy the ncnn subplugin::ncnn subplugin object
 */
ncnn_subplugin::~ncnn_subplugin ()
{
  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));

  if (empty_model)
    return;

  g_free (model_path);
  model_path = nullptr;
  empty_model = true;
}

/**
 * @brief Method to get an empty object
 */
tensor_filter_subplugin &
ncnn_subplugin::getEmptyInstance ()
{
  return *(new ncnn_subplugin ());
}

/**
 * @brief Configure ncnn instance
 */
void
ncnn_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  gst_tensors_info_copy (std::addressof (inputInfo), std::addressof (prop->input_meta));
  gst_tensors_info_copy (std::addressof (outputInfo), std::addressof (prop->output_meta));
  // ml_logw ("%d %d", outputInfo.format, prop->output_meta.format);
  // outputInfo.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  net.opt.use_vulkan_compute = true;

  if (prop->num_models == 1) {
    if (net.load_param_bin (prop->model_files[0]))
      throw std::invalid_argument (
          "Failed to open the file " + std::string (prop->model_files[0]));
  } else if (prop->num_models == 2) {
    if (net.load_param (prop->model_files[0]))
      throw std::invalid_argument (
          "Failed to open the file " + std::string (prop->model_files[0]));
    if (net.load_model (prop->model_files[1]))
      throw std::invalid_argument (
          "Failed to open the file " + std::string (prop->model_files[1]));
  } else {
    throw std::invalid_argument ("Number of model files must be 1 or 2");
  }

  ml_logw ("model loaded from tensor_filter_ncnn");

  const std::vector<int> &input_indexes = net.input_indexes ();
  input_mats.clear ();
  for (int i = 0; i < (int) input_indexes.size (); i++) {
    const uint32_t *dim = inputInfo.info[i].dimension;
    std::vector<int> shape;
    while (*dim)
      shape.push_back (*dim++);
    ncnn::Mat in;
    switch (shape.size ()) {
      case 1:
        in = ncnn::Mat (shape[0]);
        break;
      case 2:
        in = ncnn::Mat (shape[0], shape[1]);
        break;
      case 3:
        in = ncnn::Mat (shape[0], shape[1], shape[2]);
        break;
      case 4:
        in = ncnn::Mat (shape[0], shape[1], shape[2], shape[3]);
        break;
      default:
        throw std::invalid_argument ("Wrong input dimension");
    }
    input_mats.push_back (in);
  }

  const std::vector<int> &output_indexes = net.output_indexes ();
  output_mats.resize (output_indexes.size ());
}


void
extract_thread (ncnn::Extractor &ex, const int idx, ncnn::Mat &out)
{
  ex.extract (idx, out);
}

void
input_thread (ncnn::Extractor &ex, const int idx, const ncnn::Mat &in,
    const void *input_data, const uint32_t num_bytes)
{
  memcpy (in.data, input_data, num_bytes);
  ex.input (idx, in);
}

/**
 * @brief Invoke ncnn instance
 */
void
ncnn_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  ncnn::Extractor ex = net.create_extractor ();

  static int cnt = 0;
  static clock_t start = clock ();
  ml_logw ("FPS : %f", (float) (++cnt) / (clock () - start) * CLOCKS_PER_SEC);

  std::vector<std::thread> input_thrs;
  const std::vector<int> &input_indexes = net.input_indexes ();
  const char *input_data = (const char *) input->data;
  for (int i = 0; i < (int) input_indexes.size (); i++) {
    ncnn::Mat &in = input_mats.at (i);
    const uint32_t num_bytes = (in.elembits () / 8) * in.total ();
    input_thrs.emplace_back (input_thread, std::ref (ex), input_indexes.at (i),
        std::ref (in), input_data, num_bytes);
    input_data += num_bytes;
  }
  for (std::thread &thr : input_thrs)
    thr.join ();

  std::vector<std::thread> output_thrs;
  const std::vector<int> &output_indexes = net.output_indexes ();
  for (int i = 0; i < (int) output_indexes.size (); i++) {
    ncnn::Mat &out = output_mats.at (i);
    output_thrs.emplace_back (
        extract_thread, std::ref (ex), output_indexes.at (i), std::ref (out));
  }
  memset (output->data, 0, output->size);
  for (std::thread &thr : output_thrs)
    thr.join ();

  for (int i = 0; i < (int) output_indexes.size (); i++) {
    ncnn::Mat &out = output_mats.at (i);
    const int label_count = outputInfo.info[i].dimension[0];
    float *output_data = (float *) output->data;
    for (int j = 0; j < out.h; j++) {
      float *values = out.row (j);
      values[2] = fmaxf (fminf (values[2], 1.0), 0.0);
      values[3] = fmaxf (fminf (values[3], 1.0), 0.0);
      values[4] = fmaxf (fminf (values[4], 1.0), 0.0);
      values[5] = fmaxf (fminf (values[5], 1.0), 0.0);

      output_data[0] = (values[2] + values[4]) / 2;
      output_data[1] = (values[3] + values[5]) / 2;
      output_data[2] = values[4] - values[2];
      output_data[3] = values[5] - values[3];
      output_data[4] = values[1];
      output_data[5 + (int) values[0]] = 1;
      output_data += label_count;
    }
  }
}

/**
 * @brief Get ncnn frameworks info
 */
void
ncnn_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
}

/**
 * @brief Get ncnn model information
 */
int
ncnn_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle the event
 */
int
ncnn_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

ncnn_subplugin *ncnn_subplugin::registeredRepresentation = nullptr;

/**
 * @brief Initialize the object for runtime register
 */
void
ncnn_subplugin::init_filter_ncnn (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<ncnn_subplugin> ();
}

/**
 * @brief Destruct the subplugin
 */
void
ncnn_subplugin::fini_filter_ncnn (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief initializer
 */
void
init_filter_ncnn ()
{
  ncnn_subplugin::init_filter_ncnn ();
}

/**
 * @brief finalizer
 */
void
fini_filter_ncnn ()
{
  ncnn_subplugin::fini_filter_ncnn ();
}

} // namespace tensorfilter_ncnn
} /* namespace nnstreamer */
