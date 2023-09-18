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
  UNUSED (prop);
  ml_logw ("Configuring ncnn subplugin instance as hardcoded data");
  inputInfo.num_tensors = 1;
  outputInfo.num_tensors = 2;
  for (unsigned int i = 0; i < inputInfo.num_tensors; ++i) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&inputInfo, i);
    for (int idx = 0; idx < NNS_TENSOR_RANK_LIMIT; ++idx)
      info->dimension[idx] = 0;
    info->dimension[0] = 3;
    info->dimension[1] = 300;
    info->dimension[2] = 300;
    info->dimension[3] = 1;
    info->type = _NNS_FLOAT32;
  }

  for (unsigned int i = 0; i < outputInfo.num_tensors; ++i) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&outputInfo, i);
    for (int idx = 0; idx < NNS_TENSOR_RANK_LIMIT; ++idx)
      info->dimension[idx] = 0;
    if (i == 0) {
      info->dimension[0] = 4;
      info->dimension[1] = 1;
      info->dimension[2] = 1917;
      info->dimension[3] = 1;
    } else if (i == 1) {
      info->dimension[0] = 91;
      info->dimension[1] = 1917;
      info->dimension[2] = 1;
    }
    info->type = _NNS_FLOAT32;
  }
}

/**
 * @brief Invoke ncnn instance
 */
void
ncnn_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  UNUSED (input);
  UNUSED (output);

  ml_logw (__func__);
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
ncnn_subplugin::getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
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
  registeredRepresentation = tensor_filter_subplugin::register_subplugin<ncnn_subplugin> ();
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
