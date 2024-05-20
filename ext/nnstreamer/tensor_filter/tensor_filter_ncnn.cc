/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    tensor_filter_ncnn.cc
 * @date    18 Dec 2023
 * @brief   NNStreamer tensor-filter sub-plugin for Tencent ncnn
 * @author  Sungbin Jo <goranmoomin@daum.net>
 * @author  SangLyul Cho <chosanglyul@gmail.com>
 * @author  Kijun Shin <sharaelong.shin@gmail.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs.
 *
 * This is the ncnn plugin for tensor_filter.
 *
 * @details Usage examples
 *  Case 1: image classification by squeezenet
 *  Case 2: object detection by mobilenetv2-ssdlite
 *
 * @note Special considerations on properties:
 *  input, inputtype, output, outputtype:
 *    It is essential to set these four options correctly.
 *    For assistance in configuring the shape and type,
 *    please refer to the examples provided.
 *
 *  accelerator:
 *    Enable Vulkan acceleration by setting accelerator=true:gpu.
 *    This option is applicable if your device is equipped
 *    with any Vulkan-acceleratable processor.
 *
 *  custom:
 *    Each entries are separated by ','
 *    Each entries have property_key:value format.
 *    There must be no spaces.
 *
 *    Supported custom properties:
 *      use_yolo_decoder (optional, default=false)
 *        Enable this option by setting use_yolo_decoder=true if your model
 *        includes a Yolov3DetectionOutput layer or yolo-related output layers,
 *        especially when dealing with variable output sizes (num_detection, 6).
 *        In such cases, you must also configure
 *        output=(5+num_labels, max_detection, 1) and outputtype=float32.
 *        To calculate the max_detection for an input image of size (w, h),
 *        use the formula: (w/32)*(h/32) + (w/16)*(h/16) + (w/8)*(h/8)*3.
 *        See also: https://github.com/nnstreamer/nnstreamer/blob/main/ext/nnstreamer/tensor_decoder/box_properties/yolo.cc#L130
 */

#include <functional>
#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
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
 * @brief Class for ncnn subplugin.
 */
class ncnn_subplugin final : public tensor_filter_subplugin
{
  public:
  static void init_filter_ncnn (); /**< Dynamic library contstructor helper */
  static void fini_filter_ncnn (); /**< Dynamic library desctructor helper */

  ncnn_subplugin ();
  ~ncnn_subplugin ();

  /**< Implementations of ncnn tensor_filter_subplugin */
  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  private:
  bool empty_model; /**< Empty (not initialized) model flag */
  static const GstTensorFilterFrameworkInfo info; /**< Framework info */
  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */
  bool use_yolo_decoder; /**< Yolo decoder flag to fix output dimension */

  static ncnn_subplugin *registeredRepresentation;

  ncnn::Net net; /**< Model symbol */
  std::vector<ncnn::Mat> input_mats; /**< Matrices of inputs */
  std::vector<ncnn::Mat> output_mats; /**< Matrices of outputs */

  void parseCustomProperties (const GstTensorFilterProperties *prop);
  static void input_thread (ncnn::Extractor &ex, const int idx,
      const ncnn::Mat &in, const void *input_data, const uint32_t num_bytes);
  static void extract_thread (ncnn::Extractor &ex, const int idx,
      ncnn::Mat &out, void *output_data, const uint32_t num_bytes);
};

/**
 * @brief Describe framework information.
 */
const GstTensorFilterFrameworkInfo ncnn_subplugin::info = { .name = "ncnn",
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .verify_model_path = TRUE,
  .hw_list = (const accl_hw[]){ ACCL_CPU, ACCL_GPU },
  .num_hw = 2,
  .accl_auto = ACCL_CPU,
  .accl_default = ACCL_CPU,
  .statistics = nullptr };

/**
 * @brief Construct a new ncnn subplugin::ncnn subplugin object
 */
ncnn_subplugin::ncnn_subplugin () : tensor_filter_subplugin ()
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));
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

  empty_model = true;
}

/**
 * @brief Method to get empty instance of ncnn subplugin.
 */
tensor_filter_subplugin &
ncnn_subplugin::getEmptyInstance ()
{
  return *(new ncnn_subplugin ());
}

/**
 * @brief Configure the instance of the ncnn subplugin.
 */
void
ncnn_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  /* get input / output info from properties */
  gst_tensors_info_copy (std::addressof (inputInfo), std::addressof (prop->input_meta));
  gst_tensors_info_copy (std::addressof (outputInfo), std::addressof (prop->output_meta));

  /* check number of model files */
  if (prop->num_models > 2 || prop->num_models <= 0) {
    throw std::invalid_argument (std::string ("Number of model files must be 1 or 2;")
                                 + " Multiple model is not supported.");
  }

  /* try to parse custom properties of the ncnn_subplugin */
  try {
    /* parse custom properties */
    parseCustomProperties (prop);
  } catch (const std::invalid_argument &e) {
    throw std::invalid_argument (
        "Failed to parse custom property : " + std::string (e.what ()));
  }

  /* decide use vulkan acceleration */
  if (std::find (prop->hw_list, prop->hw_list + prop->num_hw, ACCL_GPU)
      != (prop->hw_list + prop->num_hw)) {
    net.opt.use_vulkan_compute = true;
    g_message ("accl = gpu\n");
  } else {
    net.opt.use_vulkan_compute = false;
  }

  /* load model files */
  /* ncnn returns nonzero value when an error occurs */
  if (prop->num_models == 1) {
    if (net.load_param_bin (prop->model_files[0]))
      throw std::invalid_argument (
          "Failed to open the model file " + std::string (prop->model_files[0]));
  } else {
    if (net.load_param (prop->model_files[0]))
      throw std::invalid_argument (
          "Failed to open the param file " + std::string (prop->model_files[0]));
    if (net.load_model (prop->model_files[1]))
      throw std::invalid_argument (
          "Failed to open the bin file " + std::string (prop->model_files[1]));
  }

  /* get input layers from the ncnn network */
  const std::vector<int> &input_indexes = net.input_indexes ();
  input_mats.clear ();
  if (inputInfo.num_tensors != input_indexes.size ())
    throw std::invalid_argument (
        std::string ("Wrong number of input matrices")
        + ": Found in argument = " + std::to_string (inputInfo.num_tensors)
        + ", Found in model file = " + std::to_string (input_indexes.size ()));

  /* init input matrices */
  for (guint i = 0; i < inputInfo.num_tensors; i++) {
    /* get dimensions of the input matrix from inputInfo */
    const uint32_t *dim = gst_tensors_info_get_nth_info (&inputInfo, i)->dimension;
    std::vector<int> shape;
    while (*dim)
      shape.push_back (*dim++);

    /* make ncnn matrix object */
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
        throw std::invalid_argument ("ncnn subplugin supports only up to 4 ranks and does not support input tensors of "
                                     + std::to_string (shape.size ()) + " dimensions.");
    }
    input_mats.push_back (in);
  }

  /* get output layers from the ncnn network */
  const std::vector<int> &output_indexes = net.output_indexes ();
  output_mats.clear ();
  if (outputInfo.num_tensors != output_indexes.size ())
    throw std::invalid_argument (
        std::string ("Wrong number of output matrices")
        + ": Found in argument = " + std::to_string (outputInfo.num_tensors)
        + ", Found in model file = " + std::to_string (output_indexes.size ()));

  /* init output matrices */
  output_mats.resize (outputInfo.num_tensors);

  empty_model = false;
}

/**
 * @brief Invoke ncnn model and get the inference result.
 */
void
ncnn_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  if (empty_model)
    throw std::runtime_error (
        "Model is empty: the ncnn instance is not configured and "
        "its \"invoke\" method is called. This may be an internal bug of "
        "nnstreamer or ncnn-subplugin unless if you have directly accessed "
        "ncnn-subplugin.");

  /* make extractor instance for each inference */
  ncnn::Extractor ex = net.create_extractor ();

  /* get input layer indices */
  std::vector<std::thread> input_thrs;
  const std::vector<int> &input_indexes = net.input_indexes ();

  /* get input from input tensor and push to the network */
  const char *input_data = (const char *) input->data;
  for (guint i = 0; i < inputInfo.num_tensors; i++) {
    ncnn::Mat &in = input_mats.at (i);
    const uint32_t num_bytes = (in.elembits () / 8) * in.total ();
    input_thrs.emplace_back (input_thread, std::ref (ex), input_indexes.at (i),
        std::ref (in), input_data, num_bytes);
    input_data += num_bytes;
  }

  /* join threads */
  for (std::thread &thr : input_thrs)
    thr.join ();

  /* get output layer indices */
  std::vector<std::thread> output_thrs;
  const std::vector<int> &output_indexes = net.output_indexes ();

  if (use_yolo_decoder) {
    /* get output and store to ncnn matrix */
    for (guint i = 0; i < outputInfo.num_tensors; i++) {
      ncnn::Mat &out = output_mats.at (i);
      output_thrs.emplace_back (extract_thread, std::ref (ex),
          output_indexes.at (i), std::ref (out), nullptr, 0);
    }

    /* memset output to zero and hide latency by multithreading */
    memset (output->data, 0, output->size);

    /* join threads */
    for (std::thread &thr : output_thrs)
      thr.join ();

    /* write detection-box infos to the output tensor */
    for (guint i = 0; i < outputInfo.num_tensors; i++) {
      ncnn::Mat &out = output_mats.at (i);
      const int label_count
          = gst_tensors_info_get_nth_info (&outputInfo, i)->dimension[0];
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
  } else {
    /* get output and store to the output tensor */
    char *output_data = (char *) output->data;
    for (guint i = 0; i < outputInfo.num_tensors; i++) {
      ncnn::Mat &out = output_mats.at (i);
      const uint32_t num_bytes = (out.elembits () / 8) * out.total ();
      output_thrs.emplace_back (extract_thread, std::ref (ex),
          output_indexes.at (i), std::ref (out), output_data, num_bytes);
      output_data += num_bytes;
    }

    /* join threads */
    for (std::thread &thr : output_thrs)
      thr.join ();
  }
}

/**
 * @brief Get ncnn frameworks info
 */
void
ncnn_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info = ncnn_subplugin::info;
}

/**
 * @brief Get ncnn model information
 */
int
ncnn_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  switch (ops) {
    case GET_IN_OUT_INFO:
      gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
      gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
      break;
    case SET_INPUT_INFO:
    default:
      return -ENOENT;
  }
  return 0;
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

/**
 * @brief Parse custom prop and set instance options accordingly.
 */
void
ncnn_subplugin::parseCustomProperties (const GstTensorFilterProperties *prop)
{
  using uniq_g_strv = std::unique_ptr<gchar *, std::function<void (gchar **)>>;
  const char *custom_props = prop->custom_properties;

  /* set default values */
  use_yolo_decoder = false;

  if (custom_props) {
    /* split with , to parse options */
    uniq_g_strv options (g_strsplit (custom_props, ",", -1), g_strfreev);
    guint len = g_strv_length (options.get ());

    for (guint i = 0; i < len; i++) {
      /* split with = to parse single option */
      uniq_g_strv option (g_strsplit (options.get ()[i], ":", -1), g_strfreev);

      /* we only have key=value form option */
      if (g_strv_length (option.get ()) == 2) {
        g_strstrip (option.get ()[0]);
        g_strstrip (option.get ()[1]);

        if (g_ascii_strcasecmp (option.get ()[0], "use_yolo_decoder") == 0) {
          /* true or false (default) only */
          if (g_ascii_strcasecmp (option.get ()[1], "true") == 0) {
            use_yolo_decoder = true;
          } else if (g_ascii_strcasecmp (option.get ()[1], "false") == 0) {
            use_yolo_decoder = false;
          } else {
            throw std::invalid_argument ("Invalid option for use_yolo_decoder: "
                                         + std::string (option.get ()[1]) + ".");
          }
        } else {
          throw std::invalid_argument (
              "Unsupported custom property: " + std::string (option.get ()[0]) + ".");
        }
      } else {
        throw std::invalid_argument (
            "Unsupported custom property: " + std::string (options.get ()[i]) + ".");
      }
    }
  }
}

/**
 * @brief Worker function when inserting inputs to the input layer.
 */
void
ncnn_subplugin::input_thread (ncnn::Extractor &ex, const int idx,
    const ncnn::Mat &in, const void *input_data, const uint32_t num_bytes)
{
  /* copy from the input matrix */
  memcpy (in.data, input_data, num_bytes);

  /* input to the network */
  ex.input (idx, in);
}

/**
 * @brief Worker function when getting result from the output layer.
 */
void
ncnn_subplugin::extract_thread (ncnn::Extractor &ex, const int idx,
    ncnn::Mat &out, void *output_data, const uint32_t num_bytes)
{
  /* output from the network */
  ex.extract (idx, out);

  /* copy to the output matrix */
  if (output_data)
    memcpy (output_data, out.data, num_bytes);
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
  g_assert (registeredRepresentation != nullptr);
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

} /* namespace tensorfilter_ncnn */
} /* namespace nnstreamer */
