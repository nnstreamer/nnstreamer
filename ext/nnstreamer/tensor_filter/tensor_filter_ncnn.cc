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
 *    All four are optional. The input shape is taken from the negotiated
 *    pad capabilities when "input" is omitted, and the output shape is
 *    inferred from the model by running it once with a dummy input when
 *    "output" is omitted. ncnn operates on float32 tensors only, so
 *    "inputtype" and "outputtype" default to float32 and no other type is
 *    accepted. When given, the tensor count, the rank, the type and the
 *    tensor size are checked against the model and a mismatch is an error.
 *    They are still mandatory when use_yolo_decoder is enabled, because the
 *    decoded tensor shape is not a shape the model itself produces.
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
 *        The decoder reads a label, a score and 4 corners out of every
 *        detection, so a model whose output rows are narrower than those 6
 *        values is rejected. Detections past max_detection are dropped.
 *        To calculate the max_detection for an input image of size (w, h),
 *        use the formula: (w/32)*(h/32) + (w/16)*(h/16) + (w/8)*(h/8)*3.
 *        See also: https://github.com/nnstreamer/nnstreamer/blob/main/ext/nnstreamer/tensor_decoder/box_properties/yolo.cc#L130
 */

#include <functional>
#include <glib.h>
#include <memory>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <vector>

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
  static void init_filter_ncnn (); /**< Dynamic library constructor helper */
  static void fini_filter_ncnn (); /**< Dynamic library destructor helper */

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
  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */
  bool use_yolo_decoder; /**< Yolo decoder flag to fix output dimension */
  bool input_from_prop; /**< Input metadata came from the "input" property */
  bool output_from_prop; /**< Output metadata came from the "output" property */
  bool yolo_rows_dropped; /**< Detections were already dropped and reported once */

  static ncnn_subplugin *registeredRepresentation;
  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw = 2;

  ncnn::Net net; /**< Model symbol */
  std::vector<ncnn::Mat> input_mats; /**< Matrices of inputs */

  void parseCustomProperties (const GstTensorFilterProperties *prop);
  void setInputInfo (const GstTensorsInfo &info);
  void inferOutputInfo (std::vector<ncnn::Mat> &mats, GstTensorsInfo &info);
  static std::vector<ncnn::Mat> allocMats (GstTensorsInfo &info);
  static void normalizeInfo (GstTensorsInfo &info, size_t num_expected, const char *direction);
  static std::vector<int> getShape (const GstTensorInfo *info);
  static ncnn::Mat allocMat (const GstTensorInfo *info);
  static void copyToMat (const void *src, ncnn::Mat &mat, size_t size);
  static void copyFromMat (const ncnn::Mat &mat, void *dst, size_t size);
};

/**
 * @brief Construct a new ncnn subplugin::ncnn subplugin object
 */
ncnn_subplugin::ncnn_subplugin ()
    : tensor_filter_subplugin (), empty_model (true), use_yolo_decoder (false),
      input_from_prop (false), output_from_prop (false), yolo_rows_dropped (false)
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

  /**
   * Both metadata are optional. What is missing here is resolved later:
   * the input shape from the negotiated pad caps via SET_INPUT_INFO, and
   * the output shape by running the model once with a dummy input.
   */
  input_from_prop = (inputInfo.num_tensors > 0);
  output_from_prop = (outputInfo.num_tensors > 0);

  if (input_from_prop) {
    normalizeInfo (inputInfo, net.input_indexes ().size (), "input");
    input_mats = allocMats (inputInfo);
  }
  if (output_from_prop)
    normalizeInfo (outputInfo, net.output_indexes ().size (), "output");

  if (input_from_prop && !output_from_prop)
    inferOutputInfo (input_mats, outputInfo);

  empty_model = false;
}

/**
 * @brief Convert an nnstreamer tensor dimension into an ncnn matrix shape.
 * @details An nnstreamer dimension ends either at the first 0 or at explicit
 *          trailing 1s, which pad-caps carry and which gst_tensor_dimension_is_equal
 *          treats as equal to no axis at all. An ncnn matrix has no such padding,
 *          so both are trimmed here. The axis order is kept as is, and an ncnn
 *          matrix reads what is left as (w), (w, h), (w, h, c) or (w, h, d, c).
 */
std::vector<int>
ncnn_subplugin::getShape (const GstTensorInfo *info)
{
  std::vector<int> shape;

  for (guint i = 0; i < NNS_TENSOR_RANK_LIMIT && info->dimension[i]; i++)
    shape.push_back ((int) info->dimension[i]);
  while (shape.size () > 1 && shape.back () == 1)
    shape.pop_back ();

  if (shape.empty () || shape.size () > 4)
    throw std::invalid_argument ("ncnn subplugin supports only up to 4 ranks and does not support tensors of "
                                 + std::to_string (shape.size ()) + " dimensions.");

  return shape;
}

/**
 * @brief Validate tensors metadata against the model and fill in the defaults.
 */
void
ncnn_subplugin::normalizeInfo (GstTensorsInfo &info, size_t num_expected, const char *direction)
{
  if (info.num_tensors != num_expected)
    throw std::invalid_argument (
        std::string ("Wrong number of ") + direction + " matrices"
        + ": Found in argument = " + std::to_string (info.num_tensors)
        + ", Found in model file = " + std::to_string (num_expected));

  for (guint i = 0; i < info.num_tensors; i++) {
    GstTensorInfo *each = gst_tensors_info_get_nth_info (&info, i);

    if (each->type == _NNS_END)
      each->type = _NNS_FLOAT32;
    if (each->type != _NNS_FLOAT32)
      throw std::invalid_argument (
          std::string ("ncnn handles float32 tensors only, but the given ")
          + direction + " tensor " + std::to_string (i) + " is "
          + gst_tensor_get_type_string (each->type) + ".");

    getShape (each);
  }
}

/**
 * @brief Set the input tensors metadata and update the output accordingly.
 * @details Everything is prepared on the side and only swapped in once it has
 *          all succeeded, so that a rejected candidate cannot leave the
 *          instance holding an input shape that its output no longer matches.
 */
void
ncnn_subplugin::setInputInfo (const GstTensorsInfo &info)
{
  GstTensorsInfo given, inferred;
  std::vector<ncnn::Mat> mats;

  gst_tensors_info_init (std::addressof (given));
  gst_tensors_info_init (std::addressof (inferred));
  gst_tensors_info_copy (std::addressof (given), std::addressof (info));

  try {
    normalizeInfo (given, net.input_indexes ().size (), "input");

    /* negotiation retries the same info, so infer only on a change */
    if (gst_tensors_info_is_equal (std::addressof (inputInfo), std::addressof (given))
        && outputInfo.num_tensors > 0) {
      gst_tensors_info_free (std::addressof (given));
      return;
    }

    mats = allocMats (given);
    if (!output_from_prop)
      inferOutputInfo (mats, inferred);
  } catch (...) {
    gst_tensors_info_free (std::addressof (given));
    gst_tensors_info_free (std::addressof (inferred));
    throw;
  }

  gst_tensors_info_free (std::addressof (inputInfo));
  inputInfo = given;
  input_mats = std::move (mats);

  if (!output_from_prop) {
    gst_tensors_info_free (std::addressof (outputInfo));
    outputInfo = inferred;
  }
}

/**
 * @brief Allocate the matrices that the given tensors are fed through.
 */
std::vector<ncnn::Mat>
ncnn_subplugin::allocMats (GstTensorsInfo &info)
{
  std::vector<ncnn::Mat> mats;

  for (guint i = 0; i < info.num_tensors; i++)
    mats.push_back (allocMat (gst_tensors_info_get_nth_info (std::addressof (info), i)));

  return mats;
}

/**
 * @brief Infer the output tensors metadata by running the model once.
 * @details The ncnn model format does not always carry blob shape hints, so
 *          the shapes are read back from a single inference with a zeroed
 *          input instead. This runs at configuration time only.
 */
void
ncnn_subplugin::inferOutputInfo (std::vector<ncnn::Mat> &mats, GstTensorsInfo &info)
{
  const std::vector<int> &input_indexes = net.input_indexes ();
  const std::vector<int> &output_indexes = net.output_indexes ();

  if (use_yolo_decoder)
    throw std::invalid_argument (
        "The \"output\" and \"outputtype\" properties are mandatory when use_yolo_decoder is enabled, "
        "because the decoded bounding boxes do not have a shape that the model itself produces.");

  if (output_indexes.size () > NNS_TENSOR_SIZE_LIMIT)
    throw std::invalid_argument (
        "The model has " + std::to_string (output_indexes.size ()) + " output layers, which exceeds the nnstreamer limit of "
        + std::to_string (NNS_TENSOR_SIZE_LIMIT) + ".");

  ncnn::Extractor ex = net.create_extractor ();
  for (size_t i = 0; i < mats.size (); i++) {
    mats[i].fill (0.0f);
    ex.input (input_indexes.at (i), mats[i]);
  }

  info.num_tensors = (unsigned int) output_indexes.size ();

  for (guint i = 0; i < info.num_tensors; i++) {
    ncnn::Mat out;
    GstTensorInfo *each;

    if (ex.extract (output_indexes.at (i), out) != 0 || out.empty ())
      throw std::invalid_argument (
          "Failed to infer the shape of the output tensor " + std::to_string (i)
          + " from the model. Set the \"output\" and \"outputtype\" properties explicitly.");

    each = gst_tensors_info_get_nth_info (std::addressof (info), i);
    each->type = _NNS_FLOAT32;
    each->dimension[0] = (uint32_t) out.w;
    each->dimension[1] = (uint32_t) out.h;
    if (out.dims >= 3)
      each->dimension[2] = (uint32_t) ((out.dims == 4) ? out.d : out.c);
    if (out.dims == 4)
      each->dimension[3] = (uint32_t) out.c;
  }
}

/**
 * @brief Allocate an ncnn matrix shaped after the given tensor metadata.
 */
ncnn::Mat
ncnn_subplugin::allocMat (const GstTensorInfo *info)
{
  const std::vector<int> shape = getShape (info);

  switch (shape.size ()) {
    case 1:
      return ncnn::Mat (shape[0]);
    case 2:
      return ncnn::Mat (shape[0], shape[1]);
    case 3:
      return ncnn::Mat (shape[0], shape[1], shape[2]);
    default:
      return ncnn::Mat (shape[0], shape[1], shape[2], shape[3]);
  }
}

/**
 * @brief Copy a plain tensor buffer into an ncnn matrix.
 * @details An ncnn matrix pads every channel plane up to its cstep, which is
 *          not necessarily w*h*d, so the planes are copied one by one. A flat
 *          copy of mat.total() elements would both overrun the tensor buffer
 *          and shift every plane but the first.
 */
void
ncnn_subplugin::copyToMat (const void *src, ncnn::Mat &mat, size_t size)
{
  const size_t plane = (size_t) mat.w * mat.h * mat.d * mat.elemsize;
  const char *in = (const char *) src;

  if (plane * mat.c != size)
    throw std::runtime_error (
        "The model takes " + std::to_string (plane * mat.c)
        + " bytes while the given input tensor is " + std::to_string (size)
        + " bytes. Check the \"input\" and \"inputtype\" properties.");

  for (int i = 0; i < mat.c; i++)
    memcpy (mat.channel (i).data, in + plane * i, plane);
}

/**
 * @brief Copy the contents of an ncnn matrix into a plain tensor buffer.
 */
void
ncnn_subplugin::copyFromMat (const ncnn::Mat &mat, void *dst, size_t size)
{
  const size_t plane = (size_t) mat.w * mat.h * mat.d * mat.elemsize;
  char *out = (char *) dst;

  if (plane * mat.c != size)
    throw std::runtime_error (
        "The model produced " + std::to_string (plane * mat.c)
        + " bytes while the configured output tensor is " + std::to_string (size)
        + " bytes. Check the \"output\" and \"outputtype\" properties.");

  for (int i = 0; i < mat.c; i++)
    memcpy (out + plane * i, mat.channel (i).data, plane);
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

  /* push the input tensors to the network */
  const std::vector<int> &input_indexes = net.input_indexes ();
  for (guint i = 0; i < inputInfo.num_tensors; i++) {
    copyToMat (input[i].data, input_mats.at (i), input[i].size);
    ex.input (input_indexes.at (i), input_mats.at (i));
  }

  const std::vector<int> &output_indexes = net.output_indexes ();
  for (guint i = 0; i < outputInfo.num_tensors; i++) {
    ncnn::Mat out;

    if (ex.extract (output_indexes.at (i), out) != 0)
      throw std::runtime_error ("Failed to extract the output tensor "
                                + std::to_string (i) + " from the ncnn network.");

    if (!use_yolo_decoder) {
      copyFromMat (out, output[i].data, output[i].size);
      continue;
    }

    /* write detection-box infos to the output tensor */
    const int label_count = gst_tensors_info_get_nth_info (&outputInfo, i)->dimension[0];
    const int max_rows = (int) (output[i].size / (label_count * sizeof (float)));
    float *output_data = (float *) output[i].data;

    if (out.w < 6)
      throw std::runtime_error ("The yolo decoder reads a label, a score and 4 corners out of every detection, "
                                "but the model produced only "
                                + std::to_string (out.w) + " values per row.");

    if (out.h > max_rows && !yolo_rows_dropped) {
      yolo_rows_dropped = true;
      ml_logw ("The model detected %d objects while the output tensor holds %d of them. "
               "Raise max_detection in the \"output\" property to keep them all.",
          out.h, max_rows);
    }

    memset (output_data, 0, output[i].size);

    for (int j = 0; j < out.h && j < max_rows; j++) {
      float *values = out.row (j);
      const int label = (int) values[0];

      values[2] = fmaxf (fminf (values[2], 1.0), 0.0);
      values[3] = fmaxf (fminf (values[3], 1.0), 0.0);
      values[4] = fmaxf (fminf (values[4], 1.0), 0.0);
      values[5] = fmaxf (fminf (values[5], 1.0), 0.0);

      output_data[0] = (values[2] + values[4]) / 2;
      output_data[1] = (values[3] + values[5]) / 2;
      output_data[2] = values[4] - values[2];
      output_data[3] = values[5] - values[3];
      output_data[4] = values[1];
      if (label >= 0 && label + 5 < label_count)
        output_data[5 + label] = 1;
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
  info.allow_in_place = FALSE;
  info.allocate_in_invoke = FALSE;
  info.run_without_model = FALSE;
  info.verify_model_path = TRUE;
  info.hw_list = hw_list;
  info.num_hw = num_hw;
  info.accl_auto = ACCL_CPU;
  info.accl_default = ACCL_CPU;
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
      /**
       * Only the "input" property fixes the input shape. Anything resolved
       * through SET_INPUT_INFO is a negotiation candidate that may still
       * change, so report it as unknown and let the caller drive.
       */
      if (!input_from_prop || outputInfo.num_tensors == 0)
        return -ENOENT;
      gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
      gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
      break;
    case SET_INPUT_INFO:
      try {
        setInputInfo (in_info);
      } catch (const std::exception &e) {
        ml_loge ("Failed to set the input tensor info of the ncnn subplugin: %s",
            e.what ());
        return -EINVAL;
      }
      gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
      break;
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

ncnn_subplugin *ncnn_subplugin::registeredRepresentation = nullptr;
const char *ncnn_subplugin::name = "ncnn";
const accl_hw ncnn_subplugin::hw_list[] = { ACCL_CPU, ACCL_GPU };

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
