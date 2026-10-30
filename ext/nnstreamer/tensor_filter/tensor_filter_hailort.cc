/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for HailoRT
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file    tensor_filter_hailort.cc
 * @date    28 Oct 2026
 * @brief   NNStreamer tensor-filter sub-plugin for Hailo NPUs (HailoRT)
 * @see     http://github.com/nnstreamer/nnstreamer
 * @see     https://github.com/hailo-ai/hailort
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (HailoRT) for tensor_filter.
 * It runs a compiled Hailo model (.hef) on a virtual device with the
 * blocking virtual-stream C API of HailoRT (4.18 or later).
 *
 * Custom properties (comma separated "key:value" pairs):
 *  - input_format:  auto | uint8 | uint16 | float32 (default: auto)
 *  - output_format: auto | uint8 | uint16 | float32 (default: auto)
 *  - group_id:      the virtual device group id. Filters with the same id
 *                   share the physical device through the HailoRT scheduler.
 *                   (default: a unique virtual device per filter)
 *
 * "auto" keeps the data type of the device, which is usually quantized uint8.
 * float32 makes HailoRT (de)quantize the data on the host.
 */

#include <cassert>
#include <cerrno>
#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <hailo/hailort.h>

namespace nnstreamer
{
namespace tensor_filter_hailort
{

G_BEGIN_DECLS

void init_filter_hailort (void) __attribute__ ((constructor));
void fini_filter_hailort (void) __attribute__ ((destructor));

G_END_DECLS

/**
 * @brief tensor-filter-subplugin concrete class for HailoRT
 */
class hailort_subplugin final : public tensor_filter_subplugin
{
  private:
  static hailort_subplugin *registeredRepresentation;
  static const GstTensorFilterFrameworkInfo framework_info;
  static const accl_hw hw_list[];

  hailo_vdevice vdevice; /**< The virtual device running the model */
  hailo_hef hef; /**< The compiled model */
  std::vector<hailo_input_vstream> input_vstreams;
  std::vector<hailo_output_vstream> output_vstreams;
  std::vector<size_t> input_frame_sizes;
  std::vector<size_t> output_frame_sizes;
  bool unusable; /**< A failed transfer may leave a frame of an earlier invoke queued */
  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */

  void cleanup ();
  static void check (hailo_status status, const char *what);
  static hailo_format_type_t parseFormatType (const gchar *value);
  static tensor_type convertType (hailo_format_type_t type);
  static void setTensorInfo (GstTensorInfo *info, const hailo_vstream_info_t &vinfo,
      const hailo_format_t &format, size_t frame_size);

  public:
  static void init_filter_hailort ();
  static void fini_filter_hailort ();

  hailort_subplugin ();
  ~hailort_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

/**
 * @brief The accelerators of the sub-plugin.
 */
const accl_hw hailort_subplugin::hw_list[] = { ACCL_NPU };

/**
 * @brief Describe framework information.
 */
const GstTensorFilterFrameworkInfo hailort_subplugin::framework_info = { .name = "hailort",
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .verify_model_path = TRUE,
  .hw_list = hw_list,
  .num_hw = 1,
  .accl_auto = ACCL_NPU,
  .accl_default = ACCL_NPU,
  .statistics = nullptr };

/**
 * @brief Constructor for HailoRT subplugin.
 */
hailort_subplugin::hailort_subplugin ()
    : tensor_filter_subplugin (), vdevice (nullptr), hef (nullptr), unusable (false)
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));
}

/**
 * @brief Destructor for HailoRT subplugin.
 */
hailort_subplugin::~hailort_subplugin ()
{
  cleanup ();
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
hailort_subplugin::getEmptyInstance ()
{
  return *(new hailort_subplugin ());
}

/**
 * @brief Release the virtual streams, the model and the virtual device.
 */
void
hailort_subplugin::cleanup ()
{
  if (!output_vstreams.empty ())
    hailo_release_output_vstreams (output_vstreams.data (), output_vstreams.size ());
  if (!input_vstreams.empty ())
    hailo_release_input_vstreams (input_vstreams.data (), input_vstreams.size ());
  output_vstreams.clear ();
  input_vstreams.clear ();
  input_frame_sizes.clear ();
  output_frame_sizes.clear ();

  if (hef) {
    hailo_release_hef (hef);
    hef = nullptr;
  }

  /* The configured network group is owned by the virtual device. */
  if (vdevice) {
    hailo_release_vdevice (vdevice);
    vdevice = nullptr;
  }

  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));
  unusable = false;
}

/**
 * @brief Throw if a HailoRT call failed.
 */
void
hailort_subplugin::check (hailo_status status, const char *what)
{
  if (status == HAILO_SUCCESS)
    return;

  const char *message = hailo_get_status_message (status);
  throw std::runtime_error (std::string (what) + " failed: " + (message ? message : "unknown")
                            + " (" + std::to_string ((int) status) + ")");
}

/**
 * @brief Parse the value of input_format or output_format.
 */
hailo_format_type_t
hailort_subplugin::parseFormatType (const gchar *value)
{
  if (g_ascii_strcasecmp (value, "auto") == 0)
    return HAILO_FORMAT_TYPE_AUTO;
  if (g_ascii_strcasecmp (value, "uint8") == 0)
    return HAILO_FORMAT_TYPE_UINT8;
  if (g_ascii_strcasecmp (value, "uint16") == 0)
    return HAILO_FORMAT_TYPE_UINT16;
  if (g_ascii_strcasecmp (value, "float32") == 0)
    return HAILO_FORMAT_TYPE_FLOAT32;

  throw std::invalid_argument (std::string ("Unknown HailoRT format type: ") + value);
}

/**
 * @brief Convert a HailoRT host-side format type to the NNStreamer tensor type.
 * @return _NNS_END if the type has no NNStreamer counterpart.
 */
tensor_type
hailort_subplugin::convertType (hailo_format_type_t type)
{
  switch (type) {
    case HAILO_FORMAT_TYPE_UINT8:
      return _NNS_UINT8;
    case HAILO_FORMAT_TYPE_UINT16:
      return _NNS_UINT16;
    case HAILO_FORMAT_TYPE_FLOAT32:
      return _NNS_FLOAT32;
    default:
      break;
  }

  return _NNS_END;
}

/**
 * @brief Describe a virtual stream as a tensor.
 *
 * The frame size HailoRT reports is the authority. NHWC-like frames become
 * features:width:height:1; any other layout (NCHW, NMS results, ...) is
 * passed as a flat tensor of the same byte size.
 */
void
hailort_subplugin::setTensorInfo (GstTensorInfo *info,
    const hailo_vstream_info_t &vinfo, const hailo_format_t &format, size_t frame_size)
{
  info->type = convertType (format.type);
  if (info->type == _NNS_END)
    throw std::invalid_argument (std::string ("Unsupported data type of ")
                                 + vinfo.name + ": "
                                 + std::to_string ((int) format.type));

  info->name = g_strdup (vinfo.name);

  const size_t element_size = gst_tensor_get_element_size (info->type);
  if (frame_size == 0 || frame_size % element_size != 0)
    throw std::invalid_argument (std::string ("Invalid frame size of ") + vinfo.name
                                 + ": " + std::to_string (frame_size));

  const size_t count = frame_size / element_size;
  const bool nhwc = (format.order == HAILO_FORMAT_ORDER_NHWC || format.order == HAILO_FORMAT_ORDER_NHW
                     || format.order == HAILO_FORMAT_ORDER_NC);

  if (nhwc && (guint64) vinfo.shape.height * vinfo.shape.width * vinfo.shape.features == count) {
    info->dimension[0] = vinfo.shape.features;
    info->dimension[1] = vinfo.shape.width;
    info->dimension[2] = vinfo.shape.height;
    info->dimension[3] = 1;
    return;
  }

  if (count > G_MAXUINT32)
    throw std::invalid_argument (std::string ("Too large frame of ") + vinfo.name);

  info->dimension[0] = (uint32_t) count;
}

/**
 * @brief Method to prepare/configure HailoRT instance.
 * @note If this throws, the framework deletes the object and the destructor
 *       releases whatever was created.
 */
void
hailort_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  hailo_format_type_t input_format = HAILO_FORMAT_TYPE_AUTO;
  hailo_format_type_t output_format = HAILO_FORMAT_TYPE_AUTO;
  std::string group_id;

  if (prop->num_models != 1 || !prop->model_files || !prop->model_files[0]
      || prop->model_files[0][0] == '\0')
    throw std::invalid_argument ("Exactly one model path (.hef) is required.");

  if (prop->custom_properties) {
    gchar **options = g_strsplit (prop->custom_properties, ",", -1);

    try {
      for (guint op = 0; options[op]; ++op) {
        gchar **option = g_strsplit (options[op], ":", 2);
        std::string key, value;

        if (option[0] && option[1]) {
          key = g_strstrip (option[0]);
          value = g_strstrip (option[1]);
        }
        g_strfreev (option);

        if (key == "input_format")
          input_format = parseFormatType (value.c_str ());
        else if (key == "output_format")
          output_format = parseFormatType (value.c_str ());
        else if (key == "group_id" && !value.empty ())
          group_id = value;
        else
          throw std::invalid_argument (
              std::string ("Invalid custom property: ") + options[op]);
      }
    } catch (...) {
      g_strfreev (options);
      throw;
    }

    g_strfreev (options);
  }

  check (hailo_create_hef_file (&hef, prop->model_files[0]), "hailo_create_hef_file");

  hailo_vdevice_params_t vdevice_params;
  check (hailo_init_vdevice_params (&vdevice_params), "hailo_init_vdevice_params");
  if (!group_id.empty ())
    vdevice_params.group_id = group_id.c_str ();
  check (hailo_create_vdevice (&vdevice_params, &vdevice), "hailo_create_vdevice");

  hailo_configure_params_t configure_params;
  check (hailo_init_configure_params_by_vdevice (hef, vdevice, &configure_params),
      "hailo_init_configure_params_by_vdevice");

  hailo_configured_network_group network_groups[HAILO_MAX_NETWORK_GROUPS];
  size_t num_network_groups = HAILO_MAX_NETWORK_GROUPS;
  check (hailo_configure_vdevice (vdevice, hef, &configure_params, network_groups, &num_network_groups),
      "hailo_configure_vdevice");
  if (num_network_groups != 1)
    throw std::invalid_argument ("The model has " + std::to_string (num_network_groups)
                                 + " network groups; only one is supported.");

  std::vector<hailo_input_vstream_params_by_name_t> input_params (HAILO_MAX_STREAMS_COUNT);
  size_t num_inputs = input_params.size ();
  check (hailo_make_input_vstream_params (network_groups[0], false,
             input_format, input_params.data (), &num_inputs),
      "hailo_make_input_vstream_params");

  std::vector<hailo_output_vstream_params_by_name_t> output_params (HAILO_MAX_STREAMS_COUNT);
  size_t num_outputs = output_params.size ();
  check (hailo_make_output_vstream_params (network_groups[0], false,
             output_format, output_params.data (), &num_outputs),
      "hailo_make_output_vstream_params");

  if (num_inputs == 0 || num_outputs == 0)
    throw std::invalid_argument ("The model has no input or no output.");

  std::vector<hailo_input_vstream> inputs (num_inputs);
  check (hailo_create_input_vstreams (network_groups[0], input_params.data (),
             num_inputs, inputs.data ()),
      "hailo_create_input_vstreams");
  input_vstreams = inputs;

  std::vector<hailo_output_vstream> outputs (num_outputs);
  check (hailo_create_output_vstreams (network_groups[0], output_params.data (),
             num_outputs, outputs.data ()),
      "hailo_create_output_vstreams");
  output_vstreams = outputs;

  inputInfo.num_tensors = num_inputs;
  for (size_t i = 0; i < num_inputs; ++i) {
    hailo_vstream_info_t vinfo;
    hailo_format_t format;
    size_t frame_size;

    check (hailo_get_input_vstream_info (input_vstreams[i], &vinfo),
        "hailo_get_input_vstream_info");
    check (hailo_get_input_vstream_user_format (input_vstreams[i], &format),
        "hailo_get_input_vstream_user_format");
    check (hailo_get_input_vstream_frame_size (input_vstreams[i], &frame_size),
        "hailo_get_input_vstream_frame_size");

    setTensorInfo (gst_tensors_info_get_nth_info (std::addressof (inputInfo), i),
        vinfo, format, frame_size);
    input_frame_sizes.push_back (frame_size);
  }

  outputInfo.num_tensors = num_outputs;
  for (size_t i = 0; i < num_outputs; ++i) {
    hailo_vstream_info_t vinfo;
    hailo_format_t format;
    size_t frame_size;

    check (hailo_get_output_vstream_info (output_vstreams[i], &vinfo),
        "hailo_get_output_vstream_info");
    check (hailo_get_output_vstream_user_format (output_vstreams[i], &format),
        "hailo_get_output_vstream_user_format");
    check (hailo_get_output_vstream_frame_size (output_vstreams[i], &frame_size),
        "hailo_get_output_vstream_frame_size");

    setTensorInfo (gst_tensors_info_get_nth_info (std::addressof (outputInfo), i),
        vinfo, format, frame_size);
    output_frame_sizes.push_back (frame_size);
  }
}

/**
 * @brief Method to execute the model.
 */
void
hailort_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");
  if (unusable)
    throw std::runtime_error ("An earlier invoke failed in HailoRT; reopen the model.");

  for (size_t i = 0; i < input_vstreams.size (); ++i) {
    if (input[i].size != input_frame_sizes[i])
      throw std::invalid_argument (
          "Input " + std::to_string (i) + " has " + std::to_string (input[i].size)
          + " bytes, expected " + std::to_string (input_frame_sizes[i]));
  }
  for (size_t i = 0; i < output_vstreams.size (); ++i) {
    if (output[i].size != output_frame_sizes[i])
      throw std::invalid_argument (
          "Output " + std::to_string (i) + " has " + std::to_string (output[i].size)
          + " bytes, expected " + std::to_string (output_frame_sizes[i]));
  }

  try {
    for (size_t i = 0; i < input_vstreams.size (); ++i)
      check (hailo_vstream_write_raw_buffer (
                 input_vstreams[i], input[i].data, input[i].size),
          "hailo_vstream_write_raw_buffer");

    for (size_t i = 0; i < output_vstreams.size (); ++i)
      check (hailo_vstream_read_raw_buffer (
                 output_vstreams[i], output[i].data, output[i].size),
          "hailo_vstream_read_raw_buffer");
  } catch (...) {
    unusable = true;
    throw;
  }
}

/**
 * @brief Method to get the information of HailoRT subplugin.
 */
void
hailort_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info = hailort_subplugin::framework_info;
}

/**
 * @brief Method to get the model information.
 */
int
hailort_subplugin::getModelInfo (
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
 * @brief Method to handle events.
 */
int
hailort_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);

  return -ENOENT;
}

hailort_subplugin *hailort_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
hailort_subplugin::init_filter_hailort (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<hailort_subplugin> ();
}

/** @brief Destruct the subplugin */
void
hailort_subplugin::fini_filter_hailort (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief Register the sub-plugin for HailoRT.
 */
void
init_filter_hailort ()
{
  hailort_subplugin::init_filter_hailort ();
}

/**
 * @brief Destruct the sub-plugin for HailoRT.
 */
void
fini_filter_hailort ()
{
  hailort_subplugin::fini_filter_hailort ();
}

} /* namespace tensor_filter_hailort */
} /* namespace nnstreamer */
