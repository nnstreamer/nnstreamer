/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    hailort_mock.cc
 * @date    28 Oct 2026
 * @brief   Mock of the HailoRT C API for the sub-plugin tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * See hailort_mock.h for the model format and the computation it emulates.
 */

#include <cstring>
#include <deque>
#include <fstream>
#include <glib.h>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "hailort_mock.h"

/** @brief A stream declared in a mock HEF */
struct mock_stream {
  std::string name;
  hailo_format_type_t type; /**< the device-side type */
  hailo_format_order_t order;
  hailo_3d_image_shape_t shape;
};

/** @brief A parsed mock HEF */
struct mock_model {
  std::vector<std::string> groups;
  std::vector<mock_stream> inputs;
  std::vector<mock_stream> outputs;
};

/** @brief A configured network group, owned by its virtual device */
struct _hailo_configured_network_group {
  std::shared_ptr<mock_model> model;
  bool first; /**< whether it is the group the streams belong to */
  std::vector<std::deque<std::vector<uint8_t>>> pending_inputs;
  std::vector<std::deque<std::vector<uint8_t>>> pending_outputs;
  std::vector<hailo_format_type_t> input_types; /**< user types, AUTO until created */
  std::vector<hailo_format_type_t> output_types;
};

/** @brief A virtual device */
struct _hailo_vdevice {
  std::vector<std::unique_ptr<_hailo_configured_network_group>> groups;
};

/** @brief A loaded HEF */
struct _hailo_hef {
  std::shared_ptr<mock_model> model;
};

/** @brief An input virtual stream */
struct _hailo_input_vstream {
  _hailo_configured_network_group *group;
  size_t index;
};

/** @brief An output virtual stream */
struct _hailo_output_vstream {
  _hailo_configured_network_group *group;
  size_t index;
};

static std::mutex mock_lock;
static std::map<std::string, hailo_status> mock_failures;
static std::map<std::string, int> mock_call_counts;
static bool mock_frame_size_overridden = false;
static size_t mock_frame_size_override = 0;
static hailo_format_type_t mock_user_type_override = HAILO_FORMAT_TYPE_AUTO;
static int mock_live = 0;
static std::string mock_group_id;

/**
 * @brief Count a call and return the failure injected into it, if any.
 */
static hailo_status
mock_enter (const char *function)
{
  std::lock_guard<std::mutex> guard (mock_lock);
  auto it = mock_failures.find (function);

  mock_call_counts[function]++;
  return (it == mock_failures.end ()) ? HAILO_SUCCESS : it->second;
}

#define MOCK_ENTER()                               \
  do {                                             \
    hailo_status injected = mock_enter (__func__); \
    if (injected != HAILO_SUCCESS)                 \
      return injected;                             \
  } while (0)

/**
 * @brief Track the number of live objects.
 */
static void
mock_live_add (int delta)
{
  std::lock_guard<std::mutex> guard (mock_lock);
  mock_live += delta;
}

/**
 * @brief The byte size of one element of the type.
 */
static size_t
mock_type_size (hailo_format_type_t type)
{
  switch (type) {
    case HAILO_FORMAT_TYPE_UINT16:
      return 2;
    case HAILO_FORMAT_TYPE_FLOAT32:
      return 4;
    default:
      return 1;
  }
}

/**
 * @brief Parse a type name of a mock HEF.
 */
static bool
mock_parse_type (const std::string &word, hailo_format_type_t &type)
{
  if (word == "uint8")
    type = HAILO_FORMAT_TYPE_UINT8;
  else if (word == "uint16")
    type = HAILO_FORMAT_TYPE_UINT16;
  else if (word == "float32")
    type = HAILO_FORMAT_TYPE_FLOAT32;
  else
    return false;
  return true;
}

/**
 * @brief Parse an order name of a mock HEF.
 */
static bool
mock_parse_order (const std::string &word, hailo_format_order_t &order)
{
  if (word == "nhwc")
    order = HAILO_FORMAT_ORDER_NHWC;
  else if (word == "nhw")
    order = HAILO_FORMAT_ORDER_NHW;
  else if (word == "nc")
    order = HAILO_FORMAT_ORDER_NC;
  else if (word == "nchw")
    order = HAILO_FORMAT_ORDER_NCHW;
  else if (word == "nms")
    order = HAILO_FORMAT_ORDER_HAILO_NMS;
  else
    return false;
  return true;
}

/**
 * @brief Parse a mock HEF file.
 */
static hailo_status
mock_parse_model (const char *file_name, mock_model &model)
{
  std::ifstream file (file_name);
  std::string line;

  if (!file.is_open ())
    return HAILO_OPEN_FILE_FAILURE;

  if (!std::getline (file, line) || line.rfind ("hailort-mock-hef", 0) != 0)
    return HAILO_INVALID_HEF;

  while (std::getline (file, line)) {
    std::istringstream words (line);
    std::string kind, name, type, order;
    mock_stream stream;

    if (!(words >> kind) || kind[0] == '#')
      continue;

    if (kind == "network_group") {
      if (!(words >> name))
        return HAILO_INVALID_HEF;
      model.groups.push_back (name);
      continue;
    }

    if (kind != "input" && kind != "output")
      return HAILO_INVALID_HEF;

    if (!(words >> stream.name >> type >> order >> stream.shape.height
            >> stream.shape.width >> stream.shape.features)
        || !mock_parse_type (type, stream.type)
        || !mock_parse_order (order, stream.order))
      return HAILO_INVALID_HEF;

    if (stream.name.size () >= HAILO_MAX_STREAM_NAME_SIZE)
      return HAILO_INVALID_HEF;

    (kind == "input" ? model.inputs : model.outputs).push_back (stream);
  }

  if (model.groups.empty ())
    return HAILO_INVALID_HEF;

  return HAILO_SUCCESS;
}

/**
 * @brief Host-side frame size of a stream in the given user type.
 */
static size_t
mock_frame_size (const mock_stream &stream, hailo_format_type_t user_type)
{
  if (mock_frame_size_overridden)
    return mock_frame_size_override;

  return (size_t) stream.shape.height * stream.shape.width
         * stream.shape.features * mock_type_size (user_type);
}

/**
 * @brief Read element @a index of a frame as a double.
 */
static double
mock_get (const std::vector<uint8_t> &frame, hailo_format_type_t type, size_t index)
{
  switch (type) {
    case HAILO_FORMAT_TYPE_UINT16:
      {
        uint16_t v;
        memcpy (&v, frame.data () + index * sizeof (v), sizeof (v));
        return v;
      }
    case HAILO_FORMAT_TYPE_FLOAT32:
      {
        float v;
        memcpy (&v, frame.data () + index * sizeof (v), sizeof (v));
        return v;
      }
    default:
      return frame[index];
  }
}

/**
 * @brief Write element @a index of a frame from a double.
 */
static void
mock_set (std::vector<uint8_t> &frame, hailo_format_type_t type, size_t index, double value)
{
  switch (type) {
    case HAILO_FORMAT_TYPE_UINT16:
      {
        uint16_t v = (uint16_t) value;
        memcpy (frame.data () + index * sizeof (v), &v, sizeof (v));
        break;
      }
    case HAILO_FORMAT_TYPE_FLOAT32:
      {
        float v = (float) value;
        memcpy (frame.data () + index * sizeof (v), &v, sizeof (v));
        break;
      }
    default:
      frame[index] = (uint8_t) value;
      break;
  }
}

/**
 * @brief Run the network once every input has a frame.
 */
static void
mock_run (_hailo_configured_network_group *group)
{
  for (auto &queue : group->pending_inputs) {
    if (queue.empty ())
      return;
  }

  const std::vector<uint8_t> in = group->pending_inputs[0].front ();
  const hailo_format_type_t in_type = group->input_types[0];
  const size_t in_count = in.size () / mock_type_size (in_type);

  for (size_t k = 0; k < group->model->outputs.size (); ++k) {
    const hailo_format_type_t out_type = group->output_types[k];
    std::vector<uint8_t> out (mock_frame_size (group->model->outputs[k], out_type));
    const size_t out_count = out.size () / mock_type_size (out_type);

    for (size_t j = 0; j < out_count; ++j)
      mock_set (out, out_type, j, mock_get (in, in_type, j % in_count) + (double) k);

    group->pending_outputs[k].push_back (out);
  }

  for (auto &queue : group->pending_inputs)
    queue.pop_front ();
}

/**
 * @brief Fill the information of a stream.
 */
static void
mock_fill_info (const mock_stream &stream, hailo_stream_direction_t direction,
    hailo_vstream_info_t *info)
{
  memset (info, 0, sizeof (*info));
  g_strlcpy (info->name, stream.name.c_str (), sizeof (info->name));
  g_strlcpy (info->network_name, "mock", sizeof (info->network_name));
  info->direction = direction;
  info->format.type = stream.type;
  info->format.order = stream.order;
  info->format.flags = HAILO_FORMAT_FLAGS_NONE;
  info->shape = stream.shape;
  info->quant_info.qp_scale = 1.0f;
}

/**
 * @brief The user format of a created stream.
 */
static void
mock_fill_user_format (
    const mock_stream &stream, hailo_format_type_t type, hailo_format_t *format)
{
  format->type = (mock_user_type_override != HAILO_FORMAT_TYPE_AUTO) ?
                     mock_user_type_override :
                     type;
  format->order = stream.order;
  format->flags = HAILO_FORMAT_FLAGS_NONE;
}

/**
 * @brief Common part of hailo_make_{input,output}_vstream_params ().
 */
template <typename T>
static hailo_status
mock_make_params (const std::vector<mock_stream> &streams,
    hailo_format_type_t format_type, T *params, size_t *count)
{
  if (!params || !count)
    return HAILO_INVALID_ARGUMENT;
  if (*count < streams.size ())
    return HAILO_INSUFFICIENT_BUFFER;

  for (size_t i = 0; i < streams.size (); ++i) {
    memset (&params[i], 0, sizeof (params[i]));
    g_strlcpy (params[i].name, streams[i].name.c_str (), sizeof (params[i].name));
    params[i].params.user_buffer_format.type = format_type;
    params[i].params.user_buffer_format.order = HAILO_FORMAT_ORDER_AUTO;
    params[i].params.timeout_ms = 10000;
    params[i].params.queue_size = 2;
  }

  *count = streams.size ();
  return HAILO_SUCCESS;
}

/**
 * @brief Find a stream by name.
 */
static bool
mock_find (const std::vector<mock_stream> &streams, const char *name, size_t &index)
{
  for (index = 0; index < streams.size (); ++index) {
    if (streams[index].name == name)
      return true;
  }
  return false;
}

extern "C" {

/** @brief Mock of hailo_get_status_message (). */
const char *
hailo_get_status_message (hailo_status status)
{
  switch (status) {
    case HAILO_SUCCESS:
      return "HAILO_SUCCESS";
    case HAILO_INVALID_ARGUMENT:
      return "HAILO_INVALID_ARGUMENT";
    case HAILO_TIMEOUT:
      return "HAILO_TIMEOUT";
    case HAILO_INVALID_HEF:
      return "HAILO_INVALID_HEF";
    default:
      return nullptr;
  }
}

/** @brief Mock of hailo_init_vdevice_params (). */
hailo_status
hailo_init_vdevice_params (hailo_vdevice_params_t *params)
{
  MOCK_ENTER ();
  if (!params)
    return HAILO_INVALID_ARGUMENT;

  params->device_count = 1;
  params->device_ids = nullptr;
  params->scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
  params->group_id = HAILO_DEFAULT_VDEVICE_GROUP_ID;
  params->multi_process_service = false;
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_create_vdevice (). */
hailo_status
hailo_create_vdevice (hailo_vdevice_params_t *params, hailo_vdevice *vdevice)
{
  MOCK_ENTER ();
  if (!vdevice)
    return HAILO_INVALID_ARGUMENT;

  {
    std::lock_guard<std::mutex> guard (mock_lock);
    mock_group_id = (params && params->group_id) ? params->group_id : "";
  }

  *vdevice = new _hailo_vdevice ();
  mock_live_add (1);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_configure_vdevice (). */
hailo_status
hailo_configure_vdevice (hailo_vdevice vdevice, hailo_hef hef,
    hailo_configure_params_t *params,
    hailo_configured_network_group *network_groups, size_t *number_of_network_groups)
{
  MOCK_ENTER ();
  if (!vdevice || !hef || !network_groups || !number_of_network_groups)
    return HAILO_INVALID_ARGUMENT;
  if (params && params->network_group_params_count != hef->model->groups.size ())
    return HAILO_INVALID_ARGUMENT;
  if (*number_of_network_groups < hef->model->groups.size ())
    return HAILO_INSUFFICIENT_BUFFER;

  for (size_t g = 0; g < hef->model->groups.size (); ++g) {
    auto group = std::make_unique<_hailo_configured_network_group> ();

    group->model = hef->model;
    group->first = (g == 0);
    if (group->first) {
      group->pending_inputs.resize (hef->model->inputs.size ());
      group->pending_outputs.resize (hef->model->outputs.size ());
      group->input_types.assign (hef->model->inputs.size (), HAILO_FORMAT_TYPE_AUTO);
      group->output_types.assign (hef->model->outputs.size (), HAILO_FORMAT_TYPE_AUTO);
    }
    network_groups[g] = group.get ();
    vdevice->groups.push_back (std::move (group));
  }

  *number_of_network_groups = hef->model->groups.size ();
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_release_vdevice (). */
hailo_status
hailo_release_vdevice (hailo_vdevice vdevice)
{
  MOCK_ENTER ();
  if (!vdevice)
    return HAILO_INVALID_ARGUMENT;

  delete vdevice;
  mock_live_add (-1);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_create_hef_file (). */
hailo_status
hailo_create_hef_file (hailo_hef *hef, const char *file_name)
{
  MOCK_ENTER ();
  if (!hef || !file_name)
    return HAILO_INVALID_ARGUMENT;

  auto model = std::make_shared<mock_model> ();
  hailo_status status = mock_parse_model (file_name, *model);
  if (status != HAILO_SUCCESS)
    return status;

  *hef = new _hailo_hef ();
  (*hef)->model = model;
  mock_live_add (1);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_release_hef (). */
hailo_status
hailo_release_hef (hailo_hef hef)
{
  MOCK_ENTER ();
  if (!hef)
    return HAILO_INVALID_ARGUMENT;

  delete hef;
  mock_live_add (-1);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_init_configure_params_by_vdevice (). */
hailo_status
hailo_init_configure_params_by_vdevice (
    hailo_hef hef, hailo_vdevice vdevice, hailo_configure_params_t *params)
{
  MOCK_ENTER ();
  if (!hef || !vdevice || !params)
    return HAILO_INVALID_ARGUMENT;

  memset (params, 0, sizeof (*params));
  params->network_group_params_count = hef->model->groups.size ();
  for (size_t g = 0; g < hef->model->groups.size () && g < HAILO_MAX_NETWORK_GROUPS; ++g)
    g_strlcpy (params->network_group_params[g].name, hef->model->groups[g].c_str (),
        sizeof (params->network_group_params[g].name));
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_make_input_vstream_params (). */
hailo_status
hailo_make_input_vstream_params (hailo_configured_network_group network_group,
    bool unused, hailo_format_type_t format_type,
    hailo_input_vstream_params_by_name_t *input_params, size_t *input_params_count)
{
  MOCK_ENTER ();
  (void) unused;
  if (!network_group)
    return HAILO_INVALID_ARGUMENT;

  static const std::vector<mock_stream> none;
  return mock_make_params (network_group->first ? network_group->model->inputs : none,
      format_type, input_params, input_params_count);
}

/** @brief Mock of hailo_make_output_vstream_params (). */
hailo_status
hailo_make_output_vstream_params (hailo_configured_network_group network_group,
    bool unused, hailo_format_type_t format_type,
    hailo_output_vstream_params_by_name_t *output_params, size_t *output_params_count)
{
  MOCK_ENTER ();
  (void) unused;
  if (!network_group)
    return HAILO_INVALID_ARGUMENT;

  static const std::vector<mock_stream> none;
  return mock_make_params (network_group->first ? network_group->model->outputs : none,
      format_type, output_params, output_params_count);
}

/** @brief Mock of hailo_create_input_vstreams (). */
hailo_status
hailo_create_input_vstreams (hailo_configured_network_group configured_network_group,
    const hailo_input_vstream_params_by_name_t *inputs_params,
    size_t inputs_count, hailo_input_vstream *input_vstreams)
{
  MOCK_ENTER ();
  if (!configured_network_group || !configured_network_group->first
      || !inputs_params || !input_vstreams)
    return HAILO_INVALID_ARGUMENT;

  for (size_t i = 0; i < inputs_count; ++i) {
    size_t index;

    if (!mock_find (configured_network_group->model->inputs, inputs_params[i].name, index))
      return HAILO_NOT_FOUND;
  }

  for (size_t i = 0; i < inputs_count; ++i) {
    size_t index;
    hailo_format_type_t type = inputs_params[i].params.user_buffer_format.type;

    mock_find (configured_network_group->model->inputs, inputs_params[i].name, index);
    if (type == HAILO_FORMAT_TYPE_AUTO)
      type = configured_network_group->model->inputs[index].type;
    configured_network_group->input_types[index] = type;
    input_vstreams[i] = new _hailo_input_vstream{ configured_network_group, index };
    mock_live_add (1);
  }

  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_create_output_vstreams (). */
hailo_status
hailo_create_output_vstreams (hailo_configured_network_group configured_network_group,
    const hailo_output_vstream_params_by_name_t *outputs_params,
    size_t outputs_count, hailo_output_vstream *output_vstreams)
{
  MOCK_ENTER ();
  if (!configured_network_group || !configured_network_group->first
      || !outputs_params || !output_vstreams)
    return HAILO_INVALID_ARGUMENT;

  for (size_t i = 0; i < outputs_count; ++i) {
    size_t index;

    if (!mock_find (configured_network_group->model->outputs, outputs_params[i].name, index))
      return HAILO_NOT_FOUND;
  }

  for (size_t i = 0; i < outputs_count; ++i) {
    size_t index;
    hailo_format_type_t type = outputs_params[i].params.user_buffer_format.type;

    mock_find (configured_network_group->model->outputs, outputs_params[i].name, index);
    if (type == HAILO_FORMAT_TYPE_AUTO)
      type = configured_network_group->model->outputs[index].type;
    configured_network_group->output_types[index] = type;
    output_vstreams[i] = new _hailo_output_vstream{ configured_network_group, index };
    mock_live_add (1);
  }

  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_release_input_vstreams (). */
hailo_status
hailo_release_input_vstreams (const hailo_input_vstream *input_vstreams, size_t inputs_count)
{
  MOCK_ENTER ();
  if (!input_vstreams)
    return HAILO_INVALID_ARGUMENT;

  for (size_t i = 0; i < inputs_count; ++i) {
    delete input_vstreams[i];
    mock_live_add (-1);
  }
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_release_output_vstreams (). */
hailo_status
hailo_release_output_vstreams (const hailo_output_vstream *output_vstreams, size_t outputs_count)
{
  MOCK_ENTER ();
  if (!output_vstreams)
    return HAILO_INVALID_ARGUMENT;

  for (size_t i = 0; i < outputs_count; ++i) {
    delete output_vstreams[i];
    mock_live_add (-1);
  }
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_get_input_vstream_frame_size (). */
hailo_status
hailo_get_input_vstream_frame_size (hailo_input_vstream input_vstream, size_t *frame_size)
{
  MOCK_ENTER ();
  if (!input_vstream || !frame_size)
    return HAILO_INVALID_ARGUMENT;

  _hailo_configured_network_group *group = input_vstream->group;
  *frame_size = mock_frame_size (group->model->inputs[input_vstream->index],
      group->input_types[input_vstream->index]);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_get_input_vstream_info (). */
hailo_status
hailo_get_input_vstream_info (hailo_input_vstream input_vstream, hailo_vstream_info_t *vstream_info)
{
  MOCK_ENTER ();
  if (!input_vstream || !vstream_info)
    return HAILO_INVALID_ARGUMENT;

  mock_fill_info (input_vstream->group->model->inputs[input_vstream->index],
      HAILO_H2D_STREAM, vstream_info);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_get_input_vstream_user_format (). */
hailo_status
hailo_get_input_vstream_user_format (
    hailo_input_vstream input_vstream, hailo_format_t *user_buffer_format)
{
  MOCK_ENTER ();
  if (!input_vstream || !user_buffer_format)
    return HAILO_INVALID_ARGUMENT;

  _hailo_configured_network_group *group = input_vstream->group;
  mock_fill_user_format (group->model->inputs[input_vstream->index],
      group->input_types[input_vstream->index], user_buffer_format);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_get_output_vstream_frame_size (). */
hailo_status
hailo_get_output_vstream_frame_size (hailo_output_vstream output_vstream, size_t *frame_size)
{
  MOCK_ENTER ();
  if (!output_vstream || !frame_size)
    return HAILO_INVALID_ARGUMENT;

  _hailo_configured_network_group *group = output_vstream->group;
  *frame_size = mock_frame_size (group->model->outputs[output_vstream->index],
      group->output_types[output_vstream->index]);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_get_output_vstream_info (). */
hailo_status
hailo_get_output_vstream_info (
    hailo_output_vstream output_vstream, hailo_vstream_info_t *vstream_info)
{
  MOCK_ENTER ();
  if (!output_vstream || !vstream_info)
    return HAILO_INVALID_ARGUMENT;

  mock_fill_info (output_vstream->group->model->outputs[output_vstream->index],
      HAILO_D2H_STREAM, vstream_info);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_get_output_vstream_user_format (). */
hailo_status
hailo_get_output_vstream_user_format (
    hailo_output_vstream output_vstream, hailo_format_t *user_buffer_format)
{
  MOCK_ENTER ();
  if (!output_vstream || !user_buffer_format)
    return HAILO_INVALID_ARGUMENT;

  _hailo_configured_network_group *group = output_vstream->group;
  mock_fill_user_format (group->model->outputs[output_vstream->index],
      group->output_types[output_vstream->index], user_buffer_format);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_vstream_write_raw_buffer (). */
hailo_status
hailo_vstream_write_raw_buffer (
    hailo_input_vstream input_vstream, const void *buffer, size_t buffer_size)
{
  MOCK_ENTER ();
  if (!input_vstream || !buffer)
    return HAILO_INVALID_ARGUMENT;

  _hailo_configured_network_group *group = input_vstream->group;
  const size_t index = input_vstream->index;
  const size_t expected
      = mock_frame_size (group->model->inputs[index], group->input_types[index]);
  if (buffer_size != expected)
    return HAILO_INVALID_ARGUMENT;

  std::lock_guard<std::mutex> guard (mock_lock);
  const uint8_t *data = static_cast<const uint8_t *> (buffer);
  group->pending_inputs[index].emplace_back (data, data + buffer_size);
  mock_run (group);
  return HAILO_SUCCESS;
}

/** @brief Mock of hailo_vstream_read_raw_buffer (). */
hailo_status
hailo_vstream_read_raw_buffer (hailo_output_vstream output_vstream, void *buffer, size_t buffer_size)
{
  MOCK_ENTER ();
  if (!output_vstream || !buffer)
    return HAILO_INVALID_ARGUMENT;

  _hailo_configured_network_group *group = output_vstream->group;
  const size_t index = output_vstream->index;

  std::lock_guard<std::mutex> guard (mock_lock);
  auto &queue = group->pending_outputs[index];
  if (queue.empty ())
    return HAILO_TIMEOUT;
  if (buffer_size != queue.front ().size ())
    return HAILO_INVALID_ARGUMENT;

  memcpy (buffer, queue.front ().data (), buffer_size);
  queue.pop_front ();
  return HAILO_SUCCESS;
}

/** @brief Clear the injected failures, overrides and call counters. */
void
hailort_mock_reset (void)
{
  std::lock_guard<std::mutex> guard (mock_lock);
  mock_failures.clear ();
  mock_call_counts.clear ();
  mock_frame_size_overridden = false;
  mock_frame_size_override = 0;
  mock_user_type_override = HAILO_FORMAT_TYPE_AUTO;
  mock_group_id.clear ();
}

/** @brief Inject a failure into the named function. */
void
hailort_mock_fail (const char *function, hailo_status status)
{
  std::lock_guard<std::mutex> guard (mock_lock);
  mock_failures[function] = status;
}

/** @brief Override the reported frame sizes. */
void
hailort_mock_override_frame_size (size_t frame_size)
{
  std::lock_guard<std::mutex> guard (mock_lock);
  mock_frame_size_overridden = true;
  mock_frame_size_override = frame_size;
}

/** @brief Override the reported user format type. */
void
hailort_mock_override_user_type (hailo_format_type_t type)
{
  std::lock_guard<std::mutex> guard (mock_lock);
  mock_user_type_override = type;
}

/** @brief The number of calls of the named function. */
int
hailort_mock_calls (const char *function)
{
  std::lock_guard<std::mutex> guard (mock_lock);
  auto it = mock_call_counts.find (function);
  return (it == mock_call_counts.end ()) ? 0 : it->second;
}

/** @brief The number of objects not released yet. */
int
hailort_mock_live_objects (void)
{
  std::lock_guard<std::mutex> guard (mock_lock);
  return mock_live;
}

/** @brief The group id of the last virtual device. */
char *
hailort_mock_last_group_id (void)
{
  std::lock_guard<std::mutex> guard (mock_lock);
  return g_strdup (mock_group_id.c_str ());
}

} /* extern "C" */
