/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    trix_mock.cc
 * @date    22 Nov 2026
 * @brief   A mock npu-engine library for the trix-engine sub-plugin tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * See trix_mock.h for what it does.
 */
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "trix_mock.h"

namespace
{
/** @brief A request and what setNPU_requestData () gave it. */
struct mock_request {
  npudev_h dev;
  uint32_t model_id;
  bool data_set;
  input_buffers input;
  output_buffers *output;
  tensors_data_info out_info;
};

/** @brief An open device and the models registered to it. */
struct mock_device {
  std::map<uint32_t, npubin_meta> models;
};

std::set<void *> devices;
std::map<int, mock_request> requests;
std::set<std::string> failing;
std::set<std::string> failing_once;
std::map<std::string, int> calls;
uint32_t next_model_id = 1;
int next_req_id = 1;
int live_models = 0;
dev_type last_device_type = NPUCOND_CONN_UNKNOWN;
tensors_data_info last_in_info = {};
tensors_data_info last_out_info = {};

/** @brief Count a call and tell whether it is to fail. */
bool
enter (const char *function)
{
  calls[function]++;
  if (failing_once.erase (function) > 0)
    return true;
  return failing.count (function) > 0;
}

/** @brief The size of an element of a data type; the model type is uint8. */
uint64_t
element_size (data_type type)
{
  switch (type) {
    case DATA_TYPE_INT16:
    case DATA_TYPE_UINT16:
      return 2;
    case DATA_TYPE_INT32:
    case DATA_TYPE_UINT32:
    case DATA_TYPE_FLOAT32:
      return 4;
    case DATA_TYPE_INT64:
    case DATA_TYPE_UINT64:
    case DATA_TYPE_FLOAT64:
      return 8;
    default:
      return 1;
  }
}

/** @brief The number of elements of a segment. */
uint64_t
elements (const uint32_t *dims)
{
  uint64_t n = 1;
  for (int i = 0; i < MAX_RANK; i++)
    n *= dims[i];
  return n;
}

/** @brief Find a model of an open device. */
const npubin_meta *
find_model (npudev_h dev, uint32_t model_id)
{
  if (devices.count (dev) == 0)
    return nullptr;
  mock_device *d = static_cast<mock_device *> (dev);
  auto it = d->models.find (model_id);
  return (it == d->models.end ()) ? nullptr : &it->second;
}

/** @brief Parse a text model into @a meta. */
bool
parse_model (const char *path, npubin_meta *meta)
{
  std::ifstream file (path);
  std::string line;

  if (!file || !std::getline (file, line) || line != "trix-mock-model")
    return false;

  memset (meta, 0, sizeof (*meta));
  while (std::getline (file, line)) {
    std::istringstream in (line);
    std::string kind;
    uint32_t dims[MAX_RANK];

    if (!(in >> kind))
      continue;
    for (int i = 0; i < MAX_RANK; i++) {
      if (!(in >> dims[i]) || dims[i] == 0)
        return false;
    }
    if (kind == "in" && meta->input_seg_num < TRIX_MOCK_MAX_SEGMENTS) {
      memcpy (meta->input_seg_dims[meta->input_seg_num++], dims, sizeof (dims));
    } else if (kind == "out" && meta->output_seg_num < TRIX_MOCK_MAX_SEGMENTS) {
      memcpy (meta->output_seg_dims[meta->output_seg_num++], dims, sizeof (dims));
    } else {
      return false;
    }
  }

  file.clear ();
  file.seekg (0, std::ios::end);
  meta->size = (uint64_t) file.tellg ();
  return meta->input_seg_num > 0 && meta->output_seg_num > 0;
}

} /* namespace */

extern "C" {

/** @brief Read the metadata of a text model. */
npubin_meta *
getNPUmodel_metadata (const char *model, bool i64)
{
  (void) i64;
  npubin_meta meta;

  if (enter ("getNPUmodel_metadata") || !model || !parse_model (model, &meta))
    return nullptr;

  npubin_meta *copy = (npubin_meta *) malloc (sizeof (meta));
  memcpy (copy, &meta, sizeof (meta));
  return copy;
}

/** @brief Open a device. */
int
getNPUdeviceByTypeAny (npudev_h *dev, dev_type type, uint32_t tops)
{
  (void) tops;
  last_device_type = type;
  if (enter ("getNPUdeviceByTypeAny"))
    return -ENODEV;
  if (!dev || type != NPUCOND_TRIV2_CONN_SOCIP)
    return -EINVAL;

  mock_device *d = new mock_device ();
  devices.insert (d);
  *dev = d;
  return 0;
}

/** @brief Close a device. */
int
putNPUdevice (npudev_h dev)
{
  enter ("putNPUdevice");
  if (devices.erase (dev) == 0)
    return -EINVAL;

  mock_device *d = static_cast<mock_device *> (dev);
  live_models -= (int) d->models.size ();
  for (auto it = requests.begin (); it != requests.end ();) {
    if (it->second.dev == dev)
      it = requests.erase (it);
    else
      ++it;
  }
  delete d;
  return 0;
}

/** @brief Register a model of a file. */
int
registerNPUmodel (npudev_h dev, generic_buffer *model, uint32_t *model_id)
{
  npubin_meta meta;

  if (enter ("registerNPUmodel"))
    return -EIO;
  if (devices.count (dev) == 0 || !model || !model_id || model->type != BUFFER_FILE
      || !parse_model (model->filepath, &meta) || model->size != meta.size)
    return -EINVAL;

  static_cast<mock_device *> (dev)->models[next_model_id] = meta;
  live_models++;
  *model_id = next_model_id++;
  return 0;
}

/** @brief Unregister every model of a device. */
int
unregisterNPUmodel_all (npudev_h dev)
{
  enter ("unregisterNPUmodel_all");
  if (devices.count (dev) == 0)
    return -EINVAL;

  mock_device *d = static_cast<mock_device *> (dev);
  live_models -= (int) d->models.size ();
  d->models.clear ();
  return 0;
}

/** @brief Create a request. */
int
createNPU_request (npudev_h dev, uint32_t model_id, int *req_id)
{
  if (enter ("createNPU_request"))
    return -EBUSY;
  if (!find_model (dev, model_id) || !req_id)
    return -EINVAL;

  mock_request r = {};
  r.dev = dev;
  r.model_id = model_id;
  *req_id = next_req_id++;
  requests[*req_id] = r;
  return 0;
}

/** @brief Check and keep the data of a request. */
int
setNPU_requestData (npudev_h dev, int req_id, input_buffers *input,
    tensors_data_info *in_info, output_buffers *output, tensors_data_info *out_info)
{
  if (enter ("setNPU_requestData"))
    return -EINVAL;

  auto it = requests.find (req_id);
  if (it == requests.end () || it->second.dev != dev || !input || !in_info
      || !output || !out_info)
    return -EINVAL;

  const npubin_meta *meta = find_model (dev, it->second.model_id);
  last_in_info = *in_info;
  last_out_info = *out_info;
  if (!meta || input->num_buffers != meta->input_seg_num
      || in_info->num_info != meta->input_seg_num || out_info->num_info != meta->output_seg_num)
    return -EINVAL;

  for (uint32_t i = 0; i < input->num_buffers; i++) {
    const generic_buffer *b = &input->bufs[i];
    uint64_t size
        = elements (meta->input_seg_dims[i]) * element_size (in_info->info[i].type);
    if (b->type != BUFFER_MAPPED || !b->addr || b->size != size)
      return -EINVAL;
  }

  it->second.data_set = true;
  it->second.input = *input;
  it->second.output = output;
  it->second.out_info = *out_info;
  return 0;
}

/** @brief Run a request: fill its output buffers. */
int
submitNPU_request (npudev_h dev, int req_id)
{
  if (enter ("submitNPU_request"))
    return -EIO;

  auto it = requests.find (req_id);
  if (it == requests.end () || it->second.dev != dev || !it->second.data_set)
    return -EINVAL;

  const npubin_meta *meta = find_model (dev, it->second.model_id);
  const mock_request &r = it->second;
  if (!meta)
    return -EINVAL;

  const uint8_t *in0 = static_cast<const uint8_t *> (r.input.bufs[0].addr);
  uint64_t in0_size = r.input.bufs[0].size;

  r.output->num_buffers = meta->output_seg_num;
  for (uint32_t k = 0; k < meta->output_seg_num; k++) {
    uint64_t size = elements (meta->output_seg_dims[k])
                    * element_size (r.out_info.info[k].type);
    uint8_t *out = static_cast<uint8_t *> (malloc (size));

    for (uint64_t j = 0; j < size; j++)
      out[j] = (uint8_t) (in0[j % in0_size] + k + 1);
    r.output->bufs[k].addr = out;
    r.output->bufs[k].size = size;
    r.output->bufs[k].type = BUFFER_MAPPED;
    r.output->bufs[k].filepath = nullptr;
  }
  return 0;
}

/** @brief Remove a request. */
int
removeNPU_request (npudev_h dev, int req_id)
{
  if (enter ("removeNPU_request"))
    return -EINVAL;

  auto it = requests.find (req_id);
  if (it == requests.end () || it->second.dev != dev)
    return -EINVAL;
  requests.erase (it);
  return 0;
}

/** @brief Clear the injected failures and the counters. */
void
trix_mock_reset (void)
{
  failing.clear ();
  failing_once.clear ();
  calls.clear ();
  last_device_type = NPUCOND_CONN_UNKNOWN;
  memset (&last_in_info, 0, sizeof (last_in_info));
  memset (&last_out_info, 0, sizeof (last_out_info));
}

/** @brief Make the named function fail. */
void
trix_mock_fail (const char *function)
{
  failing.insert (function);
}

/** @brief Make the next call of the named function fail. */
void
trix_mock_fail_next (const char *function)
{
  failing_once.insert (function);
}

/** @brief The number of calls of the named function. */
int
trix_mock_calls (const char *function)
{
  auto it = calls.find (function);
  return (it == calls.end ()) ? 0 : it->second;
}

/** @brief The number of devices, models and requests not released. */
int
trix_mock_live_objects (void)
{
  return (int) (devices.size () + requests.size ()) + live_models;
}

/** @brief The device type of the last getNPUdeviceByTypeAny () call. */
dev_type
trix_mock_last_device_type (void)
{
  return last_device_type;
}

/** @brief The data info of a segment of the last setNPU_requestData () call. */
int
trix_mock_last_data_info (bool input, uint32_t idx, data_layout *layout, data_type *type)
{
  const tensors_data_info *info = input ? &last_in_info : &last_out_info;

  if (idx >= info->num_info || idx >= TRIX_MOCK_MAX_SEGMENTS)
    return -1;
  *layout = info->info[idx].layout;
  *type = info->info[idx].type;
  return 0;
}

} /* extern "C" */
