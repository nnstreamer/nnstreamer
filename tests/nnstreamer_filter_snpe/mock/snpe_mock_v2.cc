/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    snpe_mock_v2.cc
 * @date    22 Sep 2026
 * @brief   Mock implementation of the SNPE 2.x C API.
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * The emulated model has one input and one output named by SNPE_MOCK_INPUT_NAME
 * and SNPE_MOCK_OUTPUT_NAME, and computes output = input + SNPE_MOCK_ADDEND.
 * A container file is never read: its name decides whether the model is float
 * or quantized, see snpe_mock_model_load().
 */

#include <map>
#include <string>
#include <vector>

#include <glib.h>
#include <nnstreamer_util.h>

#include "snpe_mock.h"

#include <DlContainer/DlContainer.h>
#include <DlSystem/DlEnums.h>
#include <DlSystem/DlError.h>
#include <DlSystem/DlVersion.h>
#include <DlSystem/IUserBuffer.h>
#include <DlSystem/RuntimeList.h>
#include <DlSystem/UserBufferMap.h>
#include <SNPE/SNPE.h>
#include <SNPE/SNPEBuilder.h>
#include <SNPE/SNPEUtil.h>

#define MOCK_LIBRARY_VERSION "2.0.0.0"

namespace
{

/** @brief Mock of an opened .dlc container. */
struct MockContainer {
  snpe_mock_model model;
};

/** @brief Mock of a string list. */
struct MockStringList {
  std::vector<std::string> items;
};

/** @brief Mock of a tensor shape. */
struct MockTensorShape {
  std::vector<size_t> dims;
};

/** @brief Mock of a user buffer encoding. */
struct MockEncoding {
  Snpe_UserBufferEncoding_ElementType_t type;
  uint64_t step_exactly_0;
  float quantized_step_size;
  bool owned;
};

/** @brief Mock of the buffer attributes of one tensor. */
struct MockBufferAttributes {
  std::vector<size_t> dims;
  MockEncoding encoding;
};

/** @brief Mock of a user buffer. */
struct MockUserBuffer {
  void *address;
  size_t size;
  Snpe_UserBufferEncoding_ElementType_t type;
};

/** @brief Mock of a user buffer map. */
struct MockUserBufferMap {
  std::map<std::string, MockUserBuffer *> buffers;
};

/** @brief Mock of a runtime list. */
struct MockRuntimeList {
  std::vector<Snpe_Runtime_t> runtimes;
};

/** @brief Mock of the SDK version. */
struct MockVersion {
  int32_t major;
};

/** @brief Mock of an SNPE builder. */
struct MockBuilder {
  snpe_mock_model model;
  bool has_container;
  std::vector<std::string> output_names;
};

/** @brief Mock of a built SNPE instance. */
struct MockSNPE {
  snpe_mock_model model;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

/**
 * @brief The encoding a _Ref accessor hands out, owned by the mock itself.
 *
 * The real SDK hands back a pointer into the attributes object. Keeping one
 * process-wide instance instead lets the mock survive a release call on a
 * handle the caller does not own. Such a call is therefore invisible to a
 * test; see the README of this directory for why that is deliberate.
 */
MockEncoding referenced_encoding
    = { SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT, 0, 1.0f, false };

/**
 * @brief Get the element size in bytes of the given encoding element type.
 */
size_t
element_size (Snpe_UserBufferEncoding_ElementType_t type)
{
  return (type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT) ? sizeof (float) :
                                                               sizeof (uint8_t);
}

/**
 * @brief Get the default encoding element type of the given model.
 */
Snpe_UserBufferEncoding_ElementType_t
default_encoding (const snpe_mock_model &model)
{
  if (model.unsupported_encoding)
    return SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16;
  if (model.quantized)
    return SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8;
  return SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT;
}

/**
 * @brief Get the shape of every tensor of the given model.
 */
std::vector<size_t>
model_dims (const snpe_mock_model &model)
{
  std::vector<size_t> dims (1, 1U);

  if (model.resizable)
    dims.push_back (0U);

  return dims;
}

/**
 * @brief Add SNPE_MOCK_ADDEND to every element the output buffer can hold.
 */
void
add2 (const MockUserBuffer *in, MockUserBuffer *out)
{
  if (!in || !out || !in->address || !out->address)
    return;

  const size_t out_elems = out->size / element_size (out->type);
  const size_t in_elems = in->size / element_size (in->type);

  for (size_t i = 0; i < out_elems; i++) {
    double value = SNPE_MOCK_ADDEND;

    if (i < in_elems) {
      if (in->type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT)
        value += static_cast<const float *> (in->address)[i];
      else
        value += static_cast<const uint8_t *> (in->address)[i];
    }

    if (out->type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT)
      static_cast<float *> (out->address)[i] = static_cast<float> (value);
    else
      static_cast<uint8_t *> (out->address)[i] = static_cast<uint8_t> (value);
  }
}

} /* namespace */

/**
 * @brief Open a mock container for the given model file.
 */
Snpe_DlContainer_Handle_t
Snpe_DlContainer_Open (const char *filename)
{
  snpe_mock_model model;

  if (!snpe_mock_model_load (filename, &model))
    return nullptr;

  MockContainer *container = new MockContainer ();
  container->model = model;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_CONTAINER);

  return container;
}

/**
 * @brief Release a mock container.
 */
Snpe_ErrorCode_t
Snpe_DlContainer_Delete (Snpe_DlContainer_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockContainer *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_CONTAINER);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Get the major number of the mock SDK version.
 */
int32_t
Snpe_DlVersion_GetMajor (Snpe_DlVersion_Handle_t handle)
{
  return handle ? static_cast<MockVersion *> (handle)->major : 0;
}

/**
 * @brief Get the mock SDK version as a string.
 */
const char *
Snpe_DlVersion_ToString (Snpe_DlVersion_Handle_t handle)
{
  return handle ? MOCK_LIBRARY_VERSION : "";
}

/**
 * @brief Release a mock version handle.
 */
Snpe_ErrorCode_t
Snpe_DlVersion_Delete (Snpe_DlVersion_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockVersion *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_VERSION);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Create an empty mock string list.
 */
Snpe_StringList_Handle_t
Snpe_StringList_Create (void)
{
  snpe_mock_obj_created (SNPE_MOCK_OBJ_STRING_LIST);
  return new MockStringList ();
}

/**
 * @brief Append a string to a mock string list.
 */
Snpe_ErrorCode_t
Snpe_StringList_Append (Snpe_StringList_Handle_t handle, const char *string)
{
  if (!handle || !string)
    return SNPE_ERRORCODE_INTERNAL_ERROR;

  if (snpe_mock_get_failure () == SNPE_MOCK_FAIL_STRING_LIST_APPEND)
    return SNPE_ERRORCODE_INTERNAL_ERROR;

  static_cast<MockStringList *> (handle)->items.push_back (string);
  return SNPE_SUCCESS;
}

/**
 * @brief Get the number of strings in a mock string list.
 */
size_t
Snpe_StringList_Size (Snpe_StringList_Handle_t handle)
{
  return handle ? static_cast<MockStringList *> (handle)->items.size () : 0;
}

/**
 * @brief Get the string at the given index of a mock string list.
 */
const char *
Snpe_StringList_At (Snpe_StringList_Handle_t handle, size_t idx)
{
  MockStringList *list = static_cast<MockStringList *> (handle);

  if (!list || idx >= list->items.size ())
    return nullptr;

  return list->items[idx].c_str ();
}

/**
 * @brief Release a mock string list.
 */
Snpe_ErrorCode_t
Snpe_StringList_Delete (Snpe_StringList_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockStringList *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_STRING_LIST);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Create a mock tensor shape from the given dimensions.
 */
Snpe_TensorShape_Handle_t
Snpe_TensorShape_CreateDimsSize (const size_t *dims, size_t size)
{
  MockTensorShape *shape = new MockTensorShape ();

  if (dims)
    shape->dims.assign (dims, dims + size);
  snpe_mock_obj_created (SNPE_MOCK_OBJ_TENSOR_SHAPE);

  return shape;
}

/**
 * @brief Get the rank of a mock tensor shape.
 */
size_t
Snpe_TensorShape_Rank (Snpe_TensorShape_Handle_t handle)
{
  return handle ? static_cast<MockTensorShape *> (handle)->dims.size () : 0;
}

/**
 * @brief Get the dimensions of a mock tensor shape.
 */
const size_t *
Snpe_TensorShape_GetDimensions (Snpe_TensorShape_Handle_t handle)
{
  MockTensorShape *shape = static_cast<MockTensorShape *> (handle);

  if (!shape || shape->dims.empty ())
    return nullptr;

  return shape->dims.data ();
}

/**
 * @brief Release a mock tensor shape.
 */
Snpe_ErrorCode_t
Snpe_TensorShape_Delete (Snpe_TensorShape_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockTensorShape *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_TENSOR_SHAPE);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Get the dimensions of the given buffer attributes, as a new handle.
 */
Snpe_TensorShape_Handle_t
Snpe_IBufferAttributes_GetDims (Snpe_IBufferAttributes_Handle_t handle)
{
  MockBufferAttributes *attrs = static_cast<MockBufferAttributes *> (handle);

  if (!attrs)
    return nullptr;

  return Snpe_TensorShape_CreateDimsSize (attrs->dims.data (), attrs->dims.size ());
}

/**
 * @brief Get the element type of the given buffer attributes.
 */
Snpe_UserBufferEncoding_ElementType_t
Snpe_IBufferAttributes_GetEncodingType (Snpe_IBufferAttributes_Handle_t handle)
{
  MockBufferAttributes *attrs = static_cast<MockBufferAttributes *> (handle);

  if (!attrs)
    return SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN;

  return attrs->encoding.type;
}

/**
 * @brief Get the encoding of the given buffer attributes without owning it.
 */
Snpe_UserBufferEncoding_Handle_t
Snpe_IBufferAttributes_GetEncoding_Ref (Snpe_IBufferAttributes_Handle_t handle)
{
  MockBufferAttributes *attrs = static_cast<MockBufferAttributes *> (handle);

  if (!attrs)
    return nullptr;

  referenced_encoding = attrs->encoding;
  referenced_encoding.owned = false;

  return &referenced_encoding;
}

/**
 * @brief Release a mock buffer attributes handle.
 */
Snpe_ErrorCode_t
Snpe_IBufferAttributes_Delete (Snpe_IBufferAttributes_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockBufferAttributes *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_BUFFER_ATTRIBUTES);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Create a mock float user buffer encoding.
 */
Snpe_UserBufferEncoding_Handle_t
Snpe_UserBufferEncodingFloat_Create (void)
{
  MockEncoding *encoding = new MockEncoding ();

  encoding->type = SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT;
  encoding->step_exactly_0 = 0;
  encoding->quantized_step_size = 1.0f;
  encoding->owned = true;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_ENCODING);

  return encoding;
}

/**
 * @brief Release a mock encoding, ignoring one the caller does not own.
 */
static Snpe_ErrorCode_t
mock_encoding_delete (Snpe_UserBufferEncoding_Handle_t handle)
{
  MockEncoding *encoding = static_cast<MockEncoding *> (handle);

  if (!encoding)
    return SNPE_SUCCESS;

  if (!encoding->owned)
    return SNPE_SUCCESS;

  delete encoding;
  snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_ENCODING);

  return SNPE_SUCCESS;
}

/**
 * @brief Release a mock float user buffer encoding.
 */
Snpe_ErrorCode_t
Snpe_UserBufferEncodingFloat_Delete (Snpe_UserBufferEncoding_Handle_t handle)
{
  return mock_encoding_delete (handle);
}

/**
 * @brief Create a mock quantized user buffer encoding.
 */
Snpe_UserBufferEncoding_Handle_t
Snpe_UserBufferEncodingTfN_Create (uint64_t stepFor0, float stepSize, uint8_t bWidth)
{
  MockEncoding *encoding = new MockEncoding ();

  encoding->type = (bWidth == 8) ? SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 :
                                   SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16;
  encoding->step_exactly_0 = stepFor0;
  encoding->quantized_step_size = stepSize;
  encoding->owned = true;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_ENCODING);

  return encoding;
}

/**
 * @brief Get the quantized value that stands for zero.
 */
uint64_t
Snpe_UserBufferEncodingTfN_GetStepExactly0 (Snpe_UserBufferEncoding_Handle_t handle)
{
  return handle ? static_cast<MockEncoding *> (handle)->step_exactly_0 : 0;
}

/**
 * @brief Get the step size of a quantized encoding.
 */
float
Snpe_UserBufferEncodingTfN_GetQuantizedStepSize (Snpe_UserBufferEncoding_Handle_t handle)
{
  return handle ? static_cast<MockEncoding *> (handle)->quantized_step_size : 0.0f;
}

/**
 * @brief Release a mock quantized user buffer encoding.
 */
Snpe_ErrorCode_t
Snpe_UserBufferEncodingTfN_Delete (Snpe_UserBufferEncoding_Handle_t handle)
{
  return mock_encoding_delete (handle);
}

/**
 * @brief Create a mock user buffer of the given size and encoding.
 */
Snpe_IUserBuffer_Handle_t
Snpe_Util_CreateUserBuffer (void *buffer, size_t bufSize,
    Snpe_TensorShape_Handle_t strides, Snpe_UserBufferEncoding_Handle_t encoding)
{
  MockUserBuffer *user_buffer = new MockUserBuffer ();

  UNUSED (strides);
  user_buffer->address = buffer;
  user_buffer->size = bufSize;
  user_buffer->type = encoding ? static_cast<MockEncoding *> (encoding)->type :
                                 SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_USER_BUFFER);

  return user_buffer;
}

/**
 * @brief Point a mock user buffer at the given memory.
 */
Snpe_ErrorCode_t
Snpe_IUserBuffer_SetBufferAddress (Snpe_IUserBuffer_Handle_t handle, void *buffer)
{
  if (!handle)
    return SNPE_ERRORCODE_INTERNAL_ERROR;

  static_cast<MockUserBuffer *> (handle)->address = buffer;
  return SNPE_SUCCESS;
}

/**
 * @brief Release a mock user buffer.
 */
Snpe_ErrorCode_t
Snpe_IUserBuffer_Delete (Snpe_IUserBuffer_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockUserBuffer *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_USER_BUFFER);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Create an empty mock user buffer map.
 */
Snpe_UserBufferMap_Handle_t
Snpe_UserBufferMap_Create (void)
{
  snpe_mock_obj_created (SNPE_MOCK_OBJ_USER_BUFFER_MAP);
  return new MockUserBufferMap ();
}

/**
 * @brief Put a mock user buffer into a mock user buffer map.
 */
Snpe_ErrorCode_t
Snpe_UserBufferMap_Add (Snpe_UserBufferMap_Handle_t handle, const char *name,
    Snpe_IUserBuffer_Handle_t buffer)
{
  if (!handle || !name)
    return SNPE_ERRORCODE_INTERNAL_ERROR;

  static_cast<MockUserBufferMap *> (handle)->buffers[name]
      = static_cast<MockUserBuffer *> (buffer);
  return SNPE_SUCCESS;
}

/**
 * @brief Get a mock user buffer of a map without owning it.
 */
Snpe_IUserBuffer_Handle_t
Snpe_UserBufferMap_GetUserBuffer_Ref (Snpe_UserBufferMap_Handle_t handle, const char *name)
{
  MockUserBufferMap *map = static_cast<MockUserBufferMap *> (handle);

  if (!map || !name)
    return nullptr;

  auto it = map->buffers.find (name);
  return (it == map->buffers.end ()) ? nullptr : it->second;
}

/**
 * @brief Release a mock user buffer map, but not the buffers it refers to.
 */
Snpe_ErrorCode_t
Snpe_UserBufferMap_Delete (Snpe_UserBufferMap_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockUserBufferMap *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_USER_BUFFER_MAP);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Create an empty mock runtime list.
 */
Snpe_RuntimeList_Handle_t
Snpe_RuntimeList_Create (void)
{
  snpe_mock_obj_created (SNPE_MOCK_OBJ_RUNTIME_LIST);
  return new MockRuntimeList ();
}

/**
 * @brief Add a runtime to a mock runtime list.
 */
Snpe_ErrorCode_t
Snpe_RuntimeList_Add (Snpe_RuntimeList_Handle_t handle, Snpe_Runtime_t runtime)
{
  if (!handle)
    return SNPE_ERRORCODE_INTERNAL_ERROR;

  static_cast<MockRuntimeList *> (handle)->runtimes.push_back (runtime);
  return SNPE_SUCCESS;
}

/**
 * @brief Get the name of the given runtime.
 */
const char *
Snpe_RuntimeList_RuntimeToString (Snpe_Runtime_t runtime)
{
  switch (runtime) {
    case SNPE_RUNTIME_CPU:
      return "cpu";
    case SNPE_RUNTIME_GPU:
      return "gpu";
    case SNPE_RUNTIME_DSP:
      return "dsp";
    case SNPE_RUNTIME_AIP_FIXED8_TF:
      return "aip_fixed8_tf";
    default:
      return "unset";
  }
}

/**
 * @brief Release a mock runtime list.
 */
Snpe_ErrorCode_t
Snpe_RuntimeList_Delete (Snpe_RuntimeList_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockRuntimeList *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_RUNTIME_LIST);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Create a mock SNPE builder for the given container.
 */
Snpe_SNPEBuilder_Handle_t
Snpe_SNPEBuilder_Create (Snpe_DlContainer_Handle_t container)
{
  MockContainer *mock_container = static_cast<MockContainer *> (container);
  MockBuilder *builder = new MockBuilder ();

  builder->has_container = (mock_container != nullptr);
  if (mock_container)
    builder->model = mock_container->model;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_BUILDER);

  return builder;
}

/**
 * @brief Accept a runtime order for a mock SNPE builder.
 */
Snpe_ErrorCode_t
Snpe_SNPEBuilder_SetRuntimeProcessorOrder (
    Snpe_SNPEBuilder_Handle_t handle, Snpe_RuntimeList_Handle_t runtimeList)
{
  if (!handle || !runtimeList)
    return SNPE_ERRORCODE_INTERNAL_ERROR;

  return SNPE_SUCCESS;
}

/**
 * @brief Accept the user supplied buffer mode for a mock SNPE builder.
 */
Snpe_ErrorCode_t
Snpe_SNPEBuilder_SetUseUserSuppliedBuffers (Snpe_SNPEBuilder_Handle_t handle, int bufferMode)
{
  UNUSED (bufferMode);
  return handle ? SNPE_SUCCESS : SNPE_ERRORCODE_INTERNAL_ERROR;
}

/**
 * @brief Take the output tensor names a mock SNPE instance should expose.
 */
Snpe_ErrorCode_t
Snpe_SNPEBuilder_SetOutputTensors (
    Snpe_SNPEBuilder_Handle_t handle, Snpe_StringList_Handle_t outputTensors)
{
  MockBuilder *builder = static_cast<MockBuilder *> (handle);
  MockStringList *names = static_cast<MockStringList *> (outputTensors);

  if (!builder || !names)
    return SNPE_ERRORCODE_INTERNAL_ERROR;

  for (const std::string &name : names->items)
    if (name != SNPE_MOCK_OUTPUT_NAME)
      return SNPE_ERRORCODE_INTERNAL_ERROR;

  builder->output_names = names->items;
  return SNPE_SUCCESS;
}

/**
 * @brief Build a mock SNPE instance.
 */
Snpe_SNPE_Handle_t
Snpe_SNPEBuilder_Build (Snpe_SNPEBuilder_Handle_t handle)
{
  MockBuilder *builder = static_cast<MockBuilder *> (handle);

  if (!builder || !builder->has_container)
    return nullptr;

  if (snpe_mock_get_failure () == SNPE_MOCK_FAIL_BUILD)
    return nullptr;

  MockSNPE *snpe = new MockSNPE ();
  snpe->model = builder->model;
  snpe->input_names.push_back (SNPE_MOCK_INPUT_NAME);
  if (builder->output_names.empty ())
    snpe->output_names.push_back (SNPE_MOCK_OUTPUT_NAME);
  else
    snpe->output_names = builder->output_names;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_SNPE);

  return snpe;
}

/**
 * @brief Release a mock SNPE builder.
 */
Snpe_ErrorCode_t
Snpe_SNPEBuilder_Delete (Snpe_SNPEBuilder_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockBuilder *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_BUILDER);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Build a new mock string list holding the given names.
 */
static Snpe_StringList_Handle_t
mock_string_list_of (const std::vector<std::string> &names)
{
  MockStringList *list = new MockStringList ();

  list->items = names;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_STRING_LIST);

  return list;
}

/**
 * @brief Get the input tensor names of a mock SNPE instance.
 */
Snpe_StringList_Handle_t
Snpe_SNPE_GetInputTensorNames (Snpe_SNPE_Handle_t handle)
{
  MockSNPE *snpe = static_cast<MockSNPE *> (handle);

  return snpe ? mock_string_list_of (snpe->input_names) : nullptr;
}

/**
 * @brief Get the output tensor names of a mock SNPE instance.
 */
Snpe_StringList_Handle_t
Snpe_SNPE_GetOutputTensorNames (Snpe_SNPE_Handle_t handle)
{
  MockSNPE *snpe = static_cast<MockSNPE *> (handle);

  return snpe ? mock_string_list_of (snpe->output_names) : nullptr;
}

/**
 * @brief Get the buffer attributes of the named tensor, as a new handle.
 */
Snpe_IBufferAttributes_Handle_t
Snpe_SNPE_GetInputOutputBufferAttributes (Snpe_SNPE_Handle_t handle, const char *name)
{
  MockSNPE *snpe = static_cast<MockSNPE *> (handle);

  if (!snpe || !name)
    return nullptr;

  if (g_strcmp0 (name, SNPE_MOCK_INPUT_NAME) != 0
      && g_strcmp0 (name, SNPE_MOCK_OUTPUT_NAME) != 0)
    return nullptr;

  MockBufferAttributes *attrs = new MockBufferAttributes ();
  attrs->dims = model_dims (snpe->model);
  attrs->encoding.type = default_encoding (snpe->model);
  attrs->encoding.step_exactly_0 = 0;
  attrs->encoding.quantized_step_size = 1.0f;
  attrs->encoding.owned = false;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_BUFFER_ATTRIBUTES);

  return attrs;
}

/**
 * @brief Run the emulated model over the given user buffer maps.
 */
Snpe_ErrorCode_t
Snpe_SNPE_ExecuteUserBuffers (Snpe_SNPE_Handle_t handle,
    Snpe_UserBufferMap_Handle_t input, Snpe_UserBufferMap_Handle_t output)
{
  MockSNPE *snpe = static_cast<MockSNPE *> (handle);

  if (!snpe || !input || !output)
    return SNPE_ERRORCODE_INTERNAL_ERROR;

  const MockUserBuffer *in = static_cast<const MockUserBuffer *> (
      Snpe_UserBufferMap_GetUserBuffer_Ref (input, snpe->input_names[0].c_str ()));

  for (const std::string &name : snpe->output_names) {
    MockUserBuffer *out = static_cast<MockUserBuffer *> (
        Snpe_UserBufferMap_GetUserBuffer_Ref (output, name.c_str ()));
    add2 (in, out);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Release a mock SNPE instance.
 */
Snpe_ErrorCode_t
Snpe_SNPE_Delete (Snpe_SNPE_Handle_t handle)
{
  if (handle) {
    delete static_cast<MockSNPE *> (handle);
    snpe_mock_obj_destroyed (SNPE_MOCK_OBJ_SNPE);
  }

  return SNPE_SUCCESS;
}

/**
 * @brief Get the version of the mock SDK.
 */
Snpe_DlVersion_Handle_t
Snpe_Util_GetLibraryVersion (void)
{
  MockVersion *version = new MockVersion ();

  version->major = 2;
  snpe_mock_obj_created (SNPE_MOCK_OBJ_VERSION);

  return version;
}

/**
 * @brief Tell whether the given runtime is available; only the CPU one is.
 */
int
Snpe_Util_IsRuntimeAvailable (Snpe_Runtime_t runtime)
{
  return (runtime == SNPE_RUNTIME_CPU) ? 1 : 0;
}
