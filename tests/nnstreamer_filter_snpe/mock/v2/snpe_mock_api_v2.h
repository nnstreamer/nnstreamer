/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    snpe_mock_api_v2.h
 * @date    22 Sep 2026
 * @brief   Mock of the SNPE 2.x C API surface used by tensor_filter_snpe.cc.
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * Written from the vendor's public C API reference of the Qualcomm Neural
 * Processing SDK (https://docs.qualcomm.com/); no header of the proprietary
 * SDK is copied here. Only the declarations that
 * ext/nnstreamer/tensor_filter/tensor_filter_snpe.cc refers to are present, and
 * the enumerator values are illustrative: binary compatibility with the real
 * SDK is not a goal, because a test binary uses these headers and the mock
 * library as one pair.
 *
 * Handle ownership follows the vendor reference, because the unit tests assert
 * on it: a handle returned by a _Create or by a getter belongs to the caller
 * and has to be passed to the matching _Delete, while a handle returned by a
 * _Ref accessor belongs to the object it came from and must not be released.
 * The mock ignores a non-owning handle that is handed to a _Delete rather than
 * freeing it, so that such a call cannot turn into a crash that would hide the
 * defect a test is looking at.
 */
#ifndef __NNS_SNPE_MOCK_API_V2_H__
#define __NNS_SNPE_MOCK_API_V2_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SNPE_SUCCESS = 0,
  SNPE_ERRORCODE_INTERNAL_ERROR = 1
} Snpe_ErrorCode_t;

typedef enum {
  SNPE_RUNTIME_CPU = 0,
  SNPE_RUNTIME_GPU = 1,
  SNPE_RUNTIME_DSP = 2,
  SNPE_RUNTIME_AIP_FIXED8_TF = 3,
  SNPE_RUNTIME_UNSET = 8
} Snpe_Runtime_t;

typedef enum {
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN = 0,
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT = 1,
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNSIGNED8BIT = 2,
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16 = 3,
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 = 10,
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16 = 11
} Snpe_UserBufferEncoding_ElementType_t;

typedef void *Snpe_DlContainer_Handle_t;
typedef void *Snpe_DlVersion_Handle_t;
typedef void *Snpe_IBufferAttributes_Handle_t;
typedef void *Snpe_IUserBuffer_Handle_t;
typedef void *Snpe_RuntimeList_Handle_t;
typedef void *Snpe_SNPEBuilder_Handle_t;
typedef void *Snpe_SNPE_Handle_t;
typedef void *Snpe_StringList_Handle_t;
typedef void *Snpe_TensorShape_Handle_t;
typedef void *Snpe_UserBufferEncoding_Handle_t;
typedef void *Snpe_UserBufferMap_Handle_t;

/** @brief Open a mock container for the given model file. */
Snpe_DlContainer_Handle_t Snpe_DlContainer_Open (const char *filename);
/** @brief Release a mock container. */
Snpe_ErrorCode_t Snpe_DlContainer_Delete (Snpe_DlContainer_Handle_t handle);

/** @brief Get the major number of the mock SDK version. */
int32_t Snpe_DlVersion_GetMajor (Snpe_DlVersion_Handle_t handle);
/** @brief Get the mock SDK version as a string. */
const char *Snpe_DlVersion_ToString (Snpe_DlVersion_Handle_t handle);
/** @brief Release a mock version handle. */
Snpe_ErrorCode_t Snpe_DlVersion_Delete (Snpe_DlVersion_Handle_t handle);

/** @brief Create an empty mock string list. */
Snpe_StringList_Handle_t Snpe_StringList_Create (void);
/** @brief Append a string to a mock string list. */
Snpe_ErrorCode_t Snpe_StringList_Append (Snpe_StringList_Handle_t handle, const char *string);
/** @brief Get the number of strings in a mock string list. */
size_t Snpe_StringList_Size (Snpe_StringList_Handle_t handle);
/** @brief Get the string at the given index of a mock string list. */
const char *Snpe_StringList_At (Snpe_StringList_Handle_t handle, size_t idx);
/** @brief Release a mock string list. */
Snpe_ErrorCode_t Snpe_StringList_Delete (Snpe_StringList_Handle_t handle);

/** @brief Create a mock tensor shape from the given dimensions. */
Snpe_TensorShape_Handle_t Snpe_TensorShape_CreateDimsSize (const size_t *dims, size_t size);
/** @brief Get the rank of a mock tensor shape. */
size_t Snpe_TensorShape_Rank (Snpe_TensorShape_Handle_t handle);
/** @brief Get the dimensions of a mock tensor shape. */
const size_t *Snpe_TensorShape_GetDimensions (Snpe_TensorShape_Handle_t handle);
/** @brief Release a mock tensor shape. */
Snpe_ErrorCode_t Snpe_TensorShape_Delete (Snpe_TensorShape_Handle_t handle);

/** @brief Get the dimensions of the given buffer attributes, as a new handle. */
Snpe_TensorShape_Handle_t Snpe_IBufferAttributes_GetDims (
    Snpe_IBufferAttributes_Handle_t handle);
/** @brief Get the element type of the given buffer attributes. */
Snpe_UserBufferEncoding_ElementType_t Snpe_IBufferAttributes_GetEncodingType (
    Snpe_IBufferAttributes_Handle_t handle);
/** @brief Get the encoding of the given buffer attributes without owning it. */
Snpe_UserBufferEncoding_Handle_t Snpe_IBufferAttributes_GetEncoding_Ref (
    Snpe_IBufferAttributes_Handle_t handle);
/** @brief Release a mock buffer attributes handle. */
Snpe_ErrorCode_t Snpe_IBufferAttributes_Delete (Snpe_IBufferAttributes_Handle_t handle);

/** @brief Create a mock float user buffer encoding. */
Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingFloat_Create (void);
/** @brief Release a mock float user buffer encoding. */
Snpe_ErrorCode_t Snpe_UserBufferEncodingFloat_Delete (Snpe_UserBufferEncoding_Handle_t handle);
/** @brief Create a mock quantized user buffer encoding. */
Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingTfN_Create (
    uint64_t stepFor0, float stepSize, uint8_t bWidth);
/** @brief Get the quantized value that stands for zero. */
uint64_t Snpe_UserBufferEncodingTfN_GetStepExactly0 (Snpe_UserBufferEncoding_Handle_t handle);
/** @brief Get the step size of a quantized encoding. */
float Snpe_UserBufferEncodingTfN_GetQuantizedStepSize (Snpe_UserBufferEncoding_Handle_t handle);
/** @brief Release a mock quantized user buffer encoding. */
Snpe_ErrorCode_t Snpe_UserBufferEncodingTfN_Delete (Snpe_UserBufferEncoding_Handle_t handle);

/** @brief Point a mock user buffer at the given memory. */
Snpe_ErrorCode_t Snpe_IUserBuffer_SetBufferAddress (
    Snpe_IUserBuffer_Handle_t handle, void *buffer);
/** @brief Release a mock user buffer. */
Snpe_ErrorCode_t Snpe_IUserBuffer_Delete (Snpe_IUserBuffer_Handle_t handle);

/** @brief Create an empty mock user buffer map. */
Snpe_UserBufferMap_Handle_t Snpe_UserBufferMap_Create (void);
/** @brief Put a mock user buffer into a mock user buffer map. */
Snpe_ErrorCode_t Snpe_UserBufferMap_Add (Snpe_UserBufferMap_Handle_t handle,
    const char *name, Snpe_IUserBuffer_Handle_t buffer);
/** @brief Get a mock user buffer of a map without owning it. */
/** @brief Get a mock user buffer of a map without owning it. */
Snpe_IUserBuffer_Handle_t Snpe_UserBufferMap_GetUserBuffer_Ref (
    Snpe_UserBufferMap_Handle_t handle, const char *name);
/** @brief Release a mock user buffer map, but not the buffers it refers to. */
Snpe_ErrorCode_t Snpe_UserBufferMap_Delete (Snpe_UserBufferMap_Handle_t handle);

/** @brief Create an empty mock runtime list. */
Snpe_RuntimeList_Handle_t Snpe_RuntimeList_Create (void);
/** @brief Add a runtime to a mock runtime list. */
Snpe_ErrorCode_t Snpe_RuntimeList_Add (Snpe_RuntimeList_Handle_t handle, Snpe_Runtime_t runtime);
/** @brief Get the name of the given runtime. */
const char *Snpe_RuntimeList_RuntimeToString (Snpe_Runtime_t runtime);
/** @brief Release a mock runtime list. */
Snpe_ErrorCode_t Snpe_RuntimeList_Delete (Snpe_RuntimeList_Handle_t handle);

/** @brief Create a mock SNPE builder for the given container. */
Snpe_SNPEBuilder_Handle_t Snpe_SNPEBuilder_Create (Snpe_DlContainer_Handle_t container);
/** @brief Accept a runtime order for a mock SNPE builder. */
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetRuntimeProcessorOrder (
    Snpe_SNPEBuilder_Handle_t handle, Snpe_RuntimeList_Handle_t runtimeList);
/** @brief Accept the user supplied buffer mode for a mock SNPE builder. */
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetUseUserSuppliedBuffers (
    Snpe_SNPEBuilder_Handle_t handle, int bufferMode);
/** @brief Take the output tensor names a mock SNPE instance should expose. */
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetOutputTensors (
    Snpe_SNPEBuilder_Handle_t handle, Snpe_StringList_Handle_t outputTensors);
/** @brief Build a mock SNPE instance. */
Snpe_SNPE_Handle_t Snpe_SNPEBuilder_Build (Snpe_SNPEBuilder_Handle_t handle);
/** @brief Release a mock SNPE builder. */
Snpe_ErrorCode_t Snpe_SNPEBuilder_Delete (Snpe_SNPEBuilder_Handle_t handle);

/** @brief Get the input tensor names of a mock SNPE instance. */
Snpe_StringList_Handle_t Snpe_SNPE_GetInputTensorNames (Snpe_SNPE_Handle_t handle);
/** @brief Get the output tensor names of a mock SNPE instance. */
Snpe_StringList_Handle_t Snpe_SNPE_GetOutputTensorNames (Snpe_SNPE_Handle_t handle);
/** @brief Get the buffer attributes of the named tensor, as a new handle. */
Snpe_IBufferAttributes_Handle_t Snpe_SNPE_GetInputOutputBufferAttributes (
    Snpe_SNPE_Handle_t handle, const char *name);
/** @brief Run the emulated model over the given user buffer maps. */
Snpe_ErrorCode_t Snpe_SNPE_ExecuteUserBuffers (Snpe_SNPE_Handle_t handle,
    Snpe_UserBufferMap_Handle_t input, Snpe_UserBufferMap_Handle_t output);
/** @brief Release a mock SNPE instance. */
Snpe_ErrorCode_t Snpe_SNPE_Delete (Snpe_SNPE_Handle_t handle);

/** @brief Get the version of the mock SDK. */
Snpe_DlVersion_Handle_t Snpe_Util_GetLibraryVersion (void);
/** @brief Tell whether the given runtime is available; only the CPU one is. */
int Snpe_Util_IsRuntimeAvailable (Snpe_Runtime_t runtime);
/** @brief Create a mock user buffer of the given size and encoding. */
Snpe_IUserBuffer_Handle_t Snpe_Util_CreateUserBuffer (void *buffer, size_t bufSize,
    Snpe_TensorShape_Handle_t strides, Snpe_UserBufferEncoding_Handle_t encoding);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_SNPE_MOCK_API_V2_H__ */
