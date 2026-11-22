/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    QnnInterface.h
 * @date    22 Nov 2026
 * @brief   A reduced QNN (Qualcomm AI Engine Direct) API for the mock tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * This header stands in for QnnInterface.h of the QNN SDK when the qnn
 * sub-plugin is built for the tests in this directory. It declares only what
 * tensor_filter_qnn.cc and the mock libraries use, with the type names,
 * member names and data type values of the QNN 2.x API. The interface table
 * holds only the functions the sub-plugin and the mock model call, so a
 * library built against this header cannot load a real QNN backend.
 *
 * It is written from the QNN 2.x API reference and is not checked against a
 * particular SDK release. Of the quantization encodings it declares only
 * scale-offset and axis-scale-offset, the two the sub-plugin copies; the
 * sub-plugin copies any other encoding (e.g., bit-width or block) as
 * QNN_QUANTIZATION_ENCODING_UNDEFINED.
 */
#ifndef __QNN_INTERFACE_MOCK_H__
#define __QNN_INTERFACE_MOCK_H__

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t Qnn_ErrorHandle_t;
#define QNN_SUCCESS 0
#define QNN_GRAPH_NO_ERROR QNN_SUCCESS

typedef void *Qnn_Handle_t;
typedef Qnn_Handle_t Qnn_BackendHandle_t;
typedef Qnn_Handle_t Qnn_ContextHandle_t;
typedef Qnn_Handle_t Qnn_GraphHandle_t;
typedef Qnn_Handle_t Qnn_LogHandle_t;
typedef Qnn_Handle_t Qnn_DeviceHandle_t;
typedef Qnn_Handle_t Qnn_ProfileHandle_t;
typedef Qnn_Handle_t Qnn_SignalHandle_t;
typedef void *Qnn_MemHandle_t;

typedef enum {
  QNN_DATATYPE_INT_8 = 0x0008,
  QNN_DATATYPE_INT_16 = 0x0016,
  QNN_DATATYPE_INT_32 = 0x0032,
  QNN_DATATYPE_INT_64 = 0x0064,
  QNN_DATATYPE_UINT_8 = 0x0108,
  QNN_DATATYPE_UINT_16 = 0x0116,
  QNN_DATATYPE_UINT_32 = 0x0132,
  QNN_DATATYPE_UINT_64 = 0x0164,
  QNN_DATATYPE_FLOAT_16 = 0x0216,
  QNN_DATATYPE_FLOAT_32 = 0x0232,
  QNN_DATATYPE_FLOAT_64 = 0x0264,
  QNN_DATATYPE_SFIXED_POINT_4 = 0x0304,
  QNN_DATATYPE_SFIXED_POINT_8 = 0x0308,
  QNN_DATATYPE_SFIXED_POINT_16 = 0x0316,
  QNN_DATATYPE_SFIXED_POINT_32 = 0x0332,
  QNN_DATATYPE_UFIXED_POINT_4 = 0x0404,
  QNN_DATATYPE_UFIXED_POINT_8 = 0x0408,
  QNN_DATATYPE_UFIXED_POINT_16 = 0x0416,
  QNN_DATATYPE_UFIXED_POINT_32 = 0x0432,
  QNN_DATATYPE_BOOL_8 = 0x0508,
  QNN_DATATYPE_STRING = 0x0608,
  QNN_DATATYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_DataType_t;

typedef enum {
  QNN_TENSOR_VERSION_1 = 1,
  QNN_TENSOR_VERSION_2 = 2,
  QNN_TENSOR_VERSION_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorVersion_t;

typedef enum {
  QNN_TENSOR_TYPE_APP_WRITE = 0,
  QNN_TENSOR_TYPE_APP_READ = 1,
  QNN_TENSOR_TYPE_APP_READWRITE = 2,
  QNN_TENSOR_TYPE_NATIVE = 3,
  QNN_TENSOR_TYPE_STATIC = 4,
  QNN_TENSOR_TYPE_NULL = 5,
  QNN_TENSOR_TYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorType_t;

typedef uint32_t Qnn_TensorDataFormat_t;

typedef enum {
  QNN_TENSORMEMTYPE_RAW = 0,
  QNN_TENSORMEMTYPE_MEMHANDLE = 1,
  QNN_TENSORMEMTYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorMemType_t;

typedef enum {
  QNN_DEFINITION_IMPL_GENERATED = 0,
  QNN_DEFINITION_DEFINED = 1,
  QNN_DEFINITION_UNDEFINED = 0x7FFFFFFF
} Qnn_Definition_t;

typedef enum {
  QNN_QUANTIZATION_ENCODING_SCALE_OFFSET = 0,
  QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET = 1,
  QNN_QUANTIZATION_ENCODING_UNDEFINED = 0x7FFFFFFF
} Qnn_QuantizationEncoding_t;

/** @brief Scale and offset of a quantized tensor. */
typedef struct {
  float scale;
  int32_t offset;
} Qnn_ScaleOffset_t;

/** @brief Per-axis scales and offsets of a quantized tensor. */
typedef struct {
  int32_t axis;
  uint32_t numScaleOffsets;
  Qnn_ScaleOffset_t *scaleOffset;
} Qnn_AxisScaleOffset_t;

/** @brief Quantization parameters of a tensor. */
typedef struct {
  Qnn_Definition_t encodingDefinition;
  Qnn_QuantizationEncoding_t quantizationEncoding;
  union {
    Qnn_ScaleOffset_t scaleOffsetEncoding;
    Qnn_AxisScaleOffset_t axisScaleOffsetEncoding;
  };
} Qnn_QuantizeParams_t;

#define QNN_QUANTIZE_PARAMS_INIT { QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, { { 0.0f, 0 } } }

/** @brief A buffer of the application. */
typedef struct {
  void *data;
  uint32_t dataSize;
} Qnn_ClientBuffer_t;

/** @brief Sparse layout of a tensor (only its storage matters here). */
typedef struct {
  uint32_t type;
  uint32_t numSpecifiedElements;
  uint32_t numSparseDimensions;
} Qnn_SparseParams_t;

/** @brief Version 1 of a tensor. */
typedef struct {
  uint32_t id;
  const char *name;
  Qnn_TensorType_t type;
  Qnn_TensorDataFormat_t dataFormat;
  Qnn_DataType_t dataType;
  Qnn_QuantizeParams_t quantizeParams;
  uint32_t rank;
  uint32_t *dimensions;
  Qnn_TensorMemType_t memType;
  union {
    Qnn_ClientBuffer_t clientBuf;
    Qnn_MemHandle_t memHandle;
  };
} Qnn_TensorV1_t;

/** @brief Version 2 of a tensor: version 1 plus dynamic dimensions and sparsity. */
typedef struct {
  uint32_t id;
  const char *name;
  Qnn_TensorType_t type;
  Qnn_TensorDataFormat_t dataFormat;
  Qnn_DataType_t dataType;
  Qnn_QuantizeParams_t quantizeParams;
  uint32_t rank;
  uint32_t *dimensions;
  Qnn_TensorMemType_t memType;
  union {
    Qnn_ClientBuffer_t clientBuf;
    Qnn_MemHandle_t memHandle;
  };
  uint8_t *isDynamicDimensions;
  Qnn_SparseParams_t sparseParams;
  uint8_t isProduced;
} Qnn_TensorV2_t;

/** @brief A tensor of either version. */
typedef struct {
  Qnn_TensorVersion_t version;
  union {
    Qnn_TensorV1_t v1;
    Qnn_TensorV2_t v2;
  };
} Qnn_Tensor_t;

#define QNN_TENSOR_INIT { QNN_TENSOR_VERSION_1, {} }

typedef struct _QnnBackend_Config_t QnnBackend_Config_t;
typedef struct _QnnContext_Config_t QnnContext_Config_t;
typedef struct _QnnGraph_Config_t QnnGraph_Config_t;

typedef enum {
  QNN_LOG_LEVEL_ERROR = 1,
  QNN_LOG_LEVEL_WARN = 2,
  QNN_LOG_LEVEL_INFO = 3,
  QNN_LOG_LEVEL_VERBOSE = 4,
  QNN_LOG_LEVEL_DEBUG = 5,
  QNN_LOG_LEVEL_MAX = 0x7FFFFFFF
} QnnLog_Level_t;

typedef void (*QnnLog_Callback_t) (const char *fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args);

typedef Qnn_ErrorHandle_t (*QnnError_GetMessageFn_t) (Qnn_ErrorHandle_t errorHandle, const char **errorMessage);
typedef Qnn_ErrorHandle_t (*QnnBackend_CreateFn_t) (Qnn_LogHandle_t logger, const QnnBackend_Config_t **config, Qnn_BackendHandle_t *backend);
typedef Qnn_ErrorHandle_t (*QnnBackend_FreeFn_t) (Qnn_BackendHandle_t backend);
typedef Qnn_ErrorHandle_t (*QnnContext_CreateFn_t) (Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, const QnnContext_Config_t **config, Qnn_ContextHandle_t *context);
typedef Qnn_ErrorHandle_t (*QnnContext_FreeFn_t) (Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile);
typedef Qnn_ErrorHandle_t (*QnnGraph_CreateFn_t) (Qnn_ContextHandle_t contextHandle, const char *graphName, const QnnGraph_Config_t **config, Qnn_GraphHandle_t *graphHandle);
typedef Qnn_ErrorHandle_t (*QnnGraph_FinalizeFn_t) (Qnn_GraphHandle_t graphHandle, Qnn_ProfileHandle_t profileHandle, Qnn_SignalHandle_t signalHandle);
typedef Qnn_ErrorHandle_t (*QnnGraph_ExecuteFn_t) (Qnn_GraphHandle_t graphHandle, const Qnn_Tensor_t *inputs, uint32_t numInputs, Qnn_Tensor_t *outputs, uint32_t numOutputs, Qnn_ProfileHandle_t profileHandle, Qnn_SignalHandle_t signalHandle);

/** @brief The functions of a backend (a subset of the QNN 2.x interface). */
typedef struct {
  QnnError_GetMessageFn_t errorGetMessage;
  QnnBackend_CreateFn_t backendCreate;
  QnnBackend_FreeFn_t backendFree;
  QnnContext_CreateFn_t contextCreate;
  QnnContext_FreeFn_t contextFree;
  QnnGraph_CreateFn_t graphCreate;
  QnnGraph_FinalizeFn_t graphFinalize;
  QnnGraph_ExecuteFn_t graphExecute;
} QnnInterface_ImplementationV2_t;

#define QNN_INTERFACE_VER_TYPE QnnInterface_ImplementationV2_t
#define QNN_INTERFACE_VER_NAME v2
#define QNN_INTERFACE_VER_TYPE_INIT { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL }

/** @brief A version of the QNN API. */
typedef struct {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
} Qnn_Version_t;

/** @brief Versions of the core and backend APIs of a provider. */
typedef struct {
  Qnn_Version_t coreApiVersion;
  Qnn_Version_t backendApiVersion;
} Qnn_ApiVersion_t;

/** @brief A provider of the QNN interface, as a backend library lists it. */
typedef struct {
  uint32_t backendId;
  const char *providerName;
  Qnn_ApiVersion_t apiVersion;
  union {
    QNN_INTERFACE_VER_TYPE QNN_INTERFACE_VER_NAME;
  };
} QnnInterface_t;

/**
 * @brief List the interface providers of a backend library.
 */
Qnn_ErrorHandle_t QnnInterface_getProviders (const QnnInterface_t ***providerList, uint32_t *numProviders);

#ifdef __cplusplus
}
#endif

#endif /* __QNN_INTERFACE_MOCK_H__ */
