/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    libnpuhost.h
 * @date    22 Nov 2026
 * @brief   The host API of npu-engine, reduced for the mock tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * This header stands in for libnpuhost.h of npu-engine (the TRIx NPU
 * runtime) when the trix-engine sub-plugin is built for the tests in this
 * directory. npu-engine is not public; the names here are those
 * tensor_filter_trix_engine.cc uses, and the types, members and values are
 * written from that use, not copied from npu-engine. mock/trix_mock.cc
 * implements the functions.
 */
#ifndef __LIBNPUHOST_MOCK_H__
#define __LIBNPUHOST_MOCK_H__

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "npubinfmt.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief A handle of an NPU device. */
typedef void *npudev_h;

/** @brief Kinds of NPU device and connection. */
typedef enum {
  NPUCOND_CONN_UNKNOWN = 0,
  NPUCOND_TRIV2_CONN_SOCIP,
} dev_type;

/** @brief Data layouts of a tensor. */
typedef enum {
  DATA_LAYOUT_MODEL = 0, /**< the layout of the model */
  DATA_LAYOUT_NHWC,
  DATA_LAYOUT_NCHW,
} data_layout;

/** @brief Data types of a tensor. */
typedef enum {
  DATA_TYPE_MODEL = 0, /**< the type of the model */
  DATA_TYPE_INT8,
  DATA_TYPE_UINT8,
  DATA_TYPE_INT16,
  DATA_TYPE_UINT16,
  DATA_TYPE_INT32,
  DATA_TYPE_UINT32,
  DATA_TYPE_INT64,
  DATA_TYPE_UINT64,
  DATA_TYPE_FLOAT32,
  DATA_TYPE_FLOAT64,
} data_type;

/** @brief Kinds of buffer. */
typedef enum {
  BUFFER_FILE = 0, /**< a file named by filepath */
  BUFFER_MAPPED, /**< memory at addr */
} buffer_types;

/** @brief A buffer of a model or a tensor. */
typedef struct {
  const char *filepath;
  void *addr;
  uint64_t size;
  buffer_types type;
} generic_buffer;

/** @brief The buffers of the input tensors of a request. */
typedef struct {
  uint32_t num_buffers;
  generic_buffer bufs[TRIX_MOCK_MAX_SEGMENTS];
} input_buffers;

/** @brief The buffers of the output tensors of a request. */
typedef input_buffers output_buffers;

/** @brief The layout and the type of a tensor. */
typedef struct {
  data_layout layout;
  data_type type;
} data_info;

/** @brief The layouts and the types of the tensors of a request. */
typedef struct {
  uint32_t num_info;
  data_info info[TRIX_MOCK_MAX_SEGMENTS];
} tensors_data_info;

/** @brief Read the metadata of a model file; free it with free (). NULL on error. */
npubin_meta *getNPUmodel_metadata (const char *model, bool i64);

/** @brief Open any device of @a type with at least @a tops TOPS. */
int getNPUdeviceByTypeAny (npudev_h *dev, dev_type type, uint32_t tops);

/** @brief Close a device; its models and requests go with it. */
int putNPUdevice (npudev_h dev);

/** @brief Load a model into a device. */
int registerNPUmodel (npudev_h dev, generic_buffer *model, uint32_t *model_id);

/** @brief Unload every model of a device. */
int unregisterNPUmodel_all (npudev_h dev);

/** @brief Create a request to run a model. */
int createNPU_request (npudev_h dev, uint32_t model_id, int *req_id);

/**
 * @brief Set the input and the output of a request. The device fills
 *        @a output with buffers it allocates, freed with free () by the caller.
 */
int setNPU_requestData (npudev_h dev, int req_id, input_buffers *input,
    tensors_data_info *in_info, output_buffers *output, tensors_data_info *out_info);

/** @brief Run a request and wait for it. */
int submitNPU_request (npudev_h dev, int req_id);

/** @brief Remove a request. */
int removeNPU_request (npudev_h dev, int req_id);

#ifdef __cplusplus
}
#endif

#endif /* __LIBNPUHOST_MOCK_H__ */
