/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    npubinfmt.h
 * @date    22 Nov 2026
 * @brief   The model metadata of npu-engine, reduced for the mock tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * This header stands in for npubinfmt.h of npu-engine (the TRIx NPU
 * runtime, pkg-config 'npu-engine') when the trix-engine sub-plugin is built
 * for the tests in this directory. npu-engine is not public; this header is
 * written from what tensor_filter_trix_engine.cc uses and declares nothing
 * else. The array bounds are those of the mock, not of npu-engine.
 */
#ifndef __NPUBINFMT_MOCK_H__
#define __NPUBINFMT_MOCK_H__

#include <stdint.h>

/** @brief The rank of a segment (tensor) of a model. */
#define MAX_RANK (4)

/** @brief The most input or output segments a mock model has. */
#define TRIX_MOCK_MAX_SEGMENTS (16)

/** @brief The metadata of a model: its size and its segments (tensors). */
typedef struct {
  uint64_t size; /**< the size of the model file */
  uint32_t input_seg_num; /**< the number of input segments */
  uint32_t input_seg_dims[TRIX_MOCK_MAX_SEGMENTS][MAX_RANK]; /**< outermost first */
  uint32_t output_seg_num; /**< the number of output segments */
  uint32_t output_seg_dims[TRIX_MOCK_MAX_SEGMENTS][MAX_RANK]; /**< outermost first */
} npubin_meta;

#endif /* __NPUBINFMT_MOCK_H__ */
