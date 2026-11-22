/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    trix_mock.h
 * @date    22 Nov 2026
 * @brief   Controls of the mock npu-engine for the trix-engine sub-plugin tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The mock reads a text model instead of a compiled one:
 *
 *   trix-mock-model
 *   in  <d0> <d1> <d2> <d3>
 *   out <d0> <d1> <d2> <d3>
 *
 * with one line per segment (tensor), dimensions outermost first. A request
 * checks every buffer and data info it is given against the model, and its
 * output segment k gets bytes j = (byte j % N of input 0) + k + 1, in
 * buffers allocated with malloc ().
 */
#ifndef __TRIX_MOCK_H__
#define __TRIX_MOCK_H__

#include <libnpuhost.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Clear the injected failures and the call counters.
 */
void trix_mock_reset (void);

/**
 * @brief Make every later call of the named npu-engine function fail.
 */
void trix_mock_fail (const char *function);

/**
 * @brief Make only the next call of the named npu-engine function fail.
 */
void trix_mock_fail_next (const char *function);

/**
 * @brief The number of calls of the named function since the last reset.
 */
int trix_mock_calls (const char *function);

/**
 * @brief The number of open devices, registered models and requests not
 *        removed.
 */
int trix_mock_live_objects (void);

/**
 * @brief The device type the last getNPUdeviceByTypeAny () call asked for.
 */
dev_type trix_mock_last_device_type (void);

/**
 * @brief The data info of segment @a idx of the last setNPU_requestData ()
 *        call, of the input (@a input true) or of the output.
 * @return 0 if there is such a segment, -1 if not.
 */
int trix_mock_last_data_info (bool input, uint32_t idx, data_layout *layout, data_type *type);

#ifdef __cplusplus
}
#endif

#endif /* __TRIX_MOCK_H__ */
