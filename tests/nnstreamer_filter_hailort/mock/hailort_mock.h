/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    hailort_mock.h
 * @date    28 Oct 2026
 * @brief   Controls of the mock HailoRT library for the sub-plugin tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The mock reads a text "HEF" instead of a compiled model:
 *
 *   hailort-mock-hef
 *   network_group <name>
 *   input  <name> <uint8|uint16|float32> <nhwc|nhw|nc|nchw|nms> <height> <width> <features>
 *   output <name> <uint8|uint16|float32> <nhwc|nhw|nc|nchw|nms> <height> <width> <features>
 *
 * Every "network_group" line adds a group; the streams belong to the first one.
 * '#' starts a comment line. Once every input has a frame, each output gets one
 * frame whose element j is (element j % N of input 0) + (the output index),
 * converted between the user formats of the two streams. Reading an output
 * that has no frame returns HAILO_TIMEOUT.
 */
#ifndef __HAILORT_MOCK_H__
#define __HAILORT_MOCK_H__

#include <hailo/hailort.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Clear the injected failures, overrides and call counters.
 */
HAILORTAPI void hailort_mock_reset (void);

/**
 * @brief Make every later call of the named HailoRT function return @a status.
 */
HAILORTAPI void hailort_mock_fail (const char *function, hailo_status status);

/**
 * @brief Make every later frame-size query report @a frame_size until reset.
 */
HAILORTAPI void hailort_mock_override_frame_size (size_t frame_size);

/**
 * @brief Make every later user-format query report @a type (AUTO to stop).
 */
HAILORTAPI void hailort_mock_override_user_type (hailo_format_type_t type);

/**
 * @brief The number of calls of the named HailoRT function since the last reset.
 */
HAILORTAPI int hailort_mock_calls (const char *function);

/**
 * @brief The number of virtual devices, HEFs and virtual streams not released yet.
 */
HAILORTAPI int hailort_mock_live_objects (void);

/**
 * @brief The group id the last virtual device was created with ("" for NULL).
 * @return A newly allocated string; free it with g_free().
 */
HAILORTAPI char *hailort_mock_last_group_id (void);

#ifdef __cplusplus
}
#endif

#endif /* __HAILORT_MOCK_H__ */
