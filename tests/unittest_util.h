/**
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file        unittest_util.h
 * @date        15 Jan 2019
 * @brief       Unit test utility.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#ifndef _NNS_UNITTEST_UTIL_H__
#define _NNS_UNITTEST_UTIL_H__
#include <gst/gst.h>
#include <glib.h>
#include <stdint.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef DBG
#define DBG FALSE
#endif
#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

#define UNITTEST_STATECHANGE_TIMEOUT (500U)
#define TEST_DEFAULT_SLEEP_TIME (10000U)

/**
 * @brief Set pipeline state, wait until it's done.
 * @return 0 success, -EPIPE if failed, -ETIME if timeout happens.
 */
extern int setPipelineStateSync (GstElement *pipeline, GstState state, uint32_t timeout_ms);

/**
 * @brief Get temp file name.
 * @return file name (should free string with g_free)
 */
extern gchar * getTempFilename (void);

/**
 * @brief Wait until the pipeline processing the buffers
 * @return TRUE on success, FALSE when a time-out occurs
 */
extern gboolean wait_pipeline_process_buffers (const guint * data_received, guint expected_num_buffers, guint timeout_ms);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _NNS_UNITTEST_UTIL_H__ */
