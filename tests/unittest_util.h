/**
 * @file        unittest_util.h
 * @date        15 Jan 2019
 * @brief       Unit test utility.
 * @see         https://github.com/nnsuite/nnstreamer
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

/**
 * @brief Set pipeline state, wait until it's done.
 * @return 0 success, -EPIPE if failed, -ETIME if timeout happens.
 */
extern int setPipelineStateSync (GstElement *pipeline, GstState state, uint32_t timeout_ms);

/**
 * @brief Get temp file name.
 * @return file name (should free string with g_free)
 */
extern gchar * _get_temp_filename (void);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _NNS_UNITTEST_UTIL_H__ */

