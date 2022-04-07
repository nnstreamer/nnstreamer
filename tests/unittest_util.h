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
#include <glib/gstdio.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef DBG
#define DBG FALSE
#endif
#define _print_log(...) do { if (DBG) g_message (__VA_ARGS__); } while (0)

#define UNITTEST_STATECHANGE_TIMEOUT (2000U)
#define TEST_DEFAULT_SLEEP_TIME (10000U)
#define TEST_TIMEOUT_LIMIT (10000000U) /* 10 secs */

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

/**
 * @brief Replaces string.
 * This function deallocates the input source string.
 * @param[in] source The input string. This will be freed when returning the replaced string.
 * @param[in] what The string to search for.
 * @param[in] to The string to be replaced.
 * @param[in] delimiters The characters which specify the place to split the string. Set NULL to replace all matched string.
 * @param[out] count The count of replaced. Set NULL if it is unnecessary.
 * @return Newly allocated string. The returned string should be freed with g_free().
 */
extern gchar *
replace_string (gchar * source, const gchar * what, const gchar * to, const gchar * delimiters, guint * count);

/**
 * @brief Wait until the pipeline saving the file
 * @return TRUE on success, FALSE when a time-out occurs
 */
#define _wait_pipeline_save_files(file, content, len, exp_len, timeout_ms) \
  do {                                                                     \
    guint timer = 0;                                                       \
    guint load_failed = 0;                                                 \
    guint tick = TEST_DEFAULT_SLEEP_TIME / 1000U;                          \
    if (tick == 0)                                                         \
      tick = 1;                                                            \
    do {                                                                   \
      g_usleep (TEST_DEFAULT_SLEEP_TIME);                                  \
      if (!g_file_get_contents (file, &content, &len, NULL)) {             \
        if (load_failed)                                                   \
          break;                                                           \
        load_failed++;                                                     \
      }                                                                    \
      timer += tick;                                                       \
      if (timer > timeout_ms) {                                            \
        EXPECT_GE (timeout_ms, timer);                                     \
        break;                                                             \
      }                                                                    \
      if (len < exp_len) {                                                 \
        g_free (content);                                                  \
        content = NULL;                                                    \
      }                                                                    \
    } while (len < exp_len);                                               \
  } while (0)


#ifdef FAKEDLOG
/**
 * @brief enum definition copied from Tizen dlog (MIT License)
 * @detail If real dlog is included, this will generate errors.
 *         Do not include real dlog.
 */
typedef enum {
        DLOG_UNKNOWN = 0, /**< Keep this always at the start */
        DLOG_DEFAULT, /**< Default */
        DLOG_VERBOSE, /**< Verbose */
        DLOG_DEBUG, /**< Debug */
        DLOG_INFO, /**< Info */
        DLOG_WARN, /**< Warning */
        DLOG_ERROR, /**< Error */
        DLOG_FATAL, /**< Fatal */
        DLOG_SILENT, /**< Silent */
        DLOG_PRIO_MAX /**< Keep this always at the end. */
} log_priority;

/**
 * @brief Hijack dlog Tizen infra for unit testing to force printing out.
 * @bug The original dlog_print returns the number of bytes printed.
 *      This returns 0.
 */
extern int dlog_print (log_priority prio, const char *tag, const char *fmt, ...);
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _NNS_UNITTEST_UTIL_H__ */
