/**
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file        unittest_util.c
 * @date        15 Jan 2019
 * @brief       Unit test utility.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <gst/gst.h>
#include <glib/gstdio.h>
#include <string.h>
#include "unittest_util.h"

/**
 * @brief Set pipeline state, wait until it's done.
 * @return 0 success, -EPIPE if failed, -ETIME if timeout happens.
 */
int
setPipelineStateSync (GstElement * pipeline, GstState state,
    uint32_t timeout_ms)
{
  GstState cur_state = GST_STATE_VOID_PENDING;
  GstStateChangeReturn ret;
  guint counter = 0;
  ret = gst_element_set_state (pipeline, state);

  if (ret == GST_STATE_CHANGE_FAILURE)
    return -EPIPE;

  do {
    ret = gst_element_get_state (pipeline, &cur_state, NULL, 10 * GST_MSECOND);
    if (ret == GST_STATE_CHANGE_FAILURE)
      return -EPIPE;
    if (cur_state == state)
      return 0;
    g_usleep (10000);
  } while ((timeout_ms / 20) > counter++);
  return -ETIME;
}

/**
 * @brief Get temp file name.
 * @return file name (should finalize it with g_remove() and g_free() after use)
 */
gchar *
getTempFilename (void)
{
  const gchar *tmp_dir;
  gchar *tmp_fn;
  gint fd;

  if ((tmp_dir = g_get_tmp_dir ()) == NULL) {
    _print_log ("failed to get tmp dir");
    return NULL;
  }

  tmp_fn = g_build_filename (tmp_dir, "nnstreamer_unittest_temp_XXXXXX", NULL);
  fd = g_mkstemp (tmp_fn);

  if (fd < 0) {
    _print_log ("failed to create temp file %s", tmp_fn);
    g_free (tmp_fn);
    return NULL;
  }

  g_close (fd, NULL);

  return tmp_fn;
}

/**
 * @brief Wait until the pipeline processing the buffers
 * @return TRUE on success, FALSE when a time-out occurs
 */
gboolean
wait_pipeline_process_buffers (const guint * data_received,
    guint expected_num_buffers, guint timeout_ms)
{
  guint timer = 0;
  guint tick = TEST_DEFAULT_SLEEP_TIME / 1000U;

  /* Waiting for expected buffers to arrive */
  while (*data_received < expected_num_buffers) {
    g_usleep (TEST_DEFAULT_SLEEP_TIME);
    timer += tick;
    if (timer > timeout_ms)
      return FALSE;
  }
  return TRUE;
}

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
gchar *
replace_string (gchar * source, const gchar * what, const gchar * to,
    const gchar * delimiters, guint * count)
{
  GString *builder;
  gchar *start, *pos, *result;
  guint changed = 0;
  gsize len;

  g_return_val_if_fail (source, NULL);
  g_return_val_if_fail (what && to, source);

  len = strlen (what);
  start = source;

  builder = g_string_new (NULL);
  while ((pos = g_strstr_len (start, -1, what)) != NULL) {
    gboolean skip = FALSE;

    if (delimiters) {
      const gchar *s;
      gchar *prev, *next;
      gboolean prev_split, next_split;

      prev = next = NULL;
      prev_split = next_split = FALSE;

      if (pos != source)
        prev = pos - 1;
      if (*(pos + len) != '\0')
        next = pos + len;

      for (s = delimiters; *s != '\0'; ++s) {
        if (!prev || *s == *prev)
          prev_split = TRUE;
        if (!next || *s == *next)
          next_split = TRUE;
        if (prev_split && next_split)
          break;
      }

      if (!prev_split || !next_split)
        skip = TRUE;
    }

    builder = g_string_append_len (builder, start, pos - start);

    /* replace string if found */
    if (skip)
      builder = g_string_append_len (builder, pos, len);
    else
      builder = g_string_append (builder, to);

    start = pos + len;
    if (!skip)
      changed++;
  }

  /* append remains */
  builder = g_string_append (builder, start);
  result = g_string_free (builder, FALSE);

  if (count)
    *count = changed;

  g_free (source);
  return result;
}

#ifdef FAKEDLOG
/**
 * @brief Hijack dlog Tizen infra for unit testing to force printing out.
 * @bug The original dlog_print returns the number of bytes printed.
 *      This returns 0.
 */
int
dlog_print (log_priority prio, const char *tag, const char *fmt, ...)
{
  va_list arg_ptr;
  GLogLevelFlags level;
  switch (prio) {
    case DLOG_FATAL:
      level = G_LOG_LEVEL_ERROR;
      break;
    case DLOG_ERROR:
      level = G_LOG_LEVEL_CRITICAL;
      break;
    case DLOG_WARN:
      level = G_LOG_LEVEL_WARNING;
      break;
    case DLOG_INFO:
      level = G_LOG_LEVEL_INFO;
      break;
    case DLOG_DEBUG:
      level = G_LOG_LEVEL_DEBUG;
      break;
    default:
      level = G_LOG_LEVEL_DEBUG;
  }
  va_start (arg_ptr, fmt);
  g_logv (tag, level, fmt, arg_ptr);
  va_end (arg_ptr);

  return 0;
}
#endif
