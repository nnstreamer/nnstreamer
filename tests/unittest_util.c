/**
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file        unittest_util.c
 * @date        15 Jan 2019
 * @brief       Unit test utility.
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <gst/gst.h>
#include <glib/gstdio.h>
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
  gint counter = 0;
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
  } while ((timeout_ms / 20) < counter++);
  return -ETIME;
}

/**
 * @brief Get temp file name.
 * @return file name (should free string with g_free)
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
  if (g_remove (tmp_fn) != 0) {
    _print_log ("failed to remove temp file %s", tmp_fn);
  }

  return tmp_fn;
}
