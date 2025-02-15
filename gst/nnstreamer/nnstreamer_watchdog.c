/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer watchdog
 * Copyright (C) 2024 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	nnstreamer_watchdog.c
 * @date	30 Oct 2024
 * @brief	NNStreamer watchdog to manage the schedule in the element.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <nnstreamer_log.h>
#include "nnstreamer_watchdog.h"

/**
 * @brief Structure for NNStreamer watchdog.
 */
typedef struct _NnstWatchdog
{
  GMainContext *context;
  GMainLoop *loop;
  GThread *thread;
  GSource *source;
  GMutex lock;
  GCond cond;
} NnstWatchdog;

/**
 * @brief Called when loop is running
 */
static gboolean
_loop_running_cb (NnstWatchdog * watchdog)
{
  g_mutex_lock (&watchdog->lock);
  g_cond_signal (&watchdog->cond);
  g_mutex_unlock (&watchdog->lock);

  return G_SOURCE_REMOVE;
}

/**
 * @brief Watchdog thread.
 */
static gpointer
_nnstreamer_watchdog_thread (gpointer ptr)
{
  NnstWatchdog *watchdog = (NnstWatchdog *) ptr;
  GSource *idle_source;

  g_main_context_push_thread_default (watchdog->context);

  idle_source = g_idle_source_new ();
  g_source_set_callback (idle_source,
      (GSourceFunc) _loop_running_cb, watchdog, NULL);
  g_source_attach (idle_source, watchdog->context);
  g_source_unref (idle_source);

  g_main_loop_run (watchdog->loop);

  g_main_context_pop_thread_default (watchdog->context);

  return NULL;
}

/**
 * @brief Create nnstreamer watchdog. Recommended using watchdog handle with proper lock (e.g., GST_OBJECT_LOCK())
 */
gboolean NNS_API
nnstreamer_watchdog_create (nns_watchdog_h * watchdog_h)
{
  gint64 end_time;
  gboolean ret = TRUE;
  GError *error = NULL;
  NnstWatchdog *watchdog;

  if (!watchdog_h) {
    ml_loge ("Invalid parameter: watchdog handle is NULL.");
    return FALSE;
  }

  watchdog = g_try_new0 (NnstWatchdog, 1);
  if (!watchdog) {
    ml_loge ("Failed to allocate memory for watchdog.");
    return FALSE;
  }

  watchdog->context = g_main_context_new ();
  watchdog->loop = g_main_loop_new (watchdog->context, FALSE);

  g_mutex_init (&watchdog->lock);
  g_cond_init (&watchdog->cond);
  g_mutex_lock (&watchdog->lock);
  watchdog->thread =
      g_thread_try_new ("suspend_watchdog", _nnstreamer_watchdog_thread,
      watchdog, &error);

  if (!watchdog->thread) {
    ml_loge ("Failed to create watchdog thread: %s", error->message);
    g_clear_error (&error);
    ret = FALSE;
    goto done;
  }

  end_time = g_get_monotonic_time () + 5 * G_TIME_SPAN_SECOND;
  while (!g_main_loop_is_running (watchdog->loop)) {
    if (!g_cond_wait_until (&watchdog->cond, &watchdog->lock, end_time)) {
      ml_loge ("Failed to wait main loop running.");
      ret = FALSE;
      goto done;
    }
  }

done:
  g_mutex_unlock (&watchdog->lock);
  g_mutex_clear (&watchdog->lock);
  g_cond_clear (&watchdog->cond);
  if (!ret) {
    nnstreamer_watchdog_destroy (watchdog);
    watchdog = NULL;
  }
  *watchdog_h = watchdog;

  return ret;
}

/**
 * @brief Destroy watchdog source. Recommended using watchdog handle with proper lock (e.g., GST_OBJECT_LOCK())
 */
void NNS_API
nnstreamer_watchdog_destroy (nns_watchdog_h watchdog_h)
{
  NnstWatchdog *watchdog = (NnstWatchdog *) watchdog_h;
  nnstreamer_watchdog_release (watchdog);

  if (watchdog && watchdog->context) {
    g_main_loop_quit (watchdog->loop);
    g_thread_join (watchdog->thread);
    watchdog->thread = NULL;

    g_main_loop_unref (watchdog->loop);
    watchdog->loop = NULL;

    g_main_context_unref (watchdog->context);
    watchdog->context = NULL;

    g_free (watchdog_h);
  }
}

/**
 * @brief Release watchdog source. Recommended using watchdog handle with proper lock (e.g., GST_OBJECT_LOCK())
 */
void NNS_API
nnstreamer_watchdog_release (nns_watchdog_h watchdog_h)
{
  NnstWatchdog *watchdog = (NnstWatchdog *) watchdog_h;
  if (watchdog && watchdog->source) {
    g_source_destroy (watchdog->source);
    g_source_unref (watchdog->source);
    watchdog->source = NULL;
  }
}

/**
 * @brief Set watchdog timeout. Recommended using watchdog handle with proper lock (e.g., GST_OBJECT_LOCK())
 */
gboolean NNS_API
nnstreamer_watchdog_feed (nns_watchdog_h watchdog_h, GSourceFunc func,
    guint interval, void *user_data)
{
  NnstWatchdog *watchdog = (NnstWatchdog *) watchdog_h;

  if (!watchdog || !func) {
    ml_loge ("Invalid parameter: watchdog handle or func is NULL.");
    return FALSE;
  }

  if (watchdog->context) {
    watchdog->source = g_timeout_source_new (interval);
    g_source_set_callback (watchdog->source, func, user_data, NULL);
    g_source_attach (watchdog->source, watchdog->context);
  } else {
    ml_loge
        ("Failed to feed to watchdog. Watchdog is not created or context is invalid.");
    return FALSE;
  }

  return TRUE;
}
