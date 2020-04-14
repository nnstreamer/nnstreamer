/**
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file	nnstreamer_profile.h
 * @date	14 April 2020
 * @brief	Internal util for profile log.
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nnstreamer_log.h"
#include "nnstreamer_profile.h"

/**
 * @brief Hashtable to handle the profile tasks.
 */
static GHashTable *g_nns_profile_task = NULL;

/**
 * @brief Struct for the profile task.
 */
typedef struct
{
  gint64 start;
  gint64 total;
  gsize count;
} nns_profile_task_s;

/**
 * @brief Internal function to free profile task.
 */
static void
nns_profile_destroy_task (gpointer data)
{
  nns_profile_task_s *task;

  task = (nns_profile_task_s *) data;

  /* free all resources in task */
  g_free (task);
}

/**
 * @brief Function to set start time.
 */
void
nns_profile_start (const gchar * name)
{
  nns_profile_task_s *task;

  if (g_nns_profile_task == NULL) {
    g_nns_profile_task =
        g_hash_table_new_full (g_str_hash, g_str_equal, g_free,
        nns_profile_destroy_task);
  }

  task = g_hash_table_lookup (g_nns_profile_task, name);
  if (task == NULL) {
    task = g_new0 (nns_profile_task_s, 1);
    g_assert (task);
    g_assert (g_hash_table_insert (g_nns_profile_task, g_strdup (name), task));
  }

  task->start = g_get_monotonic_time ();
}

/**
 * @brief Function to set end time.
 */
void
nns_profile_end (const gchar * name)
{
  nns_profile_task_s *task;
  gint64 time_diff, average;

  task = g_hash_table_lookup (g_nns_profile_task, name);
  g_assert (task);

  time_diff = g_get_monotonic_time () - task->start;
  task->total += time_diff;
  task->count++;
  average = (gint64) (task->total / task->count);

  /* print log */
  nns_logw ("[PROFILE] %s:current %" G_GINT64_FORMAT, name, time_diff);
  nns_logw ("[PROFILE] %s:average %" G_GINT64_FORMAT, name, average);
}
