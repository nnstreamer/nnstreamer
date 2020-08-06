/**
 * GStreamer Android MediaCodec (AMC) Source Looper
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2019 Dongju Chae <dongju.chae@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	  gstamcsrc_looper.h
 * @date	  19 May 2019
 * @brief   A looper thread to perform event messages between amcsrc and media codec
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __GST_AMC_SRC_LOOPER_H__
#define __GST_AMC_SRC_LOOPER_H__

#include <pthread.h>
#include <glib.h>

struct looper_message;
typedef struct looper_message looper_message;

/** @brief structure for looper_message */
struct looper_message {
  gint cmd;             /**< cmd type */
  void *data;           /**< argument */
  looper_message *next; /**< linked list */
};

#ifdef __cplusplus

/**
 * @brief Looper class to handle codec messages
 */
class Looper {
  public:
    Looper ();

    void loop (void);
    void start (void);
    void exit (void);
    void post (gint cmd, void *data, bool flush);
    void add_msg (looper_message *new_msg, bool flush);
    void (*handle) (gint cmd, void *data);  /**< should be implemented */

  private:
    static void *entry (void *data);

    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    looper_message *head;

    gboolean running;
    guint num_msg;
};

/**
 * @brief C wrapper to access Looper C++ class
 */
extern "C"
{
#endif
  void *Looper_new (void);
  void Looper_exit (void *looper);
  void Looper_post (void *looper, gint cmd, void *data, gboolean flush);
  void Looper_set_handle (void *looper, void (*handle) (gint, void*));
#ifdef __cplusplus
}
#endif

#endif /** __GST_AMC_SRC_LOOPER_H__ */
