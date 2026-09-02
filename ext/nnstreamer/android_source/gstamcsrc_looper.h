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
    /** @brief Flush pending messages and destroy the sync primitives */
    ~Looper ();

    /** @brief Dispatch queued messages to the handler until exit is posted */
    void loop (void);
    /** @brief Create the detached thread running the message loop */
    void start (void);
    /** @brief Terminate the loop by posting a flushing exit message */
    void exit (void);
    /** @brief Queue a command for loop () to hand to the handler */
    void post (gint cmd, void *data, bool flush);
    void (*handle) (gint cmd, void *data);  /**< should be implemented */

  private:
    /** @brief Thread entry point; runs loop () on the given Looper instance */
    static void *entry (void *data);
    /** @brief Append a message to the queue, flushing pending ones if asked */
    void add_msg (looper_message *new_msg, bool flush);
    /** @brief Drop and free every pending message in the queue */
    void flush_msg (void);

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
  /** @brief Create a Looper, start its thread, and return the opaque handle */
  void *Looper_new (void);
  /** @brief Destroy a Looper created by Looper_new () */
  void Looper_delete (void *looper);
  /** @brief Ask the looper thread to terminate */
  void Looper_exit (void *looper);
  /** @brief Queue a command for the looper thread to handle */
  void Looper_post (void *looper, gint cmd, void *data, gboolean flush);
  /** @brief Register the callback invoked for each queued command */
  void Looper_set_handle (void *looper, void (*handle) (gint, void*));
#ifdef __cplusplus
}
#endif

#endif /** __GST_AMC_SRC_LOOPER_H__ */
