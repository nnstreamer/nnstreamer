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
 * @file	  gstamcsrc_looper.cc
 * @date	  19 May 2019
 * @brief	  A looper thread to perform event messages between amcsrc and media codec
 * @see		  http://github.com/nnstreamer/nnstreamer
 * @author	Dongju Chae <dongju.chae@samsung.com>
 * @bug		  No known bugs except for NYI items
 */

#include "gstamcsrc_looper.h"
#include "gstamcsrc.h"

#include <errno.h>
#include <fcntl.h>
#include <jni.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <android/log.h>
#define TAG "AMCSRC-looper"
#define LOG(...) __android_log_print (ANDROID_LOG_INFO, TAG, __VA_ARGS__)

/**
 * @brief Looper constructor
 */
Looper::Looper ()
{
  head = NULL;
  running = FALSE;
  num_msg = 0;
  handle = NULL;

  pthread_mutex_init (&mutex, NULL);
  pthread_cond_init (&cond, NULL);
}

/**
 * @brief Looper desctructor
 */
Looper::~Looper ()
{
  flush_msg ();
  pthread_mutex_destroy (&mutex);
  pthread_cond_destroy (&cond);
}

/**
 * @brief Creates a looper thread
 */
void
Looper::start (void)
{
  pthread_attr_t attr;

  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

  pthread_create (&thread, &attr, entry, this);

  pthread_attr_destroy (&attr);
}

/**
 * @brief allocate a message with arguments
 * @param[in] cmd command type
 * @param[in] data argument
 * @param[in] flush whether flushing the pending messages
 */
void
Looper::post (gint cmd, void *data, bool flush)
{
  looper_message *msg = new looper_message ();

  msg->cmd = cmd;
  msg->data = data;
  msg->next = NULL;

  add_msg (msg, flush);
}

/**
 * @brief append a message
 * @param[in] new_msg new message to append
 * @param[in] flush whether flushing the pending messages
 */
void
Looper::add_msg (looper_message *new_msg, bool flush)
{
  looper_message *msg;

  pthread_mutex_lock (&mutex);

  /** Flush old pending messages */
  if (flush) {
    flush_msg ();
  }

  /** Append new message */
  if (head) {
    msg = head;
    while (msg->next)
      msg = msg->next;
    msg->next = new_msg;
  } else
    head = new_msg;

  num_msg++;

  pthread_cond_broadcast (&cond);
  pthread_mutex_unlock (&mutex);
}

/**
 * @brief Flush all messages.
 */
void
Looper::flush_msg (void)
{
  looper_message *msg;

  msg = head;
  while (msg) {
    head = msg->next;
    delete msg;
    msg = head;
  }

  head = NULL;
  num_msg = 0;
}

/**
 * @brief looper's entry function
 */
void *
Looper::entry (void *data)
{
  if (data)
    ((Looper *) data)->loop ();
  return NULL;
}

/**
 * @brief looper's loop function
 */
void
Looper::loop (void)
{
  LOG ("AMC looper started!");
  running = TRUE;

  while (running) {
    looper_message *msg;

    /** Wait new message */
    pthread_mutex_lock (&mutex);
    while (!(num_msg > 0)) {
      pthread_cond_wait (&cond, &mutex);
    }

    msg = head;
    head = head->next;
    num_msg--;

    pthread_mutex_unlock (&mutex);

    if (handle)
      handle (msg->cmd, msg->data);
    delete msg;
  }

  LOG ("Looper terminated");
}

/**
 * @brief looper's exit function
 */
void
Looper::exit (void)
{
  running = FALSE;
  post (0, NULL, TRUE);
  /* 0 == MSG_0 */
}

/**
 * @brief C-wrapper for Looper constructor
 */
void *
Looper_new (void)
{
  Looper *looper = new Looper ();

  looper->start ();
  return looper;
}

/**
 * @brief C-wrapper for Looper destructor
 */
void
Looper_delete (void *looper)
{
  if (looper)
    delete ((Looper *) looper);
}

/**
 * @brief C-wrapper for Looper post function
 */
void
Looper_post (void *looper, gint cmd, void *data, gboolean flush)
{
  if (looper)
    ((Looper *) looper)->post (cmd, data, flush);
}

/**
 * @brief C-wrapper for setting handle function of looper
 */
void
Looper_set_handle (void *looper, void (*handle) (gint, void *))
{
  if (looper)
    ((Looper *) looper)->handle = handle;
}

/**
 * @brief C-wrapper for terminating the looper
 */
void
Looper_exit (void *looper)
{
  if (looper)
    ((Looper *) looper)->exit ();
}
