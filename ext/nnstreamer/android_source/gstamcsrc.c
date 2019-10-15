/**
 * GStreamer Android MediaCodec (AMC) Source
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2019 Dongju Chae <dongju.chae@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	  gstamcsrc.c
 * @date	  19 May 2019
 * @brief	  GStreamer source element for Android MediaCodec (AMC)
 * @see		  http://github.com/nnsuite/nnstreamer
 * @author	Dongju Chae <dongju.chae@samsung.com>
 * @bug		  No known bugs except for NYI items
 */

/**
 * SECTION:element-amcsrc
 *
 * #amcsrc extends #gstpushsrc source element to reuse the preprocessing capability of
 * Android's standard MMFW (i.e., StageFright). It feeds the decoded frames from
 * Android MediaCodec (AMC) into a gstreamer pipeline.
 *
 * Note that it's recommended to use this element within Android JNI applications.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch amcsrc location=test.mp4 | autovideosink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "gstamcsrc.h"
#include "gstamcsrc_looper.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#include <string.h>
#include <endian.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <assert.h>
/** Glib related */
#include <glib.h>
#include <glib/gstdio.h>
#include <glib/gprintf.h>
/** Gstreamer related */
#include <gst/base/base.h>
#include <gst/video/video.h>
/** Android/JNI related */
#include <jni.h>
#include <android/log.h>
#include <android/native_window_jni.h>
#include <media/NdkMediaError.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>

#define TAG "AMCSRC"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__))

#ifndef S_ISREG
#define S_ISREG(mode) ((mode)&_S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(mode) ((mode)&_S_IFDIR)
#endif
#ifndef S_ISSOCK
#define S_ISSOCK(x) (0)
#endif
#ifndef O_BINARY
#define O_BINARY (0)
#endif

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

/**
 * @brief Private members in GstAMCSrc
 */
struct _GstAMCSrcPrivate
{
  GMutex mutex;
  GstDataQueue *outbound_queue;

  /** media file info */
  gchar *filename;
  gint fd;
  guint64 pos;
  guint64 size;

  /** media format */
  gint32 width;
  gint32 height;
  gint32 framerate;
  gint64 duration;

  gboolean seekable;
  gboolean is_regular;
  gboolean started;
  GstClockTime previous_ts;

  gint64 renderstart;
  gboolean renderonce;
  gboolean isPlaying;
  gboolean sawInputEOS;
  gboolean sawOutputEOS;

  /** Android MediaCodec */
  AMediaExtractor *ex;
  AMediaCodec *codec;
  void *looper;
};

#define GST_AMC_SRC_GET_PRIVATE(obj)  \
    (G_TYPE_INSTANCE_GET_PRIVATE ((obj), GST_TYPE_AMC_SRC, GstAMCSrcPrivate))

#define gst_amc_src_parent_class parent_class

G_DEFINE_TYPE_WITH_CODE (GstAMCSrc, gst_amc_src, GST_TYPE_PUSH_SRC,
    G_ADD_PRIVATE (GstAMCSrc)
    GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "amcsrc", 0, "Android MediaCodec (AMC) Source"))

/** GObject method implementation */
static void gst_amc_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_amc_src_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_amc_src_finalize (GObject * object);

/** GstBaseSrc method implementation */
static gboolean gst_amc_src_start (GstBaseSrc * src);
static gboolean gst_amc_src_stop (GstBaseSrc * src);
static GstCaps *gst_amc_src_get_caps (GstBaseSrc * src, GstCaps * filter);
static gboolean gst_amc_src_is_seekable (GstBaseSrc * src);
static gboolean gst_amc_src_unlock (GstBaseSrc * src);
static gboolean gst_amc_src_unlock_stop (GstBaseSrc * src);

/** GstPushSrc method implementation */
static GstFlowReturn gst_amc_src_create (GstPushSrc * src, GstBuffer ** buf);

/** GstElement method implementation */
static GstStateChangeReturn gst_amc_src_change_state (GstElement * element,
    GstStateChange transition);

/**
 * @brief enum for propery
 */
typedef enum
{
  PROP_0,
  PROP_LOCATION
} GstAMCSrcProperty;

/**
 * @brief enum for codec message
 */
typedef enum
{
  MSG_0,
  MSG_CODEC_BUFFER,
  MSG_CODEC_DONE,
  MSG_CODEC_SEEK,
  MSG_CODEC_PAUSE,
  MSG_CODEC_PAUSE_ACK,
  MSG_CODEC_RESUME
} GstAMCSrcMsg;

/**
 * @brief structure for a wrapped buffer
 */
typedef struct
{
  gsize refcount;
  GstAMCSrc *amcsrc;
  guint8 *buf;
  gint idx;
} GstWrappedBuf;

/**
 * @brief callback for increasing the refcount in a wrapped buffer
 * @param[in] a wrapped buffer
 */
static GstWrappedBuf *
gst_wrapped_buf_ref (GstWrappedBuf * self)
{
  g_return_val_if_fail (self != NULL, NULL);
  g_return_val_if_fail (self->amcsrc != NULL, NULL);
  g_return_val_if_fail (self->buf != NULL, NULL);
  g_return_val_if_fail (self->idx >= 0, NULL);
  g_return_val_if_fail (self->refcount >= 1, NULL);

  self->refcount++;
  return self;
}

/**
 * @brief callback for decreasing the refcount in a wrapped buffer
 * @param[in] a wrapped buffer
 */
static void
gst_wrapped_buf_unref (GstWrappedBuf * self)
{
  g_return_if_fail (self != NULL);
  g_return_if_fail (self->amcsrc != NULL);
  g_return_if_fail (self->buf != NULL);
  g_return_if_fail (self->idx >= 0);
  g_return_if_fail (self->refcount >= 1);

  if (--self->refcount == 0) {
    /** it's now released */
    GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self->amcsrc);

    AMediaCodec_releaseOutputBuffer(priv->codec, self->idx, 0);
    gst_object_unref (self->amcsrc);
  }
}

/**
 * @brief define boxed type for a wrapped buf
 */
G_DEFINE_BOXED_TYPE (GstWrappedBuf, gst_wrapped_buf,
    gst_wrapped_buf_ref, gst_wrapped_buf_unref)

/**
 * @brief get system's nanotime
 */
int64_t systemnanotime() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec * 1000000000LL + now.tv_nsec;
}

/**
 * @brief initialize the amc_src class.
 */
static void
gst_amc_src_class_init (GstAMCSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);
  GstPushSrcClass *gstpushsrc_class = GST_PUSH_SRC_CLASS (klass);

  /** property-related init */
  gobject_class->set_property = gst_amc_src_set_property;
  gobject_class->get_property = gst_amc_src_get_property;

  g_object_class_install_property (gobject_class, PROP_LOCATION,
      g_param_spec_string ("location", "File Location",
        "Location of the media file to play", NULL,
        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
        GST_PARAM_MUTABLE_READY));

  /** GstBaseSrcClass members */
  gstbasesrc_class->start = GST_DEBUG_FUNCPTR (gst_amc_src_start);
  gstbasesrc_class->stop = GST_DEBUG_FUNCPTR (gst_amc_src_stop);
  gstbasesrc_class->get_caps = GST_DEBUG_FUNCPTR (gst_amc_src_get_caps);
  gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR (gst_amc_src_is_seekable);
  gstbasesrc_class->unlock = GST_DEBUG_FUNCPTR (gst_amc_src_unlock);
  gstbasesrc_class->unlock_stop = GST_DEBUG_FUNCPTR (gst_amc_src_unlock_stop);

  /** GstPushSrcClass */
  gstpushsrc_class->create = GST_DEBUG_FUNCPTR (gst_amc_src_create);

  /** ElementClass */
  gst_element_class_add_static_pad_template (gstelement_class, &src_factory);
  gst_element_class_set_static_metadata (gstelement_class,
      "amcsrc", "Source/AMC",
      "Src element to feed the decoded data from Android MediaCodec (AMC)",
      "Dongju Chae <dongju.chae@samsung.com>");

  gstelement_class->change_state = GST_DEBUG_FUNCPTR (gst_amc_src_change_state);
}

/**
 * @brief config media codec settings; set the media source and obtain its format
 */
static gboolean
gst_amc_src_codec_config (GstAMCSrc *self)
{
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);
  media_status_t err;
  gint i, num_tracks;

  priv->ex = AMediaExtractor_new();
  err = AMediaExtractor_setDataSourceFd(priv->ex, priv->fd, priv->pos, priv->size);

  /** it's safe to close fd after setDataSourceFd() */
  close (priv->fd);

  if (err != AMEDIA_OK) {
    LOGE ("Error setting data source.");
    return FALSE;
  }

  /** find a video track to feed */
  num_tracks = AMediaExtractor_getTrackCount(priv->ex);

  for (i = 0; i < num_tracks; i++) {
    AMediaFormat *format = AMediaExtractor_getTrackFormat(priv->ex, i);
    const char *mime;
    if (AMediaFormat_getString(format, AMEDIAFORMAT_KEY_MIME, &mime)) {
      if (g_ascii_strncasecmp (mime, "video/", 6) == 0) {
        /** Use this video track */
        LOGI ("Video track found: %s", AMediaFormat_toString(format));

        priv->codec = AMediaCodec_createDecoderByType(mime);
        priv->renderstart = -1;
        priv->renderonce = TRUE;
        priv->sawInputEOS = FALSE;
        priv->sawOutputEOS = FALSE;
        priv->isPlaying = FALSE;

        AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_WIDTH, &priv->width);
        AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_HEIGHT, &priv->height);
        AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_FRAME_RATE, &priv->framerate);
        AMediaFormat_getInt64(format, AMEDIAFORMAT_KEY_DURATION, &priv->duration);

        AMediaCodec_configure(priv->codec, format, NULL /** surface */, NULL /** crypto */, 0);
        AMediaExtractor_selectTrack(priv->ex, i);
        AMediaFormat_delete(format);

        return TRUE;
      }
    } else
      LOGE ("No mime type");

    AMediaFormat_delete(format);
  }

  return FALSE;
}

/**
 * @brief open the target file and check its stats
 */
static gboolean gst_amc_src_media_open (GstAMCSrc *self)
{
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);
  struct stat stat_results;

  if (priv->filename == NULL || priv->filename[0] == '\0')
    goto error;

  LOGI ("opening file %s", priv->filename);

  priv->fd = open (priv->filename, O_RDONLY | O_BINARY, 0);
  if (priv->fd < 0) {
    LOGE ("Error: gst_open() failed");
    goto error;
  }

  if (fstat (priv->fd, &stat_results) < 0) {
    LOGE ("Error: fstat() failed");
    goto error_close;
  }

  if (S_ISDIR (stat_results.st_mode) || S_ISSOCK (stat_results.st_mode)) {
    LOGE ("Error: invalid file type");
    goto error_close;
  }

  priv->pos = 0;
  priv->size = stat_results.st_size;

  if (S_ISREG (stat_results.st_mode))
    priv->is_regular = TRUE;

  {
    off_t res = lseek (priv->fd, 0, SEEK_END);

    if (res < 0) {
      LOGW ("disabling seeking, lseek failed: %s", g_strerror (errno));
      priv->seekable = FALSE;
    } else {
      res = lseek (priv->fd, 0, SEEK_SET);
      if (res < 0) {
        priv->seekable = FALSE;
        LOGE ("Error: lseek() failed");
        goto error_close;
      }
      priv->seekable = TRUE;
    }
  }

  priv->seekable = priv->seekable && priv->is_regular;

  return gst_amc_src_codec_config (self);

error_close:
  close (priv->fd);
error:
  return FALSE;
}

/**
 * @brief set file location for media codec
 * @param[in] location Name of the media file
 */
static gboolean
gst_amc_src_set_location (GstAMCSrc * self, const gchar * location,
    GError ** err)
{
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  if (location) {
    /* check old file name & clear media info */
    if (priv->filename) {
      /* delete media info */
      g_free (priv->filename);
      AMediaCodec_delete (priv->codec);
      AMediaExtractor_delete (priv->ex);
    }

    priv->filename = g_strdup (location);
    return gst_amc_src_media_open (self);
  }

  LOGE ("A file location should be provided");
  return FALSE;
}

/**
 * @brief set amcsrc properties
 */
static void
gst_amc_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstAMCSrc *self = GST_AMC_SRC (object);

  switch (prop_id) {
    case PROP_LOCATION:
      gst_amc_src_set_location (self, g_value_get_string (value), NULL);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get amcsrc properties
 */
static void
gst_amc_src_get_property (GObject * object, guint prop_id,
    GValue *value, GParamSpec * pspec)
{
  GstAMCSrc *self = GST_AMC_SRC (object);
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  switch (prop_id) {
    case PROP_LOCATION:
      g_value_set_string (value, priv->filename);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief unlock function, flush any pending data in the data queue
 */
static gboolean
gst_amc_src_unlock (GstBaseSrc * src)
{
  GstAMCSrc *self = GST_AMC_SRC (src);
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  LOGI ("unlocking create");
  gst_data_queue_set_flushing (priv->outbound_queue, TRUE);

  return TRUE;
}

/**
 * @brief unlock_stop function, clear the previous unlock request
 */
static gboolean
gst_amc_src_unlock_stop (GstBaseSrc * src)
{
  GstAMCSrc *self = GST_AMC_SRC (src);
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  LOGI ("stopping unlock");
  gst_data_queue_set_flushing (priv->outbound_queue, FALSE);

  return TRUE;
}

/**
 * @brief get caps of subclass
 */
static GstCaps *
gst_amc_src_get_caps (GstBaseSrc * src, GstCaps * filter)
{
  GstAMCSrc *self = GST_AMC_SRC (src);
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);
  GstStructure *format;
  GstPad *pad = src->srcpad;
  GstCaps *current_caps = gst_pad_get_current_caps (pad);
  GstCaps *caps = gst_caps_new_empty ();

  GST_OBJECT_LOCK (self);

  /** width, height, and framerate were obtained from the media format */
  format = gst_structure_new ("video/x-raw",
      "format", G_TYPE_STRING, "NV12",  /** TODO Support other formats? */
      "width", G_TYPE_INT, priv->width,
      "height", G_TYPE_INT, priv->height,
      "interlaced", G_TYPE_BOOLEAN, FALSE,
      "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1,
      "framerate", GST_TYPE_FRACTION, priv->framerate, 1, NULL);

  gst_caps_append_structure (caps, format);

  if (current_caps) {
    GstCaps *intersection =
      gst_caps_intersect_full (current_caps, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (current_caps);
    gst_caps_unref (caps);
    caps = intersection;
  }

  GST_OBJECT_UNLOCK (self);

  return caps;
}

/**
 * @brief set a buffer which is a head item in data queue
 */
static GstFlowReturn
gst_amc_src_create (GstPushSrc * src, GstBuffer ** buffer)
{
  GstAMCSrc *self = GST_AMC_SRC_CAST (src);
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);
  GstDataQueueItem *item;

  if (!gst_data_queue_pop (priv->outbound_queue, &item)) {
    GST_DEBUG_OBJECT (self, "We're flushing");
    return GST_FLOW_FLUSHING;
  }

  *buffer = GST_BUFFER (item->object);
  g_free (item);

  return GST_FLOW_OK;
}

/**
 * @brief change state function for this element
 * each trainsition sends the corresponding message to a looper
 */
static GstStateChangeReturn gst_amc_src_change_state (GstElement * element,
    GstStateChange transition)
{
  GstAMCSrc *self;
  GstAMCSrcPrivate *priv;
  GstStateChangeReturn ret;
  gchar *dirname = NULL;

  self = GST_AMC_SRC (element);
  priv  = GST_AMC_SRC_GET_PRIVATE (self);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      Looper_post (priv->looper, MSG_CODEC_RESUME, self, false);
      LOGI ("PAUSED => PLAYING");
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      Looper_post (priv->looper, MSG_CODEC_PAUSE, self, false);
      LOGI ("PLAYING => PAUSED");
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief destroy callback for a data queue item
 */
static void
data_queue_item_free (GstDataQueueItem * item)
{
  g_clear_pointer (&item->object, (GDestroyNotify) gst_mini_object_unref);
  g_free (item);
}

/**
 * @brief feed a decoded data from media codec to pipeline
 * @note it avoid memcpy() by wrapping memory
 * @param[in] buf codec's output buffer
 * @param[in] idx codec's output buffer idx
 * @param[in] real_size codec's output data size
 * @param[in] buf_size codec's output buffer size
 */
static void
feed_frame_buf (GstAMCSrc *self, guint8 *buf, gint idx, gsize real_size, gsize buf_size)
{
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  GstBuffer *buffer;
  GstMemory *mem;
  GstDataQueueItem *item;
  GstClockTime duration = GST_CLOCK_TIME_NONE;
  GstClockTime current_ts = GST_CLOCK_TIME_NONE;
  GstClock *clock;

  GstWrappedBuf *wrapped_buf;

  if ((clock = gst_element_get_clock (self))) {
    GstClockTime base_time = GST_ELEMENT_CAST (self)->base_time;

    current_ts = gst_clock_get_time (clock) - base_time;
    gst_object_unref (clock);
  }

  g_mutex_lock (&priv->mutex);

  if (!priv->started || GST_CLOCK_TIME_IS_VALID (priv->previous_ts)) {
    duration = current_ts - priv->previous_ts;
    priv->previous_ts = current_ts;
  } else {
    priv->previous_ts = current_ts;

    /** Dropping first image to calculate duration */
    AMediaCodec_releaseOutputBuffer(priv->codec, idx, 0);
    LOGI ("Drop buf %d (reason: %s)", idx, priv->started ? "first frame" : "not yet started");
    g_mutex_unlock (&priv->mutex);

    return;
  }

  buffer = gst_buffer_new ();
  GST_BUFFER_DURATION (buffer) = duration;
  GST_BUFFER_PTS (buffer) = current_ts;

  wrapped_buf = g_new0 (GstWrappedBuf, 1);
  wrapped_buf->refcount = 1;
  wrapped_buf->amcsrc = g_object_ref (self);
  wrapped_buf->buf = buf;
  wrapped_buf->idx = idx;

  /** Allocate wrapped memory using codec's output buffer */
  mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
      buf, buf_size, 0, real_size,
      gst_wrapped_buf_ref (wrapped_buf),
      (GDestroyNotify) gst_wrapped_buf_unref);

  gst_buffer_append_memory (buffer, mem);

  item = g_new0 (GstDataQueueItem, 1);
  item->object = GST_MINI_OBJECT (buffer);
  item->size = gst_buffer_get_size (buffer);
  item->visible = TRUE;
  item->destroy = (GDestroyNotify) data_queue_item_free;

  if (!gst_data_queue_push (priv->outbound_queue, item)) {
    item->destroy (item);
    LOGW ("Failed to push item because we're flushing");
  }

  gst_wrapped_buf_unref (wrapped_buf);

  g_mutex_unlock (&priv->mutex);
}

/**
 * @brief check codec's input/output buffers to monitor/control its progress
 */
static void
check_codec_buf (GstAMCSrc *self)
{
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);
  gint64 delay, presentation_time;
  gssize buf_idx = -1;
  gsize buf_size;

  if (!priv->sawInputEOS) {
    buf_idx = AMediaCodec_dequeueInputBuffer(priv->codec, 2000);
    if (buf_idx > 0) {
      guint8 *buf = AMediaCodec_getInputBuffer(priv->codec, buf_idx, &buf_size);
      gint sample_size = AMediaExtractor_readSampleData(priv->ex, buf, buf_size);

      if (sample_size < 0) {
        LOGI("input EOS");
        sample_size = 0;
        priv->sawInputEOS = TRUE;
      }

      presentation_time = AMediaExtractor_getSampleTime(priv->ex);

      AMediaCodec_queueInputBuffer(priv->codec, buf_idx, 0, sample_size, presentation_time,
          priv->sawInputEOS ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0);
      AMediaExtractor_advance(priv->ex);
    }
  }

  if (!priv->sawOutputEOS) {
    AMediaCodecBufferInfo info;
    buf_idx = AMediaCodec_dequeueOutputBuffer(priv->codec, &info, 0);
    if (buf_idx >= 0) {
      if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
        LOGI("output EOS");
        priv->sawOutputEOS = TRUE;
      }

      presentation_time = info.presentationTimeUs * 1000;
      if (priv->renderstart < 0)
        priv->renderstart = systemnanotime() - presentation_time;

      delay = (priv->renderstart + presentation_time) - systemnanotime();
      if (delay > 0)
        usleep(delay / 1000);

      if (info.size > 0) {
        /**
         * we do not release this output buffer here.
         * it will be returned to the media codec when its wrapped buf is deallocated.
         */
        feed_frame_buf (self, AMediaCodec_getOutputBuffer (priv->codec, buf_idx, &buf_size),
                        buf_idx, info.size, buf_size);
      } else
        AMediaCodec_releaseOutputBuffer(priv->codec, buf_idx, 0);

      if (priv->renderonce) {
        priv->renderonce = FALSE;
        return;
      }
    } else if (buf_idx == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
      LOGI("Output buffers changed");
    } else if (buf_idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
      AMediaFormat* format = AMediaCodec_getOutputFormat(priv->codec);
      LOGI("format changed to: %s", AMediaFormat_toString(format));
      AMediaFormat_delete(format);
    } else if (buf_idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
      /** No output buffer right now */
    } else {
      LOGE("Unexpected info code: %zd", buf_idx);
    }
  }

  /** unless it reaches to EOS, check buffers again */
  if (!priv->sawInputEOS || !priv->sawOutputEOS)
    Looper_post (priv->looper, MSG_CODEC_BUFFER, self, FALSE);
}

/**
 * @brief looper internal function to handle each command
 */
static void looper_handle (gint cmd, void *data)
{
  if (data != NULL) {
    GstAMCSrc *self = GST_AMC_SRC (data);
    GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

    switch (cmd) {
      case MSG_CODEC_BUFFER:
        /** main codec function */
        check_codec_buf (self);
        break;
      case MSG_CODEC_DONE:
        /** stop media codec */
        AMediaCodec_stop (priv->codec);

        priv->sawInputEOS = TRUE;
        priv->sawOutputEOS = TRUE;

        LOGI ("Finished");
        break;
      case MSG_CODEC_SEEK:
        AMediaExtractor_seekTo (priv->ex, 0, AMEDIAEXTRACTOR_SEEK_NEXT_SYNC);
        AMediaCodec_flush (priv->ex);

        priv->renderstart = -1;
        priv->sawInputEOS = FALSE;
        priv->sawOutputEOS = FALSE;

        if (!priv->isPlaying) {
          priv->renderonce = TRUE;
          Looper_post (priv->looper, MSG_CODEC_BUFFER, self, FALSE);
        }

        LOGI ("Seeked");
        break;
      case MSG_CODEC_PAUSE:
        if (priv->isPlaying) {
          priv->isPlaying = FALSE;

          Looper_post (priv->looper, MSG_CODEC_PAUSE_ACK, self, TRUE);

          LOGI ("Paused");
        }
        break;
      case MSG_CODEC_PAUSE_ACK:
        /** Only used to flush buffers */
        break;
      case MSG_CODEC_RESUME:
        if (!priv->isPlaying) {
          priv->renderstart = -1;
          priv->isPlaying = TRUE;

          Looper_post (priv->looper, MSG_CODEC_BUFFER, self, FALSE);

          LOGI ("Resumed");
        }
        break;
      default:
        LOGE ("Unexpected message: cmd(%d)", cmd);
        break;
    }
  }
}

/**
 * @brief start function, called when state changed null to ready
 * start the media codec and its looper
 */
static gboolean
gst_amc_src_start (GstBaseSrc * src)
{
  GstAMCSrc *self = GST_AMC_SRC (src);
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  /** setup a looper */
  priv->looper = Looper_new();
  Looper_set_handle (priv->looper, looper_handle);

  priv->previous_ts = GST_CLOCK_TIME_NONE;
  priv->started = TRUE;

  /** start media codec */
  AMediaCodec_start(priv->codec);
  Looper_post (priv->looper, MSG_CODEC_BUFFER, self, false);

  return TRUE;
}

/**
 * @brief stop function, called when state changed ready to null
 */
static gboolean gst_amc_src_stop (GstBaseSrc * src)
{
  GstAMCSrc *self = GST_AMC_SRC (src);
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  g_mutex_lock (&priv->mutex);

  gst_data_queue_flush (priv->outbound_queue);

  /** stop a looper */
  Looper_post (priv->looper, MSG_CODEC_DONE, self, TRUE);
  Looper_exit (priv->looper);

  priv->previous_ts = GST_CLOCK_TIME_NONE;
  priv->started = FALSE;

  g_mutex_unlock (&priv->mutex);

  return TRUE;
}

/**
 * @brief check if source supports seeking
 */
gboolean gst_amc_src_is_seekable (GstBaseSrc * src)
{
  GstAMCSrc *self = GST_AMC_SRC_CAST (src);
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  return priv->seekable;
}

/**
 * @brief callback for checking data_queue full
 */
static gboolean
data_queue_check_full_cb (GstDataQueue * queue, guint visible,
    guint bytes, guint64 time, gpointer checkdata)
{
  /** it's dummy */
  return FALSE;
}

/**
 * @brief initialize amcsrc element.
 */
static void
gst_amc_src_init (GstAMCSrc * self)
{
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  priv->outbound_queue = gst_data_queue_new (data_queue_check_full_cb, NULL, NULL, NULL);
  priv->filename = NULL;

  priv->width = 0;
  priv->height = 0;
  priv->framerate = 0;
  priv->duration = 0;

  g_mutex_init (&priv->mutex);

  gst_base_src_set_format (GST_BASE_SRC (self), GST_FORMAT_TIME);
  gst_base_src_set_live (GST_BASE_SRC (self), TRUE);
  gst_base_src_set_async (GST_BASE_SRC (self), TRUE);
}

/**
 * @brief finalize the instance
 */
static void
gst_amc_src_finalize (GObject * object)
{
  GstAMCSrc *self = GST_AMC_SRC (object);
  GstAMCSrcPrivate *priv = GST_AMC_SRC_GET_PRIVATE (self);

  /* delete media info */
  g_free (priv->filename);
  AMediaCodec_delete (priv->codec);
  AMediaExtractor_delete (priv->ex);

  g_clear_pointer (&priv->outbound_queue, gst_object_unref);

  g_mutex_clear (&priv->mutex);
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief register this element
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "amcsrc", 0,
      "nnstreamer amcsrc source element");

  if (!gst_element_register (plugin, "amcsrc", GST_RANK_NONE, GST_TYPE_AMC_SRC)) {
    return FALSE;
  }

  return TRUE;
}

#ifndef PACKAGE
#define PACKAGE "amcsrc"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    amcsrc,
    "NNStreamer Android MediaCodec (AMC) Source",
    plugin_init, VERSION, "LGPL", "nnstreamer", "https://github.com/nnsuite/nnstreamer/")
