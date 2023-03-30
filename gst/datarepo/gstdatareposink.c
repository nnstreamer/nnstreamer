/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd.
 *
 * @file	gstdatareposink.c
 * @date	30 March 2023
 * @brief	GStreamer plugin that writes data from buffers to files in in MLOps Data repository
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * ## Example launch line
 * |[
 * gst-launch-1.0 videotestsrc ! datareposink location=filename
 * ]|
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <gst/gst.h>
#include <glib/gstdio.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>
#include "gstdatareposink.h"

static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

/**
 * @brief datareposink properties.
 */
enum
{
  PROP_0,
  PROP_LOCATION,
};

GST_DEBUG_CATEGORY_STATIC (gst_data_repo_sink_debug);
#define GST_CAT_DEFAULT gst_data_repo_sink_debug
#define _do_init \
  GST_DEBUG_CATEGORY_INIT (gst_data_repo_sink_debug, "datareposink", 0, "datareposink element");
#define gst_data_repo_sink_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstDataRepoSink, gst_data_repo_sink,
    GST_TYPE_BASE_SINK, _do_init);

static void gst_data_repo_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_data_repo_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_data_repo_sink_finalize (GObject * object);
static gboolean gst_data_repo_sink_start (GstBaseSink * basesink);
static gboolean gst_data_repo_sink_stop (GstBaseSink * basesink);
static GstFlowReturn gst_data_repo_sink_render (GstBaseSink * bsink,
    GstBuffer * buffer);
static GstCaps *gst_data_repo_sink_get_caps (GstBaseSink * bsink,
    GstCaps * filter);
static gboolean gst_data_repo_sink_set_caps (GstBaseSink * bsink,
    GstCaps * caps);
static gboolean gst_data_repo_sink_event (GstBaseSink * bsink,
    GstEvent * event);
static gboolean gst_data_repo_sink_query (GstBaseSink * sink, GstQuery * query);

/**
 * @brief Initialize datareposink class.
 */
static void
gst_data_repo_sink_class_init (GstDataRepoSinkClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbasesink_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);
  gstbasesink_class = GST_BASE_SINK_CLASS (klass);

  gobject_class->set_property = gst_data_repo_sink_set_property;
  gobject_class->get_property = gst_data_repo_sink_get_property;
  gobject_class->finalize = gst_data_repo_sink_finalize;

  g_object_class_install_property (gobject_class, PROP_LOCATION,
      g_param_spec_string ("location", "File Location",
          "Location to write files to MLOps Data Repository. "
          "if the files are images, write the index of the filename name "
          "as %04d (e.g., filenmae%04d.png).",
          NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  gst_element_class_set_static_metadata (gstelement_class,
      "NNStreamer MLOps Data Repository Sink",
      "Sink/File",
      "Write files to MLOps Data Repository", "Samsung Electronics Co., Ltd.");

  gst_element_class_add_static_pad_template (gstelement_class, &sinktemplate);

  gstbasesink_class->render = GST_DEBUG_FUNCPTR (gst_data_repo_sink_render);
  gstbasesink_class->get_caps = GST_DEBUG_FUNCPTR (gst_data_repo_sink_get_caps);
  gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR (gst_data_repo_sink_set_caps);
  gstbasesink_class->event = GST_DEBUG_FUNCPTR (gst_data_repo_sink_event);
  gstbasesink_class->query = GST_DEBUG_FUNCPTR (gst_data_repo_sink_query);
  gstbasesink_class->start = GST_DEBUG_FUNCPTR (gst_data_repo_sink_start);
  gstbasesink_class->stop = GST_DEBUG_FUNCPTR (gst_data_repo_sink_stop);

  if (sizeof (off_t) < 8) {
    GST_LOG ("No large file support, sizeof (off_t) = %" G_GSIZE_FORMAT "!",
        sizeof (off_t));
  }
}

/**
 * @brief Initialize datareposink.
 */
static void
gst_data_repo_sink_init (GstDataRepoSink * sink)
{
  sink->filename = NULL;
  sink->fd = 0;
  sink->offset = 0;
}

/**
 * @brief finalize datareposink.
 */
static void
gst_data_repo_sink_finalize (GObject * object)
{
  GstDataRepoSink *sink = GST_DATA_REPO_SINK (object);

  g_free (sink->filename);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for datareposink properties.
 */
static void
gst_data_repo_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDataRepoSink *sink = GST_DATA_REPO_SINK (object);

  switch (prop_id) {
    case PROP_LOCATION:
      sink->filename = g_value_dup_string (value);
      GST_INFO_OBJECT (sink, "filename = %s", sink->filename);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter datareposink properties.
 */
static void
gst_data_repo_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDataRepoSink *sink;

  sink = GST_DATA_REPO_SINK (object);

  switch (prop_id) {
    case PROP_LOCATION:
      g_value_set_string (value, sink->filename);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Called when a buffer should be presented or ouput.
 */
static GstFlowReturn
gst_data_repo_sink_render (GstBaseSink * bsink, GstBuffer * buf)
{
  GstDataRepoSink *sink = GST_DATA_REPO_SINK_CAST (bsink);
  gsize write_size = 0;
  GstMapInfo info;
  guint to_write = 0, byte_write = 0;

  g_return_val_if_fail (sink->fd != 0, GST_FLOW_ERROR);

  GST_OBJECT_LOCK (sink);
  gst_buffer_map (buf, &info, GST_MAP_READ);
  to_write = info.size;

  GST_LOG_OBJECT (sink,
      "Writing %d bytes at offset 0x%" G_GINT64_MODIFIER "x (%d size)",
      to_write, sink->offset + byte_write, (guint) sink->offset + byte_write);

  write_size = write (sink->fd, info.data, info.size);
  if (write_size != info.size) {
    GST_ERROR_OBJECT (sink, "Could not write data to file");
    goto error;
  }

  gst_buffer_unmap (buf, &info);
  GST_OBJECT_UNLOCK (sink);

  sink->offset += write_size;

  return GST_FLOW_OK;

error:
  gst_buffer_unmap (buf, &info);
  GST_OBJECT_UNLOCK (sink);

  return GST_FLOW_ERROR;
}

/**
 * @brief Get caps of datareposink.
 */
static GstCaps *
gst_data_repo_sink_get_caps (GstBaseSink * bsink, GstCaps * filter)
{
  GstDataRepoSink *sink = GST_DATA_REPO_SINK (bsink);
  GstCaps *caps = NULL;

  GST_OBJECT_LOCK (sink);
  caps = gst_pad_get_pad_template_caps (GST_BASE_SINK_PAD (bsink));
  GST_OBJECT_UNLOCK (sink);

  GST_INFO_OBJECT (sink, "Got caps %" GST_PTR_FORMAT, caps);

  if (caps && filter) {
    GstCaps *intersection =
        gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (caps);
    caps = intersection;
  }

  GST_DEBUG_OBJECT (sink, "result get caps: %" GST_PTR_FORMAT, caps);

  return caps;
}

/**
 * @brief Set caps of datareposink.
 */
static gboolean
gst_data_repo_sink_set_caps (GstBaseSink * bsink, GstCaps * caps)
{
  GstDataRepoSink *sink;

  sink = GST_DATA_REPO_SINK (bsink);
  GST_INFO_OBJECT (sink, "set caps %" GST_PTR_FORMAT, caps);

  return TRUE;
}

/**
 * @brief Receive Event on datareposink.
 */
static gboolean
gst_data_repo_sink_event (GstBaseSink * bsink, GstEvent * event)
{
  GstDataRepoSink *sink;
  sink = GST_DATA_REPO_SINK (bsink);

  GST_INFO_OBJECT (sink, "got event (%s)",
      gst_event_type_get_name (GST_EVENT_TYPE (event)));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_EOS:
      GST_INFO_OBJECT (sink, "get GST_EVENT_EOS event..state is %d",
          GST_STATE (sink));
      break;
    case GST_EVENT_FLUSH_START:
      GST_INFO_OBJECT (sink, "get GST_EVENT_FLUSH_START event");
      break;
    case GST_EVENT_FLUSH_STOP:
      GST_INFO_OBJECT (sink, "get GST_EVENT_FLUSH_STOP event");
      break;
    default:
      break;
  }

  return GST_BASE_SINK_CLASS (parent_class)->event (bsink, event);
}

/**
 * @brief Perform a GstQuery on datareposink.
 */
static gboolean
gst_data_repo_sink_query (GstBaseSink * bsink, GstQuery * query)
{
  gboolean ret;

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_SEEKING:{
      GstFormat fmt;

      /* we don't supporting seeking */
      gst_query_parse_seeking (query, &fmt, NULL, NULL, NULL);
      gst_query_set_seeking (query, fmt, FALSE, 0, -1);
      ret = TRUE;
      break;
    }
    default:
      ret = GST_BASE_SINK_CLASS (parent_class)->query (bsink, query);
      break;
  }

  return ret;
}

/**
 * @brief Function to open file
 */
static gboolean
gst_data_repo_sink_open_file (GstDataRepoSink * sink)
{
  gchar *filename = NULL;
  int flags = O_CREAT | O_WRONLY;

  g_return_val_if_fail (sink != NULL, FALSE);

  if (sink->filename == NULL || sink->filename[0] == '\0')
    goto no_filename;

  /* need to get filename by media type */
  filename = g_strdup (sink->filename);

  GST_INFO_OBJECT (sink, "opening file %s", filename);

  /** How about support file mode property ?
     flags |= O_APPEND ("ab") */

  flags |= O_TRUNC;             /* "wb" */

  /* open the file */
  sink->fd = g_open (filename, flags, 0644);

  if (sink->fd < 0)
    goto open_failed;

  g_free (filename);

  return TRUE;

  /* ERRORS */
no_filename:
  {
    GST_ELEMENT_ERROR (sink, RESOURCE, NOT_FOUND,
        (("No file name specified for writing.")), (NULL));
    goto error_exit;
  }
open_failed:
  {
    switch (errno) {
      case ENOENT:
        GST_ELEMENT_ERROR (sink, RESOURCE, NOT_FOUND, (NULL),
            ("No such file \"%s\"", sink->filename));
        break;
      default:
        GST_ELEMENT_ERROR (sink, RESOURCE, OPEN_READ,
            (("Could not open file \"%s\" for reading."), sink->filename),
            GST_ERROR_SYSTEM);
        break;
    }
    goto error_exit;
  }

error_exit:
  g_free (filename);

  return FALSE;
}

/**
 * @brief Start datareposink
 */
static gboolean
gst_data_repo_sink_start (GstBaseSink * basesink)
{
  GstDataRepoSink *sink;

  sink = GST_DATA_REPO_SINK_CAST (basesink);

  return gst_data_repo_sink_open_file (sink);
}

/**
 * @brief Steop datareposink
 */
static gboolean
gst_data_repo_sink_stop (GstBaseSink * basesink)
{
  GstDataRepoSink *sink;
  sink = GST_DATA_REPO_SINK_CAST (basesink);

  /* close the file */
  g_close (sink->fd, NULL);
  sink->fd = 0;

  return TRUE;
}
