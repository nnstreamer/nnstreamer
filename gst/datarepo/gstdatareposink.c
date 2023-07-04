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
 * gst-launch-1.0 videotestsrc ! datareposink location=filename json=video.json
 * gst-launch-1.0 videotestsrc ! pngenc ! datareposink location=image_%02d.png json=video.json
 * gst-launch-1.0 audiotestsrc samplesperbuffer=44100 ! audio/x-raw, format=S16LE, layout=interleaved, rate=44100, channels=1 ! \
 * datareposink location=filename json=audio.json
 * gst-launch-1.0 datareposrc location=file.dat json=file.json tensors-sequence=2,3 start-sample-index=0 stop-sample-index=199 epochs=1 !  \
 * other/tensors, format=static, num_tensors=2, framerate=0/1, dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! \
 * datareposink location=hyunil.dat json=file.json
 * ]|
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <gst/gst.h>
#include <gst/video/video-info.h>
#include <gst/audio/audio-info.h>
#include <glib/gstdio.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <nnstreamer_plugin_api.h>
#include <tensor_common.h>
#include <nnstreamer_util.h>
#include "gstdatareposink.h"

/**
 * @brief Tensors caps
 */
#define TENSOR_CAPS GST_TENSORS_CAP_MAKE ("{ static, flexible }")
/**
 * @brief Video caps
 */
#define SUPPORTED_VIDEO_FORMAT \
  "{RGB, BGR, RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR, GRAY8}"
#define VIDEO_CAPS GST_VIDEO_CAPS_MAKE (SUPPORTED_VIDEO_FORMAT) "," \
  "interlace-mode = (string) progressive"
/**
 * @brief Audio caps
 */
#define SUPPORTED_AUDIO_FORMAT \
  "{S8, U8, S16LE, S16BE, U16LE, U16BE, S32LE, S32BE, U32LE, U32BE, F32LE, F32BE, F64LE, F64BE}"
#define AUDIO_CAPS GST_AUDIO_CAPS_MAKE (SUPPORTED_AUDIO_FORMAT) "," \
  "layout = (string) interleaved"
/**
 * @brief Text caps
 */
#define TEXT_CAPS "text/x-raw, format = (string) utf8"
/**
 * @brief Octet caps
 */
#define OCTET_CAPS "application/octet-stream"
/**
 * @brief Image caps
 */
#define IMAGE_CAPS \
  "image/png, width = (int) [ 16, 1000000 ], height = (int) [ 16, 1000000 ], framerate = (fraction) [ 0/1, MAX];" \
  "image/jpeg, width = (int) [ 16, 65535 ], height = (int) [ 16, 65535 ], framerate = (fraction) [ 0/1, MAX], sof-marker = (int) { 0, 1, 2, 4, 9 };" \
  "image/tiff, endianness = (int) { BIG_ENDIAN, LITTLE_ENDIAN };" \
  "image/gif;" \
  "image/bmp"

static GstStaticPadTemplate sinktemplate =
    GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (TENSOR_CAPS ";" VIDEO_CAPS ";" AUDIO_CAPS ";" IMAGE_CAPS
        ";" TEXT_CAPS ";" OCTET_CAPS));

/**
 * @brief datareposink properties.
 */
enum
{
  PROP_0,
  PROP_LOCATION,
  PROP_JSON
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
static gboolean gst_data_repo_sink_stop (GstBaseSink * basesink);
static GstStateChangeReturn gst_data_repo_sink_change_state (GstElement *
    element, GstStateChange transition);
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
          "if the files are images, use placeholder in indexes for filename"
          "(e.g., filenmae%04d.png).",
          NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_JSON,
      g_param_spec_string ("json", "JSON file path",
          "JSON file path to write the meta information of a sample", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  gst_element_class_set_static_metadata (gstelement_class,
      "NNStreamer MLOps Data Repository Sink",
      "Sink/File",
      "Write files to MLOps Data Repository", "Samsung Electronics Co., Ltd.");

  gst_element_class_add_static_pad_template (gstelement_class, &sinktemplate);

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_data_repo_sink_change_state);
  gstbasesink_class->render = GST_DEBUG_FUNCPTR (gst_data_repo_sink_render);
  gstbasesink_class->get_caps = GST_DEBUG_FUNCPTR (gst_data_repo_sink_get_caps);
  gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR (gst_data_repo_sink_set_caps);
  gstbasesink_class->event = GST_DEBUG_FUNCPTR (gst_data_repo_sink_event);
  gstbasesink_class->query = GST_DEBUG_FUNCPTR (gst_data_repo_sink_query);
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
  sink->fd_offset = 0;
  sink->data_type = GST_DATA_REPO_DATA_UNKNOWN;
  sink->is_flexible_tensors = FALSE;
  sink->fixed_caps = NULL;
  sink->json_object = NULL;
  sink->total_samples = 0;
  sink->flexible_tensor_count = 0;
  sink->json_object = json_object_new ();
  sink->sample_offset_array = json_array_new ();
  sink->tensor_size_array = json_array_new ();
  sink->tensor_count_array = json_array_new ();
}

/**
 * @brief finalize datareposink.
 */
static void
gst_data_repo_sink_finalize (GObject * object)
{
  GstDataRepoSink *sink = GST_DATA_REPO_SINK (object);

  g_free (sink->filename);
  g_free (sink->json_filename);

  if (sink->fd) {
    g_close (sink->fd, NULL);
    sink->fd = 0;
  }

  /* Check for gst-inspect log */
  if (sink->fixed_caps)
    gst_caps_unref (sink->fixed_caps);

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
      GST_INFO_OBJECT (sink, "filename: %s", sink->filename);
      break;
    case PROP_JSON:
      sink->json_filename = g_value_dup_string (value);
      GST_INFO_OBJECT (sink, "JSON filename: %s", sink->json_filename);
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
    case PROP_JSON:
      g_value_set_string (value, sink->json_filename);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Function to write others media type (tensors(fixed), video, audio, octet and text)
 */
static GstFlowReturn
gst_data_repo_sink_write_others (GstDataRepoSink * sink, GstBuffer * buffer)
{
  gsize write_size = 0;
  GstMapInfo info;

  g_return_val_if_fail (sink != NULL, GST_FLOW_ERROR);
  g_return_val_if_fail (buffer != NULL, GST_FLOW_ERROR);
  g_return_val_if_fail (sink->fd != 0, GST_FLOW_ERROR);

  GST_OBJECT_LOCK (sink);
  gst_buffer_map (buffer, &info, GST_MAP_READ);
  sink->sample_size = info.size;

  GST_LOG_OBJECT (sink,
      "Writing %lld bytes at offset 0x%" G_GINT64_MODIFIER "x (%lld size)",
      (long long) info.size, sink->fd_offset, (long long) sink->fd_offset);

  write_size = write (sink->fd, info.data, info.size);

  gst_buffer_unmap (buffer, &info);
  GST_OBJECT_UNLOCK (sink);

  if (write_size != info.size) {
    GST_ERROR_OBJECT (sink, "Could not write data to file");
    return GST_FLOW_ERROR;
  }

  sink->fd_offset += write_size;
  sink->total_samples++;

  return GST_FLOW_OK;
}

/**
 * @brief Function to write flexible tensors
 */
static GstFlowReturn
gst_data_repo_sink_write_flexible_tensors (GstDataRepoSink * sink,
    GstBuffer * buffer)
{
  guint num_tensors, i;
  gsize write_size = 0, total_write = 0, tensor_size;
  GstMapInfo info;
  GstMemory *mem = NULL;
  GstTensorMetaInfo meta;

  g_return_val_if_fail (sink != NULL, GST_FLOW_ERROR);
  g_return_val_if_fail (buffer != NULL, GST_FLOW_ERROR);
  g_return_val_if_fail (sink->fd != 0, GST_FLOW_ERROR);
  g_return_val_if_fail (sink->json_object != NULL, GST_FLOW_ERROR);
  g_return_val_if_fail (sink->sample_offset_array != NULL, GST_FLOW_ERROR);
  g_return_val_if_fail (sink->tensor_size_array != NULL, GST_FLOW_ERROR);
  g_return_val_if_fail (sink->tensor_count_array != NULL, GST_FLOW_ERROR);

  GST_OBJECT_LOCK (sink);

  num_tensors = gst_tensor_buffer_get_count (buffer);
  GST_INFO_OBJECT (sink, "num_tensors: %u", num_tensors);

  for (i = 0; i < num_tensors; i++) {
    mem = gst_tensor_buffer_get_nth_memory (buffer, i);
    if (!gst_memory_map (mem, &info, GST_MAP_READ)) {
      GST_ERROR_OBJECT (sink, "Failed to map memory");
      goto mem_map_error;
    }

    if (!gst_tensor_meta_info_parse_header (&meta, info.data)) {
      GST_ERROR_OBJECT (sink, "Invalid flexible tensors");
      goto error;
    }
    tensor_size = info.size;

    GST_LOG_OBJECT (sink, "tensor[%u] size: %zd", i, tensor_size);
    GST_LOG_OBJECT (sink,
        "Writing %lld bytes at offset 0x%" G_GINT64_MODIFIER "x (%lld size)",
        (long long) tensor_size, sink->fd_offset + total_write,
        (long long) sink->fd_offset + total_write);

    write_size = write (sink->fd, info.data, tensor_size);
    if (write_size != tensor_size) {
      GST_ERROR_OBJECT (sink, "Could not write data to file");
      goto error;
    }

    json_array_add_int_element (sink->tensor_size_array, tensor_size);
    total_write += write_size;

    gst_memory_unmap (mem, &info);
    gst_memory_unref (mem);
  }

  json_array_add_int_element (sink->sample_offset_array, sink->fd_offset);
  sink->fd_offset += total_write;

  GST_LOG_OBJECT (sink, "flexible_tensor_count: %u",
      sink->flexible_tensor_count);
  json_array_add_int_element (sink->tensor_count_array,
      sink->flexible_tensor_count);
  sink->flexible_tensor_count += num_tensors;

  sink->total_samples++;

  GST_OBJECT_UNLOCK (sink);

  return GST_FLOW_OK;

error:
  gst_memory_unmap (mem, &info);
mem_map_error:
  gst_memory_unref (mem);
  GST_OBJECT_UNLOCK (sink);

  return GST_FLOW_ERROR;
}

/**
 * @brief Get image filename
 */
static gchar *
gst_data_repo_sink_get_image_filename (GstDataRepoSink * sink)
{
  gchar *filename = NULL;

  g_return_val_if_fail (sink != NULL, NULL);
  g_return_val_if_fail (sink->data_type == GST_DATA_REPO_DATA_IMAGE, NULL);
  g_return_val_if_fail (sink->filename != NULL, NULL);

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
  /* let's set value by property */
  filename = g_strdup_printf (sink->filename, sink->total_samples);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

  return filename;
}

/**
 * @brief Function to read multi image files
 */
static GstFlowReturn
gst_data_repo_sink_write_multi_images (GstDataRepoSink * sink,
    GstBuffer * buffer)
{
  gchar *filename;
  gboolean ret;
  GError *error = NULL;
  GstMapInfo info;

  g_return_val_if_fail (sink != NULL, GST_FLOW_ERROR);
  g_return_val_if_fail (buffer != NULL, GST_FLOW_ERROR);

  filename = gst_data_repo_sink_get_image_filename (sink);

  GST_OBJECT_LOCK (sink);
  gst_buffer_map (buffer, &info, GST_MAP_READ);

  sink->sample_size = info.size;

  GST_DEBUG_OBJECT (sink, "Writing to file \"%s\", size(%zd)", filename,
      info.size);
  ret = g_file_set_contents (filename, (char *) info.data, info.size, &error);

  gst_buffer_unmap (buffer, &info);
  GST_OBJECT_UNLOCK (sink);

  g_free (filename);

  if (!ret) {
    GST_ERROR_OBJECT (sink, "Could not write data to file: %s",
        error ? error->message : "unknown error");
    g_clear_error (&error);
    return GST_FLOW_ERROR;
  }

  sink->total_samples++;

  return GST_FLOW_OK;
}

/**
 * @brief Called when a buffer should be presented or ouput.
 */
static GstFlowReturn
gst_data_repo_sink_render (GstBaseSink * bsink, GstBuffer * buffer)
{
  GstDataRepoSink *sink = GST_DATA_REPO_SINK_CAST (bsink);

  sink->is_flexible_tensors =
      gst_tensor_pad_caps_is_flexible (GST_BASE_SINK_PAD (sink));

  switch (sink->data_type) {
    case GST_DATA_REPO_DATA_VIDEO:
    case GST_DATA_REPO_DATA_AUDIO:
    case GST_DATA_REPO_DATA_TEXT:
    case GST_DATA_REPO_DATA_OCTET:
    case GST_DATA_REPO_DATA_TENSOR:
      if (sink->is_flexible_tensors)
        return gst_data_repo_sink_write_flexible_tensors (sink, buffer);
      /* default write function for tensors(fixed), video, audio, text and octet */
      return gst_data_repo_sink_write_others (sink, buffer);
    case GST_DATA_REPO_DATA_IMAGE:
      return gst_data_repo_sink_write_multi_images (sink, buffer);
    default:
      return GST_FLOW_ERROR;
  }
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

  sink->data_type = gst_data_repo_get_data_type_from_caps (caps);
  sink->fixed_caps = gst_caps_copy (caps);

  GST_DEBUG_OBJECT (sink, "data type: %d", sink->data_type);
  return (sink->data_type != GST_DATA_REPO_DATA_UNKNOWN);
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
  g_return_val_if_fail (sink->data_type != GST_DATA_REPO_DATA_UNKNOWN, FALSE);

  if (sink->filename == NULL || sink->filename[0] == '\0')
    goto no_filename;

  /* for image, g_file_set_contents() is used in the write function */
  if (sink->data_type == GST_DATA_REPO_DATA_IMAGE) {
    return TRUE;
  }

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
 * @brief Stop datareposink
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

/**
 * @brief Write json to file
 */
static gboolean
__write_json (JsonObject * object, const gchar * filename)
{
  JsonNode *root;
  JsonGenerator *generator;
  gboolean ret = TRUE;

  g_return_val_if_fail (object != NULL, FALSE);
  g_return_val_if_fail (filename != NULL, FALSE);

  /* Make it the root node */
  root = json_node_init_object (json_node_alloc (), object);
  generator = json_generator_new ();
  json_generator_set_root (generator, root);
  json_generator_set_pretty (generator, TRUE);
  ret = json_generator_to_file (generator, filename, NULL);
  if (!ret) {
    GST_ERROR ("Failed to write JSON to file %s", filename);
  }

  /* Release everything */
  g_object_unref (generator);
  json_node_free (root);

  return ret;
}

/**
 * @brief write the meta information to a JSON file
 */
static gboolean
gst_data_repo_sink_write_json_meta_file (GstDataRepoSink * sink)
{
  gchar *caps_str = NULL;
  gboolean ret = TRUE;

  g_return_val_if_fail (sink != NULL, FALSE);
  g_return_val_if_fail (sink->json_filename != NULL, FALSE);
  g_return_val_if_fail (sink->data_type != GST_DATA_REPO_DATA_UNKNOWN, FALSE);
  g_return_val_if_fail (sink->fixed_caps != NULL, FALSE);
  g_return_val_if_fail (sink->json_object != NULL, FALSE);
  g_return_val_if_fail (sink->sample_offset_array != NULL, FALSE);
  g_return_val_if_fail (sink->tensor_size_array != NULL, FALSE);
  g_return_val_if_fail (sink->tensor_count_array != NULL, GST_FLOW_ERROR);

  caps_str = gst_caps_to_string (sink->fixed_caps);
  GST_DEBUG_OBJECT (sink, "caps string: %s", caps_str);

  json_object_set_string_member (sink->json_object, "gst_caps", caps_str);

  json_object_set_int_member (sink->json_object, "total_samples",
      sink->total_samples);

  if (sink->is_flexible_tensors) {
    json_object_set_array_member (sink->json_object, "sample_offset",
        sink->sample_offset_array);
    json_object_set_array_member (sink->json_object, "tensor_size",
        sink->tensor_size_array);
    json_object_set_array_member (sink->json_object, "tensor_count",
        sink->tensor_count_array);
  } else {
    json_object_set_int_member (sink->json_object, "sample_size",
        sink->sample_size);
  }
  ret = __write_json (sink->json_object, sink->json_filename);
  if (!ret) {
    GST_ERROR_OBJECT (sink, "Failed to write json meta file: %s",
        sink->json_filename);
  }

  json_object_unref (sink->json_object);
  g_free (caps_str);

  return ret;
}

/**
 * @brief Change state of datareposink.
 */
static GstStateChangeReturn
gst_data_repo_sink_change_state (GstElement * element,
    GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstDataRepoSink *sink = GST_DATA_REPO_SINK (element);

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      GST_INFO_OBJECT (sink, "NULL_TO_READY");
      if (sink->filename == NULL || sink->json_filename == NULL) {
        GST_ERROR_OBJECT (sink, "Set filenmae and json");
        goto state_change_failed;
      }
      break;

    case GST_STATE_CHANGE_READY_TO_PAUSED:
      GST_INFO_OBJECT (sink, "READY_TO_PAUSED");
      break;

    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_INFO_OBJECT (sink, "PAUSED_TO_PLAYING");

      if (!gst_data_repo_sink_open_file (sink))
        goto state_change_failed;
      break;

    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      GST_INFO_OBJECT (sink, "PLAYING_TO_PAUSED");
      break;

    case GST_STATE_CHANGE_PAUSED_TO_READY:
      GST_INFO_OBJECT (sink, "PAUSED_TO_READY");
      break;

    case GST_STATE_CHANGE_READY_TO_NULL:
      GST_INFO_OBJECT (sink, "READY_TO_NULL");
      if (!gst_data_repo_sink_write_json_meta_file (sink))
        goto state_change_failed;
      break;

    default:
      break;
  }
  return ret;

state_change_failed:
  GST_ERROR_OBJECT (sink, "state change failed");

  return GST_STATE_CHANGE_FAILURE;
}
