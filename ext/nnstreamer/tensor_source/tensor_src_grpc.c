/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Src_gRPC
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    tensor_src_grpc.c
 * @date    20 Oct 2020
 * @brief   GStreamer plugin to support gRPC tensor source
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_src_grpc
 *
 * #tensor_src_grpc extends #gstpushsrc source element to handle gRPC
 * requests as either server or client.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m tensor_src_grpc ! 'other/tensor,dimension=(string)1:1:1:1,type=(string)uint8,framerate=(fraction)10/1' ! fakesink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <errno.h>

#include <gst/gst.h>
#include <glib.h>
#include <gmodule.h>

#include <tensor_typedef.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>

#include "tensor_src_grpc.h"
#include "nnstreamer_grpc.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_src_grpc_debug);
#define GST_CAT_DEFAULT gst_tensor_src_grpc_debug

/**
 * @brief Flag to print minimized log
 */
#define DEFAULT_PROP_SILENT TRUE

/**
 * @brief Default gRPC server mode for tensor source
 */
#define DEFAULT_PROP_SERVER TRUE

/**
 * @brief Default gRPC blocking mode for tensor source
 */
#define DEFAULT_PROP_BLOCKING TRUE

/**
 * @brief Default IDL for RPC comm.
 */
#define DEFAULT_PROP_IDL "protobuf"

/**
 * @brief Default host and port
 */
#define DEFAULT_PROP_HOST  "localhost"
#define DEFAULT_PROP_PORT  55115

#define GST_TENSOR_SRC_GRPC_SCALED_TIME(self, count)\
  gst_util_uint64_scale (count, \
      self->config.rate_d * GST_SECOND, self->config.rate_n)

#define CAPS_STRING GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT

static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

enum
{
  PROP_0,
  PROP_SILENT,
  PROP_SERVER,
  PROP_BLOCKING,
  PROP_IDL,
  PROP_HOST,
  PROP_PORT,
  PROP_OUT,
};

/** gRPC private data */
typedef struct {
  grpc_config config;
  void * instance;
} grpc_private;

#define GET_GRPC_PRIVATE(arg) (grpc_private *) (arg->priv)

/** GObject method implementation */
static void gst_tensor_src_grpc_finalize (GObject * object);
static void gst_tensor_src_grpc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_src_grpc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/** GstBaseSrc method implementation */
static gboolean gst_tensor_src_grpc_start (GstBaseSrc * src);
static gboolean gst_tensor_src_grpc_stop (GstBaseSrc * src);
static gboolean gst_tensor_src_grpc_set_caps (GstBaseSrc * src, GstCaps * caps);
static gboolean gst_tensor_src_grpc_unlock (GstBaseSrc * src);
static gboolean gst_tensor_src_grpc_unlock_stop (GstBaseSrc * src);

/** GstPushSrc method implementation */
static GstFlowReturn gst_tensor_src_grpc_create (GstPushSrc * psrc,
    GstBuffer ** buf);

/** internal functions */
#define gst_tensor_src_grpc_parent_class parent_class
G_DEFINE_TYPE (GstTensorSrcGRPC, gst_tensor_src_grpc, GST_TYPE_PUSH_SRC);

/**
 * @brief initialize the tensor_src_grpc class.
 */
static void
gst_tensor_src_grpc_class_init (GstTensorSrcGRPCClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);
  GstPushSrcClass *gstpushsrc_class = GST_PUSH_SRC_CLASS (klass);

  gobject_class->set_property = gst_tensor_src_grpc_set_property;
  gobject_class->get_property = gst_tensor_src_grpc_get_property;
  gobject_class->finalize = gst_tensor_src_grpc_finalize;

  /* install properties */
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent",
          "Dont' produce verbose output",
          DEFAULT_PROP_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SERVER,
      g_param_spec_boolean ("server", "Server",
          "Specify its working mode either server or client",
          DEFAULT_PROP_SERVER, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_BLOCKING,
      g_param_spec_boolean ("blocking", "Blocking",
          "Specify its working mode either blocking or non-blocking",
          DEFAULT_PROP_BLOCKING, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_IDL,
      g_param_spec_string ("idl", "IDL",
          "Specify Interface Description Language (IDL) for communication",
          DEFAULT_PROP_IDL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host",
          "The hostname to listen as or connect",
          DEFAULT_PROP_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_int ("port", "Port",
          "The port to listen to (0=random available port) or connect",
          0, G_MAXUSHORT, DEFAULT_PROP_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_OUT,
      g_param_spec_uint ("out", "Out",
          "The number of output buffers generated",
          0, G_MAXUINT, 0, G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_static_pad_template (gstelement_class, &srctemplate);

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorSrcGRPC", "Source/Network",
      "Receive nnstreamer protocal buffers as a gRPC server/client",
      "Dongju Chae <dongju.chae@samsung.com>");

  /* GstBasrSrcClass */
  gstbasesrc_class->start = gst_tensor_src_grpc_start;
  gstbasesrc_class->stop = gst_tensor_src_grpc_stop;
  gstbasesrc_class->set_caps = gst_tensor_src_grpc_set_caps;
  gstbasesrc_class->unlock = gst_tensor_src_grpc_unlock;
  gstbasesrc_class->unlock_stop = gst_tensor_src_grpc_unlock_stop;

  /* GstPushSrcClass */
  gstpushsrc_class->create = gst_tensor_src_grpc_create;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_src_grpc_debug,
      "tensor_src_grpc", 0,
      "src element to support protocal buffers as a gRPC server/client");
}

/**
 * @brief callback for checking data_queue full
 */
static gboolean
_data_queue_check_full_cb (GstDataQueue * queue, guint visible,
    guint bytes, guint64 time, gpointer checkdata)
{
  /** it's dummy */
  return FALSE;
}

/**
 * @brief destroy callback for a data queue item
 */
static void
_data_queue_item_free (GstDataQueueItem * item)
{
  if (item->object)
    gst_mini_object_unref (item->object);
  g_free (item);
}

/**
 * @brief send eos event to downstream elements
 */
static void
_send_eos_event (GstTensorSrcGRPC * self)
{
  GstPad *srcpad = GST_BASE_SRC_PAD (&self->element);
  GstEvent *eos = gst_event_new_eos ();
  gst_pad_push_event (srcpad, eos);
}

/**
 * @brief callback function for gRPC requests
 */
static void
_grpc_callback (void *obj, void *data)
{
  GstTensorSrcGRPC *self;

  GstBuffer *buffer;
  GstClockTime duration;
  GstClockTime timestamp;
  GstDataQueueItem *item;

  g_return_if_fail (obj != NULL);
  g_return_if_fail (data != NULL);

  self = GST_TENSOR_SRC_GRPC_CAST (obj);
  buffer = (GstBuffer *) data;

  GST_OBJECT_LOCK (self);

  if (!GST_OBJECT_FLAG_IS_SET (self, GST_TENSOR_SRC_GRPC_STARTED) ||
      !GST_OBJECT_FLAG_IS_SET (self, GST_TENSOR_SRC_GRPC_CONFIGURED)) {
    gst_buffer_unref (buffer);

    GST_OBJECT_UNLOCK (self);
    return;
  }

  if (self->config.rate_n != 0) {
    duration = GST_TENSOR_SRC_GRPC_SCALED_TIME (self, 1);
    timestamp = GST_TENSOR_SRC_GRPC_SCALED_TIME (self, self->out++);
  } else {
    duration = 0;
    timestamp = 0;
  }

  GST_BUFFER_DURATION (buffer) = duration;
  GST_BUFFER_PTS (buffer) = timestamp;

  item = g_new0 (GstDataQueueItem, 1);
  item->object = GST_MINI_OBJECT (buffer);
  item->size = gst_buffer_get_size (buffer);
  item->visible = TRUE;
  item->destroy = (GDestroyNotify) _data_queue_item_free;

  if (!gst_data_queue_push (self->queue, item)) {
    item->destroy (item);
    ml_logw ("Failed to push item because we're flushing");
  } else {
    silent_debug ("new buffer: timestamp %" GST_TIME_FORMAT " duration %"
        GST_TIME_FORMAT, GST_TIME_ARGS (timestamp), GST_TIME_ARGS (duration));
  }

  GST_OBJECT_UNLOCK (self);

  /* special case: framerate == (fraction)0/1 */
  if (duration == 0)
    _send_eos_event (self);
}

/**
 * @brief initialize grpc config.
 */
static void
grpc_config_init (GstTensorSrcGRPC * self)
{
  grpc_private *grpc = GET_GRPC_PRIVATE (self);

  grpc->config.is_server = DEFAULT_PROP_SERVER;
  grpc->config.is_blocking = DEFAULT_PROP_BLOCKING;
  grpc->config.idl = grpc_get_idl (DEFAULT_PROP_IDL);
  grpc->config.dir = GRPC_DIRECTION_BUFFER_TO_TENSORS;
  grpc->config.port = DEFAULT_PROP_PORT;
  grpc->config.host = g_strdup (DEFAULT_PROP_HOST);
  grpc->config.cb = _grpc_callback;
  grpc->config.cb_data = (void *) self;
  grpc->config.config = &self->config;
}

/**
 * @brief initialize tensor_src_grpc element.
 */
static void
gst_tensor_src_grpc_init (GstTensorSrcGRPC * self)
{
  gst_tensors_config_init (&self->config);

  self->queue = gst_data_queue_new (_data_queue_check_full_cb,
      NULL, NULL, NULL);
  self->silent = DEFAULT_PROP_SILENT;
  self->out = 0;

  self->priv = g_new0 (grpc_private, 1);
  grpc_config_init (self);

  GST_OBJECT_FLAG_UNSET (self, GST_TENSOR_SRC_GRPC_CONFIGURED);
  GST_OBJECT_FLAG_UNSET (self, GST_TENSOR_SRC_GRPC_STARTED);
}

/**
 * @brief finalize tensor_src_grpc element.
 */
static void
gst_tensor_src_grpc_finalize (GObject * object)
{
  GstTensorSrcGRPC *self = GST_TENSOR_SRC_GRPC (object);
  grpc_private *grpc = GET_GRPC_PRIVATE (self);

  g_free (grpc->config.host);
  g_free (grpc);
  g_clear_pointer (&self->queue, gst_object_unref);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief start function
 */
static gboolean
gst_tensor_src_grpc_start (GstBaseSrc * src)
{
  GstTensorSrcGRPC *self = GST_TENSOR_SRC_GRPC (src);
  grpc_private *grpc = GET_GRPC_PRIVATE (self);
  gboolean ret;

  if (grpc->instance)
    grpc_destroy (grpc->instance);

  grpc->instance = grpc_new (&grpc->config);
  if (!grpc->instance)
    return FALSE;

  ret = grpc_start (grpc->instance);
  if (ret) {
    GST_OBJECT_FLAG_SET (self, GST_TENSOR_SRC_GRPC_STARTED);

    if (grpc->config.is_server) {
      gint port = grpc_get_listening_port (grpc->instance);
      if (port > 0)
        g_object_set (self, "port", port, NULL);
    }
  }

  return ret;
}

/**
 * @brief stop function
 */
static gboolean
gst_tensor_src_grpc_stop (GstBaseSrc * src)
{
  GstTensorSrcGRPC *self = GST_TENSOR_SRC_GRPC (src);
  grpc_private *grpc = GET_GRPC_PRIVATE (self);

  if (!GST_OBJECT_FLAG_IS_SET (self, GST_TENSOR_SRC_GRPC_STARTED))
    return TRUE;

  _send_eos_event (self);

  if (grpc->instance)
    grpc_destroy (grpc->instance);
  grpc->instance = NULL;

  GST_OBJECT_FLAG_UNSET (self, GST_TENSOR_SRC_GRPC_STARTED);

  return TRUE;
}

/**
 * @brief unlock function, flush any pending data in the data queue
 */
static gboolean
gst_tensor_src_grpc_unlock (GstBaseSrc * src)
{
  GstTensorSrcGRPC *self = GST_TENSOR_SRC_GRPC (src);
  grpc_private *grpc = GET_GRPC_PRIVATE (self);

  /* notify to gRPC */
  if (grpc->instance)
    grpc_stop (grpc->instance);

  silent_debug ("Unlocking create");
  gst_data_queue_set_flushing (self->queue, TRUE);

  return TRUE;
}

/**
 * @brief unlock_stop function, clear the previous unlock request
 */
static gboolean
gst_tensor_src_grpc_unlock_stop (GstBaseSrc * src)
{
  GstTensorSrcGRPC *self = GST_TENSOR_SRC_GRPC (src);

  silent_debug ("Stopping unlock");
  gst_data_queue_set_flushing (self->queue, FALSE);

  return TRUE;
}

/**
 * @brief set caps and configure tensor_src_grpc
 */
static gboolean
gst_tensor_src_grpc_set_caps (GstBaseSrc * src, GstCaps * caps)
{
  GstTensorSrcGRPC *self;
  GstStructure *structure;

  self = GST_TENSOR_SRC_GRPC (src);

  GST_OBJECT_LOCK (self);

  structure = gst_caps_get_structure (caps, 0);
  gst_tensors_config_from_structure (&self->config, structure);

  GST_OBJECT_FLAG_SET (self, GST_TENSOR_SRC_GRPC_CONFIGURED);

  GST_OBJECT_UNLOCK (self);

  return gst_tensors_config_validate (&self->config);
}

/**
 * @brief set a buffer which is a head item in data queue
 */
static GstFlowReturn
gst_tensor_src_grpc_create (GstPushSrc * src, GstBuffer ** buf)
{
  GstTensorSrcGRPC *self = GST_TENSOR_SRC_GRPC_CAST (src);
  GstDataQueueItem *item;

  if (!gst_data_queue_pop (self->queue, &item)) {
    silent_debug ("We're flushing");
    return GST_FLOW_FLUSHING;
  }

  *buf = GST_BUFFER (item->object);
  g_free (item);

  return GST_FLOW_OK;
}

/**
 * @brief check the validity of hostname string
 */
static gboolean
_check_hostname (gchar * str)
{
  if (g_strcmp0 (str, "localhost") == 0 ||
      g_hostname_is_ip_address (str))
    return TRUE;

  return FALSE;
}

/**
 * @brief set tensor_src_grpc properties
 */
static void
gst_tensor_src_grpc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorSrcGRPC *self;
  grpc_private *grpc;

  g_return_if_fail (GST_IS_TENSOR_SRC_GRPC (object));

  self = GST_TENSOR_SRC_GRPC (object);
  grpc = GET_GRPC_PRIVATE (self);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      silent_debug ("Set silent = %d", self->silent);
      break;
    case PROP_SERVER:
      grpc->config.is_server = g_value_get_boolean (value);
      silent_debug ("Set server = %d", grpc->config.is_server);
      break;
    case PROP_BLOCKING:
      grpc->config.is_blocking = g_value_get_boolean (value);
      silent_debug ("Set blocking = %d", grpc->config.is_blocking);
      break;
    case PROP_IDL:
    {
      const gchar * idl_str = g_value_get_string (value);

      if (idl_str) {
        grpc_idl idl = grpc_get_idl (idl_str);
        if (idl != GRPC_IDL_NONE) {
          grpc->config.idl = idl;
          silent_debug ("Set idl = %s", idl_str);
        } else {
          ml_loge ("Invalid IDL string provided: %s", idl_str);
        }
      }
      break;
    }
    case PROP_HOST:
    {
      gchar * host;

      if (!g_value_get_string (value))
        break;

      host = g_value_dup_string (value);
      if (_check_hostname (host)) {
        g_free (grpc->config.host);
        grpc->config.host = host;
        silent_debug ("Set host = %s", grpc->config.host);
      } else {
        g_free (host);
      }
      break;
    }
    case PROP_PORT:
      grpc->config.port = g_value_get_int (value);
      silent_debug ("Set port = %d", grpc->config.port);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get tensor_src_grpc properties
 */
static void
gst_tensor_src_grpc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorSrcGRPC *self;
  grpc_private *grpc;

  g_return_if_fail (GST_IS_TENSOR_SRC_GRPC (object));

  self = GST_TENSOR_SRC_GRPC (object);
  grpc = GET_GRPC_PRIVATE (self);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_SERVER:
      g_value_set_boolean (value, grpc->config.is_server);
      break;
    case PROP_BLOCKING:
      g_value_set_boolean (value, grpc->config.is_blocking);
      break;
    case PROP_IDL:
      switch (grpc->config.idl) {
        case GRPC_IDL_PROTOBUF:
          g_value_set_string (value, "protobuf");
          break;
        case GRPC_IDL_FLATBUF:
          g_value_set_string (value, "flatbuf");
          break;
        default:
          break;
      }
      break;
    case PROP_HOST:
      g_value_set_string (value, grpc->config.host);
      break;
    case PROP_PORT:
      g_value_set_int (value, grpc->config.port);
      break;
    case PROP_OUT:
      g_value_set_uint (value, self->out);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}
