/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Sink_gRPC
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file	tensor_sink_grpc.c
 * @date	22 Oct 2020
 * @brief	GStreamer plugin to support gRPC tensor sink
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Dongju Chae <dongju.chae@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_sink_grpc
 *
 * #tensor_sink_grpc extends #gstbasesink sink element to emit gRPC
 * messages as either server or client.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m videotestsrc
 *        ! video/x-raw,format=RGB,width=640,height=480,framerate=30/1
 *        ! tensor_converter ! tensor_sink_grpc
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

#include "tensor_sink_grpc.h"

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

GST_DEBUG_CATEGORY_STATIC (gst_tensor_sink_grpc_debug);
#define GST_CAT_DEFAULT gst_tensor_sink_grpc_debug

/**
 * @brief Flag to print minimized log
 */
#define DEFAULT_PROP_SILENT TRUE

/**
 * @brief Default gRPC mode for tensor sink
 */
#define DEFAULT_PROP_SERVER FALSE

/**
 * @brief Default IDL for RPC comm.
 */
#define DEFAULT_PROP_IDL "protobuf"

/**
 * @brief Default host and port
 */
#define DEFAULT_PROP_HOST  "localhost"
#define DEFAULT_PROP_PORT  55115

#define CAPS_STRING GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT

static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

enum
{
  PROP_0,
  PROP_SILENT,
  PROP_SERVER,
  PROP_IDL,
  PROP_HOST,
  PROP_PORT,
  PROP_OUT,
};

/** GObject method implementation */
static void gst_tensor_sink_grpc_finalize (GObject * gobject);
static void gst_tensor_sink_grpc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_sink_grpc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/** GstBaseSink method implementation */
static gboolean gst_tensor_sink_grpc_setcaps (GstBaseSink * sink,
    GstCaps * caps);
static GstFlowReturn gst_tensor_sink_grpc_render (GstBaseSink * sink,
    GstBuffer * buf);
static gboolean gst_tensor_sink_grpc_start (GstBaseSink * sink);
static gboolean gst_tensor_sink_grpc_stop (GstBaseSink * sink);

static gboolean gst_tensor_sink_grpc_unlock (GstBaseSink * sink);

/** internal functions */
#define gst_tensor_sink_grpc_parent_class parent_class
G_DEFINE_TYPE (GstTensorSinkGRPC, gst_tensor_sink_grpc, GST_TYPE_BASE_SINK);

/**
 * @brief initialize the tensor_sink_grpc class.
 */
static void
gst_tensor_sink_grpc_class_init (GstTensorSinkGRPCClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbasesink_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasesink_class = (GstBaseSinkClass *) klass;

  parent_class = g_type_class_peek_parent (klass);

  gobject_class->set_property = gst_tensor_sink_grpc_set_property;
  gobject_class->get_property = gst_tensor_sink_grpc_get_property;
  gobject_class->finalize = gst_tensor_sink_grpc_finalize;

  /* install properties */
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent",
        "Dont' produce verbose output",
        DEFAULT_PROP_SILENT,
        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SERVER,
      g_param_spec_boolean ("server", "Server",
        "Specify its working mode either server or client",
        DEFAULT_PROP_SERVER,
        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_IDL,
      g_param_spec_string ("idl", "IDL",
          "Specify Interface Description Language (IDL) for communication",
          DEFAULT_PROP_IDL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host", "The host/IP to send the packets to",
          DEFAULT_PROP_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_int ("port", "Port", "The port to send the packets to",
          0, G_MAXUSHORT, DEFAULT_PROP_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_OUT,
      g_param_spec_uint ("out", "Out",
        "The number of output messages generated",
        0, G_MAXUINT, 0,
        G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_static_pad_template (gstelement_class, &sinktemplate);

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorSinkGRPC", "Sink/Network",
      "Send nnstreamer protocol buffers as gRPC server/client",
      "Dongju Chae <dongju.chae@samsung.com>");

  /* GstBaseSinkClass */
  gstbasesink_class->start = gst_tensor_sink_grpc_start;
  gstbasesink_class->stop = gst_tensor_sink_grpc_stop;
  gstbasesink_class->set_caps = gst_tensor_sink_grpc_setcaps;
  gstbasesink_class->render = gst_tensor_sink_grpc_render;
  gstbasesink_class->unlock = gst_tensor_sink_grpc_unlock;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_sink_grpc_debug,
      "tensor_sink_grpc", 0,
      "sink element to support protocol buffers as a gRPC server/client");
}

/**
 * @brief initialize tensor_sink_grpc element.
 */
static void
gst_tensor_sink_grpc_init (GstTensorSinkGRPC * self)
{
  self->silent = DEFAULT_PROP_SILENT;
  self->server = DEFAULT_PROP_SERVER;
  self->idl = grpc_get_idl (DEFAULT_PROP_IDL);
  self->host = g_strdup (DEFAULT_PROP_HOST);
  self->port = DEFAULT_PROP_PORT;
  self->priv = NULL;
  self->out = 0;

  gst_tensors_config_init (&self->config);

  GST_OBJECT_FLAG_UNSET (self, GST_TENSOR_SINK_GRPC_CONFIGURED);
  GST_OBJECT_FLAG_UNSET (self, GST_TENSOR_SINK_GRPC_STARTED);
}

/**
 * @brief finalize tensor_sink_grpc element.
 */
static void
gst_tensor_sink_grpc_finalize (GObject * gobject)
{
  GstTensorSinkGRPC *this = GST_TENSOR_SINK_GRPC (gobject);

  g_free (this->host);
  this->host = NULL;

  G_OBJECT_CLASS (parent_class)->finalize (gobject);
}

/**
 * @brief set caps of tensor_sink_grpc element.
 */
static gboolean
gst_tensor_sink_grpc_setcaps (GstBaseSink * sink, GstCaps * caps)
{
  GstTensorSinkGRPC * self;
  GstStructure * structure;

  self = GST_TENSOR_SINK_GRPC (sink);

  GST_OBJECT_LOCK (self);

  structure = gst_caps_get_structure (caps, 0);
  gst_tensors_config_from_structure (&self->config, structure);

  GST_OBJECT_FLAG_SET (self, GST_TENSOR_SINK_GRPC_CONFIGURED);

  grpc_set_config (self, &self->config);

  GST_OBJECT_UNLOCK (self);

  return gst_tensors_config_validate (&self->config);
}

/**
 * @brief render function of tensor_sink_grpc element.
 */
static GstFlowReturn
gst_tensor_sink_grpc_render (GstBaseSink * sink, GstBuffer * buf)
{
  GstTensorSinkGRPC * self = GST_TENSOR_SINK_GRPC (sink);
  gboolean ret;

  g_return_val_if_fail (
      GST_OBJECT_FLAG_IS_SET (self, GST_TENSOR_SINK_GRPC_STARTED),
      GST_FLOW_FLUSHING);

  ret = grpc_send (self, buf);

  return ret ? GST_FLOW_OK : GST_FLOW_ERROR;
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
 * @brief set properties of tensor_sink_grpc element.
 */
static void
gst_tensor_sink_grpc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorSinkGRPC * self;

  g_return_if_fail (GST_IS_TENSOR_SINK_GRPC (object));
  self = GST_TENSOR_SINK_GRPC (object);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      silent_debug ("Set silent = %d", self->silent);
      break;
    case PROP_SERVER:
      self->server = g_value_get_boolean (value);
      silent_debug ("Set server = %d", self->server);
      break;
    case PROP_IDL:
    {
      const gchar * idl_str = g_value_get_string (value);

      if (idl_str) {
        grpc_idl idl = grpc_get_idl (idl_str);
        if (idl != GRPC_IDL_NONE) {
          self->idl = idl;
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
        g_free (self->host);
        self->host = host;
        silent_debug ("Set host = %s", self->host);
      } else {
        g_free (host);
      }
      break;
    }
    case PROP_PORT:
      self->port = g_value_get_int (value);
      silent_debug ("Set port = %d", self->port);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get properties of tensor_sink_grpc element.
 */
static void
gst_tensor_sink_grpc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorSinkGRPC * self;

  g_return_if_fail (GST_IS_TENSOR_SINK_GRPC (object));
  self = GST_TENSOR_SINK_GRPC (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_SERVER:
      g_value_set_boolean (value, self->server);
      break;
    case PROP_IDL:
      switch (self->idl) {
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
      g_value_set_string (value, self->host);
      break;
    case PROP_PORT:
      g_value_set_int (value, self->port);
      break;
    case PROP_OUT:
      g_value_set_uint (value, self->out);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief start tensor_sink_grpc element.
 */
static gboolean
gst_tensor_sink_grpc_start (GstBaseSink * sink)
{
  GstTensorSinkGRPC *self = GST_TENSOR_SINK_GRPC (sink);
  gboolean ret;

  if (GST_OBJECT_FLAG_IS_SET (self, GST_TENSOR_SINK_GRPC_STARTED))
    return TRUE;

  if (self->priv)
    grpc_destroy (self);

  self->priv = grpc_new (self);
  if (!self->priv)
    return FALSE;

  ret = grpc_start (self, GRPC_DIRECTION_TENSORS_TO_BUFFER);
  if (ret) {
    GST_OBJECT_FLAG_SET (self, GST_TENSOR_SINK_GRPC_STARTED);

    if (self->server)
      g_object_set (self, "port", grpc_get_listening_port (self), NULL);
  }

  return TRUE;
}

/**
 * @brief stop tensor_sink_grpc element.
 */
static gboolean
gst_tensor_sink_grpc_stop (GstBaseSink * sink)
{
  GstTensorSinkGRPC *self = GST_TENSOR_SINK_GRPC (sink);

  if (!GST_OBJECT_FLAG_IS_SET (self, GST_TENSOR_SINK_GRPC_STARTED))
    return TRUE;

  if (self->priv)
    grpc_destroy (self);
  self->priv = NULL;

  GST_OBJECT_FLAG_UNSET (self, GST_TENSOR_SINK_GRPC_STARTED);

  return TRUE;
}

/**
 * @brief unlock any blocking operations
 */
static gboolean
gst_tensor_sink_grpc_unlock (GstBaseSink * sink)
{
  GstTensorSinkGRPC *self = GST_TENSOR_SINK_GRPC (sink);

  /* notify to gRPC */
  grpc_stop (self);

  return TRUE;
}
