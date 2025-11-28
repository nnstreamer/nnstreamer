/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_src.c
 * @date    02 Aug 2022
 * @brief   Subscribe and push incoming data to the GStreamer pipeline
 * @author  Yechan Choi <yechan9.choi@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "edge_src.h"

GST_DEBUG_CATEGORY_STATIC (gst_edgesrc_debug);
#define GST_CAT_DEFAULT gst_edgesrc_debug

/**
 * @brief the capabilities of the outputs
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

/**
 * @brief edgesrc properties
 */
enum
{
  PROP_0,
  PROP_HOST,
  PROP_PORT,
  PROP_DEST_HOST,
  PROP_DEST_PORT,
  PROP_CONNECT_TYPE,
  PROP_TOPIC,
  PROP_CUSTOM_LIB,

  PROP_LAST
};

#define gst_edgesrc_parent_class parent_class
G_DEFINE_TYPE (GstEdgeSrc, gst_edgesrc, GST_TYPE_BASE_SRC);

static void gst_edgesrc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_edgesrc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_edgesrc_class_finalize (GObject * object);

static gboolean gst_edgesrc_start (GstBaseSrc * basesrc);
static gboolean gst_edgesrc_stop (GstBaseSrc * basesrc);
static GstFlowReturn gst_edgesrc_create (GstBaseSrc * basesrc, guint64 offset,
    guint size, GstBuffer ** out_buf);

static gchar *gst_edgesrc_get_dest_host (GstEdgeSrc * self);
static void gst_edgesrc_set_dest_host (GstEdgeSrc * self,
    const gchar * dest_host);

static guint16 gst_edgesrc_get_dest_port (GstEdgeSrc * self);
static void gst_edgesrc_set_dest_port (GstEdgeSrc * self,
    const guint16 dest_port);

static nns_edge_connect_type_e gst_edgesrc_get_connect_type (GstEdgeSrc * self);
static void gst_edgesrc_set_connect_type (GstEdgeSrc * self,
    const nns_edge_connect_type_e connect_type);
static GstStateChangeReturn gst_edgesrc_change_state (GstElement * element,
    GstStateChange transition);

/**
 * @brief initialize the class
 */
static void
gst_edgesrc_class_init (GstEdgeSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);

  gobject_class->set_property = gst_edgesrc_set_property;
  gobject_class->get_property = gst_edgesrc_get_property;
  gobject_class->finalize = gst_edgesrc_class_finalize;

  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host",
          "A self host address (DEPRECATED, has no effect).", DEFAULT_HOST,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_DEPRECATED));
  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_uint ("port", "Port",
          "A self port number (DEPRECATED, has no effect).",
          0, 65535, DEFAULT_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_DEPRECATED));
  g_object_class_install_property (gobject_class, PROP_DEST_HOST,
      g_param_spec_string ("dest-host", "Destination Host",
          "A host address of edgesink to receive the packets from edgesink",
          DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_DEST_PORT,
      g_param_spec_uint ("dest-port", "Destination Port",
          "A port of edgesink to receive the packets from edgesink", 0, 65535,
          DEFAULT_PORT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_CONNECT_TYPE,
      g_param_spec_enum ("connect-type", "Connect Type",
          "The connections type between edgesink and edgesrc.",
          GST_TYPE_EDGE_CONNECT_TYPE, DEFAULT_CONNECT_TYPE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TOPIC,
      g_param_spec_string ("topic", "Topic",
          "The main topic of the host and option if necessary. "
          "(topic)/(optional topic for main topic).", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_CUSTOM_LIB,
      g_param_spec_string ("custom-lib", "Custom connection lib path",
          "User defined custom connection lib path.",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&srctemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "EdgeSrc", "Source/Edge",
      "Subscribe and push incoming streams", "Samsung Electronics Co., Ltd.");

  gstbasesrc_class->start = gst_edgesrc_start;
  gstbasesrc_class->stop = gst_edgesrc_stop;
  gstbasesrc_class->create = gst_edgesrc_create;
  gstelement_class->change_state = gst_edgesrc_change_state;

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT,
      GST_EDGE_ELEM_NAME_SRC, 0, "Edge src");
}

/**
 * @brief initialize edgesrc element
 */
static void
gst_edgesrc_init (GstEdgeSrc * self)
{
  GstBaseSrc *basesrc = GST_BASE_SRC (self);

  gst_base_src_set_format (basesrc, GST_FORMAT_TIME);
  gst_base_src_set_async (basesrc, FALSE);

  self->dest_host = g_strdup (DEFAULT_HOST);
  self->dest_port = DEFAULT_PORT;
  self->topic = NULL;
  self->msg_queue = g_async_queue_new ();
  self->connect_type = DEFAULT_CONNECT_TYPE;
  self->playing = FALSE;
  self->custom_lib = NULL;
  self->edge_h = NULL;
}

/**
 * @brief set property
 */
static void
gst_edgesrc_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstEdgeSrc *self = GST_EDGESRC (object);

  switch (prop_id) {
    case PROP_HOST:
      nns_logw ("host property is deprecated.");
      break;
    case PROP_PORT:
      nns_logw ("port property is deprecated.");
      break;
    case PROP_DEST_HOST:
      gst_edgesrc_set_dest_host (self, g_value_get_string (value));
      break;
    case PROP_DEST_PORT:
      gst_edgesrc_set_dest_port (self, g_value_get_uint (value));
      break;
    case PROP_CONNECT_TYPE:
      gst_edgesrc_set_connect_type (self, g_value_get_enum (value));
      break;
    case PROP_TOPIC:
      if (!g_value_get_string (value)) {
        nns_logw ("topic property cannot be NULL. Query-hybrid is disabled.");
        break;
      }
      g_free (self->topic);
      self->topic = g_value_dup_string (value);
      break;
    case PROP_CUSTOM_LIB:
      g_free (self->custom_lib);
      self->custom_lib = g_value_dup_string (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get property
 */
static void
gst_edgesrc_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstEdgeSrc *self = GST_EDGESRC (object);

  switch (prop_id) {
    case PROP_HOST:
      nns_logw ("host property is deprecated.");
      break;
    case PROP_PORT:
      nns_logw ("port property is deprecated.");
      break;
    case PROP_DEST_HOST:
      g_value_set_string (value, gst_edgesrc_get_dest_host (self));
      break;
    case PROP_DEST_PORT:
      g_value_set_uint (value, gst_edgesrc_get_dest_port (self));
      break;
    case PROP_CONNECT_TYPE:
      g_value_set_enum (value, gst_edgesrc_get_connect_type (self));
      break;
    case PROP_TOPIC:
      g_value_set_string (value, self->topic);
      break;
    case PROP_CUSTOM_LIB:
      g_value_set_string (value, self->custom_lib);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief finalize the object
 */
static void
gst_edgesrc_class_finalize (GObject * object)
{
  GstEdgeSrc *self = GST_EDGESRC (object);
  nns_edge_data_h data_h;

  self->playing = FALSE;
  g_free (self->dest_host);
  self->dest_host = NULL;

  g_free (self->topic);
  self->topic = NULL;

  g_free (self->custom_lib);
  self->custom_lib = NULL;

  if (self->msg_queue) {
    while ((data_h = g_async_queue_try_pop (self->msg_queue))) {
      nns_edge_data_destroy (data_h);
    }
    g_async_queue_unref (self->msg_queue);
    self->msg_queue = NULL;
  }

  if (self->edge_h) {
    nns_edge_release_handle (self->edge_h);
    self->edge_h = NULL;
  }
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Change state of edgesrc.
 */
static GstStateChangeReturn
gst_edgesrc_change_state (GstElement * element, GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstEdgeSrc *self = GST_EDGESRC (element);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_INFO_OBJECT (self, "State changed from PAUSED to PLAYING.");
      self->playing = TRUE;
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      GST_INFO_OBJECT (self, "State changed from PLAYING to PAUSED.");
      self->playing = FALSE;
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief nnstreamer-edge event callback.
 */
static int
_nns_edge_event_cb (nns_edge_event_h event_h, void *user_data)
{
  nns_edge_event_e event_type;
  int ret = NNS_EDGE_ERROR_NONE;

  GstEdgeSrc *self = GST_EDGESRC (user_data);

  if (0 != nns_edge_event_get_type (event_h, &event_type)) {
    nns_loge ("Failed to get event type!");
    return NNS_EDGE_ERROR_UNKNOWN;
  }

  switch (event_type) {
    case NNS_EDGE_EVENT_NEW_DATA_RECEIVED:
    {
      nns_edge_data_h data;

      nns_edge_event_parse_new_data (event_h, &data);
      g_async_queue_push (self->msg_queue, data);
      break;
    }
    case NNS_EDGE_EVENT_CONNECTION_CLOSED:
    {
      self->playing = FALSE;
      break;
    }
    default:
      break;
  }

  return ret;
}

/**
 * @brief Release nnstreamer-edge handle and reset the playing state.
 */
static void
gst_edgesrc_release_handle (GstEdgeSrc * self)
{
  if (self->edge_h) {
    nns_edge_release_handle (self->edge_h);
    self->edge_h = NULL;
  }

  self->playing = FALSE;
}

/**
 * @brief start edgesrc, called when state changed null to ready
 */
static gboolean
gst_edgesrc_start (GstBaseSrc * basesrc)
{
  GstEdgeSrc *self = GST_EDGESRC (basesrc);

  int ret;
  char *port = NULL;

  if (NNS_EDGE_CONNECT_TYPE_CUSTOM != self->connect_type) {
    ret = nns_edge_create_handle (NULL, self->connect_type,
      NNS_EDGE_NODE_TYPE_SUB, &self->edge_h);
  } else {
    if (!self->custom_lib) {
      nns_loge ("Failed to create custom handle. Custom library is not set.");
      return FALSE;
    }
    ret = nns_edge_custom_create_handle (NULL, self->custom_lib,
      NNS_EDGE_NODE_TYPE_SUB, &self->edge_h);
  }

  if (NNS_EDGE_ERROR_NONE != ret) {
    nns_loge ("Failed to get nnstreamer edge handle.");

    gst_edgesrc_release_handle (self);

    return FALSE;
  }

  if (self->dest_host)
    nns_edge_set_info (self->edge_h, "DEST_HOST", self->dest_host);
  if (self->dest_port > 0) {
    port = g_strdup_printf ("%u", self->dest_port);
    nns_edge_set_info (self->edge_h, "DEST_PORT", port);
    g_free (port);
  }
  if (self->topic)
    nns_edge_set_info (self->edge_h, "TOPIC", self->topic);

  nns_edge_set_event_callback (self->edge_h, _nns_edge_event_cb, self);

  if (0 != nns_edge_start (self->edge_h)) {
    nns_loge
        ("Failed to start NNStreamer-edge. Please check server IP and port.");
    gst_edgesrc_release_handle (self);
    return FALSE;
  }

  if (0 != nns_edge_connect (self->edge_h, self->dest_host, self->dest_port)) {
    nns_loge ("Failed to connect to edge server!");
    gst_edgesrc_release_handle (self);
    return FALSE;
  }
  self->playing = TRUE;

  return TRUE;
}

/**
 * @brief Stop edgesrc, called when state changed ready to null
 */
static gboolean
gst_edgesrc_stop (GstBaseSrc * basesrc)
{
  GstEdgeSrc *self = GST_EDGESRC (basesrc);
  int ret;

  if (!self->edge_h)
    return TRUE;

  self->playing = FALSE;
  ret = nns_edge_stop (self->edge_h);

  if (NNS_EDGE_ERROR_NONE != ret) {
    nns_loge ("Failed to stop edgesrc (error code: %d).", ret);
    gst_edgesrc_release_handle (self);
    return FALSE;
  }

  gst_edgesrc_release_handle (self);

  return TRUE;
}

/**
 * @brief Create a buffer containing the subscribed data
 */
static GstFlowReturn
gst_edgesrc_create (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** out_buf)
{
  GstEdgeSrc *self = GST_EDGESRC (basesrc);
  GstPad *pad = GST_BASE_SRC_PAD (basesrc);
  nns_edge_data_h data_h = NULL;
  GstBuffer *buffer = NULL;
  GstMemory *mem;
  GstCaps *caps = NULL;
  GstTensorsConfig config;
  GstTensorInfo *_info;
  gboolean is_tensor = FALSE;
  guint i, num_data, max_mems;
  int ret;

  UNUSED (offset);
  UNUSED (size);
  gst_tensors_config_init (&config);

  while (self->playing && !data_h) {
    data_h = g_async_queue_timeout_pop (self->msg_queue, G_USEC_PER_SEC);
  }

  if (!data_h) {
    nns_loge ("Failed to get message from the edgesrc message queue.");
    return GST_FLOW_ERROR;
  }

  ret = nns_edge_data_get_count (data_h, &num_data);
  if (ret != NNS_EDGE_ERROR_NONE || num_data == 0) {
    nns_loge ("Failed to get the number of memories in the edge data.");
    goto done;
  }

  /* Check current caps and max memory. */
  caps = gst_pad_get_current_caps (pad);
  if (!caps)
    caps = gst_pad_get_allowed_caps (pad);

  if (caps) {
    is_tensor = gst_tensors_config_from_caps (&config, caps, TRUE);
    gst_caps_unref (caps);
  }

  max_mems = is_tensor ? NNS_TENSOR_SIZE_LIMIT : gst_buffer_get_max_memory ();
  if (num_data > max_mems) {
    nns_loge
        ("Cannot create new buffer. The edge-data has %u memories, but allowed memories is %u.",
        num_data, max_mems);
    goto done;
  }

  buffer = gst_buffer_new ();
  for (i = 0; i < num_data; i++) {
    void *data = NULL;
    nns_size_t data_len = 0;
    gpointer new_data;

    nns_edge_data_get (data_h, i, &data, &data_len);
    new_data = _g_memdup (data, data_len);
    mem = gst_memory_new_wrapped (0, new_data, data_len, 0, data_len,
        new_data, g_free);

    if (is_tensor) {
      _info = gst_tensors_info_get_nth_info (&config.info, i);
      gst_tensor_buffer_append_memory (buffer, mem, _info);
    } else {
      gst_buffer_append_memory (buffer, mem);
    }
  }

done:
  if (data_h)
    nns_edge_data_destroy (data_h);

  gst_tensors_config_free (&config);

  if (buffer == NULL) {
    nns_loge ("Failed to get buffer to push to the edgesrc.");
    return GST_FLOW_ERROR;
  }

  *out_buf = buffer;

  return GST_FLOW_OK;
}

/**
 * @brief getter for the 'host' property.
 */
static gchar *
gst_edgesrc_get_dest_host (GstEdgeSrc * self)
{
  return self->dest_host;
}

/**
 * @brief setter for the 'host' property.
 */
static void
gst_edgesrc_set_dest_host (GstEdgeSrc * self, const gchar * dest_host)
{
  g_free (self->dest_host);
  self->dest_host = g_strdup (dest_host);
}

/**
 * @brief getter for the 'port' property.
 */
static guint16
gst_edgesrc_get_dest_port (GstEdgeSrc * self)
{
  return self->dest_port;
}

/**
 * @brief setter for the 'port' property.
 */
static void
gst_edgesrc_set_dest_port (GstEdgeSrc * self, const guint16 dest_port)
{
  self->dest_port = dest_port;
}

/**
 * @brief getter for the 'connect_type' property.
 */
static nns_edge_connect_type_e
gst_edgesrc_get_connect_type (GstEdgeSrc * self)
{
  return self->connect_type;
}

/**
 * @brief setter for the 'connect_type' property.
 */
static void
gst_edgesrc_set_connect_type (GstEdgeSrc * self,
    const nns_edge_connect_type_e connect_type)
{
  self->connect_type = connect_type;
}
