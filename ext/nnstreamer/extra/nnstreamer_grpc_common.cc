/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer gRPC support
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    nnstreamer_grpc_common.cc
 * @date    21 Oct 2020
 * @brief   gRPC wrappers for nnstreamer
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "nnstreamer_grpc_common.h"

#include <gmodule.h>

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>

#include <grpcpp/health_check_service_interface.h>

#define NNS_GRPC_PROTOBUF_NAME   "libnnstreamer_grpc_protobuf.so"
#define NNS_GRPC_FLATBUF_NAME    "libnnstreamer_grpc_flatbuf.so"
#define NNS_GRPC_CREATE_INSTANCE "create_instance"

using namespace grpc;

/** @brief create new instance of NNStreamerRPC */
NNStreamerRPC *
NNStreamerRPC::createInstance (const grpc_config * config)
{
  const gchar * name = NULL;

  if (config->idl == GRPC_IDL_PROTOBUF)
    name = NNS_GRPC_PROTOBUF_NAME;
  else if (config->idl == GRPC_IDL_FLATBUF)
    name = NNS_GRPC_FLATBUF_NAME;

  if (name == NULL) {
    ml_loge ("Unsupported IDL detected: %d\n", config->idl);
    return NULL;
  }

  GModule * module = g_module_open (name, G_MODULE_BIND_LAZY);
  if (!module) {
    ml_loge ("Error opening %s\n", name);
    return NULL;
  }

  using function_ptr = void * (*)(const grpc_config * config);
  function_ptr create_instance;

  if (!g_module_symbol (module, NNS_GRPC_CREATE_INSTANCE,
        (gpointer *) &create_instance)) {
    ml_loge ("Error loading create_instance: %s\n", g_module_error ());
    g_module_close (module);
    return NULL;
  }

  NNStreamerRPC * instance = (NNStreamerRPC *) create_instance (config);
  if (!instance) {
    ml_loge ("Error creating an instance\n");
    g_module_close (module);
    return NULL;
  }

  instance->setModuleHandle (module);
  return instance;
}

/** @brief constructor of NNStreamerRPC */
NNStreamerRPC::NNStreamerRPC (const grpc_config * config):
  host_ (config->host), port_ (config->port),
  is_server_ (config->is_server), is_blocking_ (config->is_blocking),
  direction_ (config->dir), cb_ (config->cb), cb_data_ (config->cb_data),
  config_ (config->config), server_instance_ (nullptr), handle_ (nullptr),
  stop_ (false)
{
  queue_ = gst_data_queue_new (_data_queue_check_full_cb,
      NULL, NULL, NULL);
}

/** @brief destructor of NNStreamerRPC */
NNStreamerRPC::~NNStreamerRPC () {
  g_clear_pointer (&queue_, gst_object_unref);
}

/** @brief start gRPC server */
gboolean
NNStreamerRPC::start () {
  if (direction_ == GRPC_DIRECTION_NONE)
    return FALSE;

  if (is_server_)
    return _start_server ();
  else
    return _start_client ();
}

/** @brief stop the thread */
void
NNStreamerRPC::stop () {
  if (stop_)
    return;

  /* notify to the worker */
  stop_ = true;

  if (queue_) {
    /* wait until the queue's flushed */
    while (!gst_data_queue_is_empty (queue_))
      g_usleep (G_USEC_PER_SEC / 100);

    gst_data_queue_set_flushing (queue_, TRUE);
  }

  if (is_server_) {
    if (server_instance_.get ())
      server_instance_->Shutdown ();

    if (completion_queue_.get ())
      completion_queue_->Shutdown ();
  }

  if (worker_.joinable ())
    worker_.join ();
}

/** @brief send buffer holding tensors */
gboolean
NNStreamerRPC::send (GstBuffer *buffer) {
  GstDataQueueItem *item;

  buffer = gst_buffer_ref (buffer);

  item = g_new0 (GstDataQueueItem, 1);
  item->object = GST_MINI_OBJECT (buffer);
  item->size = gst_buffer_get_size (buffer);
  item->visible = TRUE;
  item->destroy = (GDestroyNotify) _data_queue_item_free;

  if (!gst_data_queue_push (queue_, item)) {
    item->destroy (item);
    return FALSE;
  }

  return TRUE;
}

/** @brief start server service */
gboolean
NNStreamerRPC::_start_server () {
  std::string address (host_);

  address += ":" + std::to_string (port_);

  grpc::EnableDefaultHealthCheckService (true);

  return start_server (address);
}

/** @brief start client service */
gboolean
NNStreamerRPC::_start_client () {
  std::string address (host_);

  address += ":" + std::to_string (port_);

  return start_client (address);
}

/** @brief private method to check full  */
gboolean
NNStreamerRPC::_data_queue_check_full_cb (GstDataQueue * queue,
    guint visible, guint bytes, guint64 time, gpointer checkdata)
{
  /* no full */
  return FALSE;
}

/** @brief private method to free a data item */
void
NNStreamerRPC::_data_queue_item_free (GstDataQueueItem * item) {
  if (item->object)
    gst_mini_object_unref (item->object);
  g_free (item);
}

/**
 * @brief get gRPC IDL enum from a given string
 */
grpc_idl
grpc_get_idl (const gchar *idl_str)
{
  if (g_ascii_strcasecmp (idl_str, "protobuf") == 0)
    return GRPC_IDL_PROTOBUF;
  else if (g_ascii_strcasecmp (idl_str, "flatbuf") == 0)
    return GRPC_IDL_FLATBUF;
  else
    return GRPC_IDL_NONE;
}

/**
 * @brief gRPC C++ wrapper to create the class instance
 */
void *
grpc_new (const grpc_config * config)
{
  g_return_val_if_fail (config != NULL, NULL);

  NNStreamerRPC * self = NNStreamerRPC::createInstance (config);

  return static_cast <void *> (self);
}

/**
 * @brief gRPC C++ wrapper to destroy the class instance
 */
void
grpc_destroy (void *instance)
{
  g_return_if_fail (instance != NULL);

  NNStreamerRPC * self = static_cast<NNStreamerRPC *> (instance);
  void *handle = self->getModuleHandle ();

  delete self;

  if (handle)
    g_module_close ((GModule *) handle);
}

/**
 * @brief gRPC C++ wrapper to start gRPC service
 */
gboolean
grpc_start (void * instance)
{
  g_return_val_if_fail (instance != NULL, FALSE);

  NNStreamerRPC * self = static_cast<NNStreamerRPC *> (instance);

  return self->start ();
}

/**
 * @brief gRPC C++ wrapper to stop service
 */
void
grpc_stop (void * instance)
{
  g_return_if_fail (instance != NULL);

  grpc::NNStreamerRPC * self = static_cast<grpc::NNStreamerRPC *> (instance);

  self->stop ();
}

/**
 * @brief gRPC C++ wrapper to send messages
 */
gboolean
grpc_send (void * instance, GstBuffer *buffer)
{
  g_return_val_if_fail (instance != NULL, FALSE);

  grpc::NNStreamerRPC * self = static_cast<grpc::NNStreamerRPC *> (instance);

  return self->send (buffer);
}

/**
 * @brief get gRPC listening port of the server instance
 */
int
grpc_get_listening_port (void * instance)
{
  g_return_val_if_fail (instance != NULL, -EINVAL);

  NNStreamerRPC * self = static_cast<NNStreamerRPC *> (instance);

  return self->getListeningPort ();
}

#define silent_debug(...) do { \
    if (* silent) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

/**
 * @brief check the validity of hostname string
 */
gboolean
_check_hostname (gchar * str)
{
  if (g_strcmp0 (str, "localhost") == 0 ||
      g_hostname_is_ip_address (str))
    return TRUE;

  return FALSE;
}

/**
 * @brief set-prop common for both grpc elements
 */
void
grpc_common_set_property (GObject * self, gboolean * silent,
    grpc_private * grpc, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  switch (prop_id) {
    case PROP_SILENT:
      *silent = g_value_get_boolean (value);
      silent_debug ("Set silent = %d", *silent);
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
      G_OBJECT_WARN_INVALID_PROPERTY_ID (self, prop_id, pspec);
      break;
  }
}

/**
 * @brief get-prop common for both grpc elements
 */
void
grpc_common_get_property (GObject * self, gboolean silent, guint out,
    grpc_private * grpc, guint prop_id, GValue * value, GParamSpec * pspec)
{
  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, silent);
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
      g_value_set_uint (value, out);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (self, prop_id, pspec);
      break;
  }
}
