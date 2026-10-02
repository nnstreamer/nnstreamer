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
#include "nnstreamer_conf.h"

#include <gmodule.h>

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>

#include <grpcpp/health_check_service_interface.h>

#include <chrono>

static constexpr const char *NNS_GRPC_PROTOBUF_NAME = "libnnstreamer_grpc_protobuf";
static constexpr const char *NNS_GRPC_FLATBUF_NAME = "libnnstreamer_grpc_flatbuf";
static constexpr const char *NNS_GRPC_CREATE_INSTANCE = "create_instance";
/** @brief How long stop () waits for a peer that takes none of the queued buffers */
static constexpr gint64 NNS_GRPC_DRAIN_STALL_USEC = G_USEC_PER_SEC;

using namespace grpc;

/** @brief create new instance of NNStreamerRPC */
NNStreamerRPC *
NNStreamerRPC::createInstance (const grpc_config *config)
{
  gchar *name = NULL;

  if (config->idl == GRPC_IDL_PROTOBUF)
    name = g_strdup_printf ("%s%s", NNS_GRPC_PROTOBUF_NAME, NNSTREAMER_SO_FILE_EXTENSION);
  else if (config->idl == GRPC_IDL_FLATBUF)
    name = g_strdup_printf ("%s%s", NNS_GRPC_FLATBUF_NAME, NNSTREAMER_SO_FILE_EXTENSION);

  if (name == NULL) {
    ml_loge ("Unsupported IDL detected: %d\n", config->idl);
    return NULL;
  }

  GModule *module = g_module_open (name, G_MODULE_BIND_LAZY);
  if (!module) {
    ml_loge ("Error opening %s\n", name);
    g_free (name);
    return NULL;
  }

  using function_ptr = void *(*) (const grpc_config *config);
  function_ptr create_instance;

  if (!g_module_symbol (module, NNS_GRPC_CREATE_INSTANCE, (gpointer *) &create_instance)) {
    ml_loge ("Error loading create_instance: %s\n", g_module_error ());
    g_free (name);
    g_module_close (module);
    return NULL;
  }

  NNStreamerRPC *instance = (NNStreamerRPC *) create_instance (config);
  if (!instance) {
    ml_loge ("Error creating an instance\n");
    g_free (name);
    g_module_close (module);
    return NULL;
  }

  g_free (name);
  instance->setModuleHandle (module);
  return instance;
}

/** @brief constructor of NNStreamerRPC */
NNStreamerRPC::NNStreamerRPC (const grpc_config *config)
    : host_ (config->host), port_ (config->port), is_server_ (config->is_server),
      is_blocking_ (config->is_blocking), direction_ (config->dir),
      cb_ (config->cb), cb_data_ (config->cb_data), config_ (config->config),
      server_instance_ (nullptr), handle_ (nullptr), stop_ (false),
      shutting_down_ (false), client_context_ (nullptr), client_stopping_ (false)
{
  queue_ = gst_data_queue_new (_data_queue_check_full_cb, NULL, NULL, NULL);
}

/** @brief destructor of NNStreamerRPC */
NNStreamerRPC::~NNStreamerRPC ()
{
  g_clear_pointer (&queue_, gst_object_unref);
}

/** @brief start gRPC server */
gboolean
NNStreamerRPC::start ()
{
  if (direction_ == GRPC_DIRECTION_NONE)
    return FALSE;

  if (is_server_)
    return _start_server ();
  else
    return _start_client ();
}

/** @brief stop the thread */
void
NNStreamerRPC::stop ()
{
  if (stop_)
    return;

  /* notify to the worker */
  stop_ = true;

  if (queue_) {
    GstDataQueueSize level;
    guint last = G_MAXUINT;
    gint64 progress = g_get_monotonic_time ();

    /* wait for the peer to take the queued buffers, as long as it keeps taking them */
    while (!gst_data_queue_is_empty (queue_)) {
      gst_data_queue_get_level (queue_, &level);
      if (level.visible < last) {
        last = level.visible;
        progress = g_get_monotonic_time ();
      } else if (g_get_monotonic_time () - progress >= NNS_GRPC_DRAIN_STALL_USEC) {
        ml_logw ("The gRPC peer took no buffer for a second; dropping %u queued.",
            level.visible);
        break;
      }
      g_usleep (G_USEC_PER_SEC / 100);
    }

    gst_data_queue_set_flushing (queue_, TRUE);
  }

  _cancel_client ();

  if (is_server_) {
    shutting_down_ = true;

    /* cancel the calls still writing to a peer that does not read */
    if (server_instance_.get ())
      server_instance_->Shutdown (std::chrono::system_clock::now ()
                                  + std::chrono::microseconds (NNS_GRPC_DRAIN_STALL_USEC));

    if (completion_queue_.get ())
      completion_queue_->Shutdown ();
  }

  if (worker_.joinable ())
    worker_.join ();
}

/** @brief register the context of a blocking client call */
void
NNStreamerRPC::_set_client_context (ClientContext *context)
{
  std::lock_guard<std::mutex> lock (client_lock_);

  if (context && client_stopping_)
    context->TryCancel ();

  client_context_ = context;
  if (!context)
    client_cond_.notify_all ();
}

/** @brief cancel the call of a blocking client that does not end in time */
void
NNStreamerRPC::_cancel_client ()
{
  std::unique_lock<std::mutex> lock (client_lock_);

  client_stopping_ = true;
  if (!client_context_)
    return;

  /* a writer may still be handing the last buffers to the peer; a reader has nothing to finish */
  if (direction_ == GRPC_DIRECTION_TENSORS_TO_BUFFER
      && client_cond_.wait_for (lock, std::chrono::microseconds (NNS_GRPC_DRAIN_STALL_USEC),
          [this] { return client_context_ == nullptr; }))
    return;

  client_context_->TryCancel ();
}

/** @brief send buffer holding tensors */
gboolean
NNStreamerRPC::send (GstBuffer *buffer)
{
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
NNStreamerRPC::_start_server ()
{
  std::string address (host_);

  address += ":" + std::to_string (port_);

  grpc::EnableDefaultHealthCheckService (true);

  return start_server (address);
}

/** @brief start client service */
gboolean
NNStreamerRPC::_start_client ()
{
  std::string address (host_);

  address += ":" + std::to_string (port_);

  return start_client (address);
}

/** @brief wait until config_ is negotiated before a received message is parsed */
gboolean
NNStreamerRPC::_wait_configured ()
{
  /* only a receiving element negotiates what it receives */
  if (direction_ != GRPC_DIRECTION_BUFFER_TO_TENSORS)
    return TRUE;

  while (!g_atomic_int_get (&configured_)) {
    /* an async server call held here keeps Server::Shutdown () waiting until it is freed */
    if (g_atomic_int_get (&stop_))
      return FALSE;
    g_usleep (G_USEC_PER_SEC / 100);
  }

  return TRUE;
}

/** @brief check the number of tensors a received message declares */
gboolean
NNStreamerRPC::_check_tensor_count (gint64 declared, gint64 carried)
{
  if (declared > carried) {
    ml_loge ("Failed to get tensors, the message declares %" G_GINT64_FORMAT
             " tensors but carries %" G_GINT64_FORMAT ".",
        declared, carried);
    return FALSE;
  }

  if (declared != config_->info.num_tensors) {
    ml_loge ("Failed to get tensors, the message declares %" G_GINT64_FORMAT
             " tensors but the caps have %u.",
        declared, config_->info.num_tensors);
    return FALSE;
  }

  return TRUE;
}

/** @brief check the data size of a received tensor */
gboolean
NNStreamerRPC::_check_tensor_size (guint index, gsize size)
{
  GstTensorInfo *info = gst_tensors_info_get_nth_info (&config_->info, index);
  gsize expected = gst_tensor_info_get_size (info);

  if (size != expected) {
    ml_loge ("Failed to get tensors, tensor %u has %zu bytes but the caps need %zu.",
        index, size, expected);
    return FALSE;
  }

  return TRUE;
}

/** @brief private method to check full  */
gboolean
NNStreamerRPC::_data_queue_check_full_cb (GstDataQueue *queue, guint visible,
    guint bytes, guint64 time, gpointer checkdata)
{
  /* no full */
  return FALSE;
}

/** @brief private method to free a data item */
void
NNStreamerRPC::_data_queue_item_free (GstDataQueueItem *item)
{
  if (item->object)
    gst_buffer_unref (GST_BUFFER (item->object));
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
grpc_new (const grpc_config *config)
{
  g_return_val_if_fail (config != NULL, NULL);

  NNStreamerRPC *self = NNStreamerRPC::createInstance (config);

  return static_cast<void *> (self);
}

/**
 * @brief gRPC C++ wrapper to destroy the class instance
 */
void
grpc_destroy (void *instance)
{
  g_return_if_fail (instance != NULL);

  NNStreamerRPC *self = static_cast<NNStreamerRPC *> (instance);
  void *handle = self->getModuleHandle ();

  delete self;

  if (handle)
    g_module_close ((GModule *) handle);
}

/**
 * @brief gRPC C++ wrapper to start gRPC service
 */
gboolean
grpc_start (void *instance)
{
  g_return_val_if_fail (instance != NULL, FALSE);

  NNStreamerRPC *self = static_cast<NNStreamerRPC *> (instance);

  return self->start ();
}

/**
 * @brief gRPC C++ wrapper to stop service
 */
void
grpc_stop (void *instance)
{
  g_return_if_fail (instance != NULL);

  grpc::NNStreamerRPC *self = static_cast<grpc::NNStreamerRPC *> (instance);

  self->stop ();
}

/**
 * @brief gRPC C++ wrapper to send messages
 */
gboolean
grpc_send (void *instance, GstBuffer *buffer)
{
  g_return_val_if_fail (instance != NULL, FALSE);

  grpc::NNStreamerRPC *self = static_cast<grpc::NNStreamerRPC *> (instance);

  return self->send (buffer);
}

/**
 * @brief get gRPC listening port of the server instance
 */
int
grpc_get_listening_port (void *instance)
{
  g_return_val_if_fail (instance != NULL, -EINVAL);

  NNStreamerRPC *self = static_cast<NNStreamerRPC *> (instance);

  return self->getListeningPort ();
}

/**
 * @brief tell the gRPC instance that the tensors config is negotiated
 */
void
grpc_set_configured (void *instance)
{
  g_return_if_fail (instance != NULL);

  NNStreamerRPC *self = static_cast<NNStreamerRPC *> (instance);

  self->setConfigured ();
}

#define silent_debug(...)                   \
  do {                                      \
    if (*silent) {                          \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    }                                       \
  } while (0)

/**
 * @brief check the validity of hostname string
 */
gboolean
_check_hostname (gchar *str)
{
  if (g_strcmp0 (str, "localhost") == 0 || g_hostname_is_ip_address (str))
    return TRUE;

  return FALSE;
}

/**
 * @brief set-prop common for both grpc elements
 */
void
grpc_common_set_property (GObject *self, gboolean *silent, grpc_private *grpc,
    guint prop_id, const GValue *value, GParamSpec *pspec)
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
        const gchar *idl_str = g_value_get_string (value);

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
        gchar *host;

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
grpc_common_get_property (GObject *self, gboolean silent, guint out,
    grpc_private *grpc, guint prop_id, GValue *value, GParamSpec *pspec)
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
