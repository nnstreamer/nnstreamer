/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer gRPC support
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    nnstreamer_grpc.h
 * @date    21 Oct 2020
 * @brief   Wrapper header for NNStreamer gRPC support
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __NNS_GRPC_H__
#define __NNS_GRPC_H__

#include <gst/gst.h>
#include <glib.h>
#include <tensor_typedef.h>

/**
 * @brief function pointer for gRPC message callback
 */
typedef void (*grpc_cb)(void *, void *);

/**
 * @brief enum for gRPC service's Interface Definition Language (IDL)
 */
typedef enum {
  GRPC_IDL_NONE = 0,
  GRPC_IDL_PROTOBUF,
  GRPC_IDL_FLATBUF
} grpc_idl;

/**
 * @brief enum for gRPC service's message streaming direction
 */
typedef enum {
  GRPC_DIRECTION_NONE = 0,
  GRPC_DIRECTION_TENSORS_TO_BUFFER, /* from tensors to protobuf/flatbuf */
  GRPC_DIRECTION_BUFFER_TO_TENSORS  /* from protobuf/flatbuf to tensors */
} grpc_direction;

/**
 * @brief structure for grpc configuration
 */
typedef struct {
  grpc_idl idl;
  grpc_direction dir;

  gchar * host;
  gint port;

  gboolean is_server;
  gboolean is_blocking;

  grpc_cb cb;
  void *cb_data;

  GstTensorsConfig *config;
} grpc_config;

/** @brief gRPC private data */
typedef struct {
  grpc_config config;
  void * instance;
} grpc_private;

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

/**
 * @brief C++ wrappers for gRPC per-IDL codes
 */
#ifdef __cplusplus
extern "C" {
#endif

/** @brief Get the grpc_idl enum matching an IDL name such as "protobuf" */
grpc_idl grpc_get_idl (const gchar *idl_str);

/** @brief Create a gRPC service instance for the given configuration */
void * grpc_new (const grpc_config * config);
/** @brief Destroy a gRPC instance created by grpc_new () */
void grpc_destroy (void * instance);

/** @brief Start the gRPC server or client service of the instance */
gboolean grpc_start (void * instance);
/** @brief Stop the gRPC service and join its worker thread */
void grpc_stop (void * instance);

/** @brief Queue a buffer holding tensors to be sent over gRPC */
gboolean grpc_send (void * instance, GstBuffer * buffer);
/** @brief Get the server's listening port, or -EINVAL for a client instance */
int grpc_get_listening_port (void * instance);

/** @brief Check whether the string is "localhost" or an IP address */
gboolean _check_hostname (gchar * str);
/** @brief Handle set-property shared by the gRPC source and sink elements */
void grpc_common_set_property (GObject * self, gboolean * silent, grpc_private * grpc, guint prop_id, const GValue * value, GParamSpec * pspec);
/** @brief Handle get-property shared by the gRPC source and sink elements */
void grpc_common_get_property (GObject * self, gboolean silent, guint out, grpc_private * grpc, guint prop_id, GValue * value, GParamSpec * pspec);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_GRPC_H__ */
