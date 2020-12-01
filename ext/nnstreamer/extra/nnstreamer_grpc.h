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
 * @brief wrapper for gRPC C++ codes
 */
#define grpc_new(self)                _grpc_new(self->idl, self->server, self->host, self->port)
#define grpc_destroy(self)            _grpc_destroy(self->priv)
#define grpc_set_callback(self, cb)   _grpc_set_callback(self->priv, cb, self)
#define grpc_set_config(self, config) _grpc_set_config(self->priv, config)
#define grpc_start(self, direction)   _grpc_start(self->priv, direction)
#define grpc_stop(self)               _grpc_stop(self->priv)
#define grpc_send(self, buffer)       _grpc_send(self->priv, buffer)
#define grpc_get_listening_port(self) _grpc_get_listening_port(self->priv)

#ifdef __cplusplus
extern "C" {
#endif

grpc_idl grpc_get_idl (const gchar *idl_str);

void * _grpc_new (grpc_idl idl, gboolean server, const gchar *host, const gint port);
void _grpc_destroy (void *priv);
void _grpc_set_callback (void *priv, grpc_cb cb, void *data);
void _grpc_set_config (void *priv, GstTensorsConfig *config);
gboolean _grpc_start (void *priv, grpc_direction direction);
void _grpc_stop (void *priv);
gboolean _grpc_send (void *priv, GstBuffer *buffer);
int _grpc_get_listening_port (void *priv);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_GRPC_H__ */
