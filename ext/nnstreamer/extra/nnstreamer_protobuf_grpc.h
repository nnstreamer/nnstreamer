/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer gRPC/protobuf support for tensor src/sink
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    nnstreamer_protobuf_grpc.h
 * @date    21 Oct 2020
 * @brief   gRPC/Protobuf wrappers for nnstreamer
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __NNS_PROTOBUF_GRPC_UTIL_H__
#define __NNS_PROTOBUF_GRPC_UTIL_H__

#include "nnstreamer_protobuf.h"

/**
 * @brief function pointer for gRPC message callback
 */
typedef void (*grpc_cb)(void *, void *);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief wrapper for gRPC C++ codes
 */
#define grpc_new(self)                _grpc_new(self->server, self->host, self->port)
#define grpc_destroy(self)            _grpc_destroy(self->priv)
#define grpc_set_callback(self, cb)   _grpc_set_callback(self->priv, cb, self)
#define grpc_set_config(self, config) _grpc_set_config(self->priv, config)
#define grpc_start(self)              _grpc_start(self->priv)
#define grpc_stop(self)               _grpc_stop(self->priv)
#define grpc_send(self, buffer)       _grpc_send(self->priv, buffer)

void * _grpc_new (gboolean server, const gchar *host, const gint port);
void _grpc_destroy (void *priv);
void _grpc_set_callback (void *priv, grpc_cb cb, void *data);
void _grpc_set_config (void *priv, GstTensorsConfig *config);
gboolean _grpc_start (void *priv);
void _grpc_stop (void *priv);
gboolean _grpc_send (void *priv, GstBuffer *buffer);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_PROTOBUF_GRPC_UTIL_H__ */
