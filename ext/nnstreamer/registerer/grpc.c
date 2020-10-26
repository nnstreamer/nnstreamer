/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * nnstreamer registerer for gRPC plugin
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */

/**
 * @file    grpc.c
 * @date    22 Oct 2020
 * @brief   Registers nnstreamer extension plugin for gRPC
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>

#include <tensor_source/tensor_src_grpc.h>

#define NNSTREAMER_GRPC_INIT(plugin,name,type) \
  do { \
    if (!gst_element_register (plugin, "tensor_" # name, GST_RANK_NONE, GST_TYPE_TENSOR_ ## type)) { \
      GST_ERROR ("Failed to register nnstreamer plugin : tensor_" # name); \
      return FALSE; \
    } \
  } while (0)

/**
 * @brief Function to initialize all nnstreamer elements
 */
static gboolean
gst_nnstreamer_grpc_init (GstPlugin * plugin)
{
  NNSTREAMER_GRPC_INIT (plugin, src_grpc, SRC_GRPC);
  return TRUE;
}

#ifndef PACKAGE
#define PACKAGE "nnstreamer_grpc"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nnstreamer_grpc,
    "nnstreamer gRPC framework extension",
    gst_nnstreamer_grpc_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnstreamer/nnstreamer");
