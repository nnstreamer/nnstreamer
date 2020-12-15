/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer gRPC support
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    nnstreamer_grpc_common.h
 * @date    21 Oct 2020
 * @brief   Common header for NNStreamer gRPC support
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __NNS_GRPC_COMMON_H__
#define __NNS_GRPC_COMMON_H__

#include "nnstreamer_grpc.h"

#include <gst/base/gstdataqueue.h>
#include <grpcpp/grpcpp.h>

#include <cstring>
#include <string>
#include <thread>

namespace grpc {

/**
 * @brief NNStreamer RPC service
 */
class NNStreamerRPC {
  public:
    static NNStreamerRPC * createInstance (grpc_idl idl, gboolean server,
        const gchar *host, const gint port);

    NNStreamerRPC (gboolean is_server, const gchar *host, const gint port);
    virtual ~NNStreamerRPC ();

    gboolean start ();
    void stop ();
    gboolean send (GstBuffer *buffer);

    /** @brief set callback */
    void setCallback (grpc_cb cb, void *cb_data) {
      cb_ = cb;
      cb_data_ = cb_data;
    }

    /** @brief set tensor config */
    void setConfig (GstTensorsConfig *config) {
      memcpy (&config_, config, sizeof (*config));
    }

    /** @brief set grpc direction */
    void setDirection (grpc_direction direction) {
      direction_ = direction;
    }

    /** @brief get gRPC listening port (server only) */
    int getListeningPort () {
      if (is_server_)
        return port_;
      else
        return -EINVAL;
    }

    /** @brief set library module handle */
    void setModuleHandle (void * handle) {
      if (handle_ == NULL)
        handle_ = handle;
    }

    /** @brief get library module handle */
    void *getModuleHandle () {
      return handle_;
    }

  protected:
    gboolean is_server_;
    const gchar *host_;
    gint port_;

    grpc_direction direction_;

    grpc_cb cb_;
    void * cb_data_;

    std::unique_ptr<Server> server_worker_;
    std::thread client_worker_;

    GstDataQueue *queue_;
    GstTensorsConfig config_;

    void * handle_;

  private:
    /** @brief start gRPC server */
    virtual gboolean start_server (std::string address) { return FALSE; }
    /** @brief start gRPC client */
    virtual gboolean start_client (std::string address) { return FALSE; }

    gboolean _start_server ();
    gboolean _start_client ();

    static gboolean _data_queue_check_full_cb (GstDataQueue * queue,
        guint visible, guint bytes, guint64 time, gpointer checkdata);
    static void _data_queue_item_free (GstDataQueueItem * item);
};

}; // namespace grpc

#endif /* __NNS_GRPC_COMMON_H__ */
