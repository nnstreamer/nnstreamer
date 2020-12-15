/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer gRPC/protobuf support
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    nnstreamer_grpc_protobuf.cc
 * @date    21 Oct 2020
 * @brief   gRPC/Protobuf wrappers for nnstreamer
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "nnstreamer_grpc_protobuf.h"

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>

#include <thread>

#include <grpcpp/grpcpp.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <gst/base/gstdataqueue.h>

using nnstreamer::protobuf::TensorService;
using nnstreamer::protobuf::Tensors;
using nnstreamer::protobuf::Tensor;

using google::protobuf::Empty;

using namespace grpc;

/** @brief constructor */
ServiceImplProtobuf::ServiceImplProtobuf (gboolean is_server,
    const gchar *host, const gint port):
  NNStreamerRPC (is_server, host, port), client_stub_ (nullptr)
{
}

/** @brief client-to-server streaming: a client sends tensors */
Status
ServiceImplProtobuf::SendTensors (ServerContext *context,
    ServerReader<Tensors> *reader, Empty *reply)
{
  return _read_tensors (reader);
}

/** @brief server-to-client streaming: a client receives tensors */
Status
ServiceImplProtobuf::RecvTensors (ServerContext *context,
    const Empty *request, ServerWriter<Tensors> *writer)
{
  return _write_tensors (writer);
}

/** @brief read tensors and invoke the registered callback */
template <typename T>
Status ServiceImplProtobuf::_read_tensors (T reader)
{
  Tensors tensors;

  while (reader->Read (&tensors)) {
    GstBuffer *buffer;

    _get_buffer_from_tensors (tensors, &buffer);

    if (cb_)
      cb_ (cb_data_, buffer);
  }

  return Status::OK;
}

/** @brief obtain tensors from data queue and send them over gRPC */
template <typename T>
Status ServiceImplProtobuf::_write_tensors (T writer)
{
  GstDataQueueItem *item;

  /* until flushing */
  while (gst_data_queue_pop (queue_, &item)) {
    Tensors tensors;

    _get_tensors_from_buffer (GST_BUFFER (item->object), tensors);
    writer->Write (tensors);

    GDestroyNotify destroy = (item->destroy) ? item->destroy : g_free;
    destroy (item);
  }

  return Status::OK;
}

/** @brief gRPC client thread */
void
ServiceImplProtobuf::_client_thread ()
{
  ClientContext context;
  Empty empty;

  if (direction_ == GRPC_DIRECTION_TENSORS_TO_BUFFER) {
    /* initiate the RPC call */
    std::unique_ptr< ClientWriter<Tensors> > writer(
        client_stub_->SendTensors (&context, &empty));

    _write_tensors (writer.get ());

    writer->WritesDone ();
    writer->Finish ();
  } else if (direction_ == GRPC_DIRECTION_BUFFER_TO_TENSORS) {
    Tensors tensors;

    /* initiate the RPC call */
    std::unique_ptr< ClientReader<Tensors> > reader(
        client_stub_->RecvTensors (&context, empty));

    _read_tensors (reader.get ());

    reader->Finish ();
  } else {
    g_assert (0); /* internal logic error */
  }
}

/** @brief start gRPC server handling protobuf */
gboolean
ServiceImplProtobuf::start_server (std::string address)
{
  /* listen on the given address without any authentication mechanism */
  ServerBuilder builder;
  builder.AddListeningPort (address, grpc::InsecureServerCredentials(), &port_);
  builder.RegisterService (this);

  /* start the server */
  server_worker_ = builder.BuildAndStart ();
  if (server_worker_.get () == nullptr)
    return FALSE;

  return TRUE;
}

/** @brief start gRPC client handling protobuf */
gboolean
ServiceImplProtobuf::start_client (std::string address)
{
  /* create a gRPC channel */
  std::shared_ptr<Channel> channel = grpc::CreateChannel(
      address, grpc::InsecureChannelCredentials());

  /* connect the server */
  client_stub_ = TensorService::NewStub (channel);
  if (client_stub_.get () == nullptr)
    return FALSE;

  client_worker_ = std::thread ([this] { this->_client_thread (); });

  return TRUE;
}

/** @brief convert tensors to buffer */
void
ServiceImplProtobuf::_get_buffer_from_tensors (Tensors &tensors,
    GstBuffer **buffer)
{
  guint num_tensor = tensors.num_tensor ();
  GstMemory *memory;

  *buffer = gst_buffer_new ();

  for (guint i = 0; i < num_tensor; i++) {
    const Tensor * tensor = &tensors.tensor (i);
    const void * data = tensor->data ().c_str ();
    gsize size = tensor->data ().length ();
    gpointer new_data = g_memdup (data, size);

    memory = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
        new_data, size, 0, size, new_data, g_free);
    gst_buffer_append_memory (*buffer, memory);
  }
}

/** @brief convert buffer to tensors */
void
ServiceImplProtobuf::_get_tensors_from_buffer (GstBuffer *buffer,
    Tensors &tensors)
{
  Tensors::frame_rate *fr;
  GstMapInfo map;
  gsize data_ptr = 0;

  tensors.set_num_tensor (config_.info.num_tensors);

  fr = tensors.mutable_fr ();
  fr->set_rate_n (config_.rate_n);
  fr->set_rate_d (config_.rate_d);

  gst_buffer_map (buffer, &map, GST_MAP_READ);

  for (guint i = 0; i < config_.info.num_tensors; i++) {
    nnstreamer::protobuf::Tensor *tensor = tensors.add_tensor ();
    const GstTensorInfo * info = &config_.info.info[i];
    gsize tsize = gst_tensor_info_get_size (info);

    /* set tensor info */
    tensor->set_name ("Anonymous");
    tensor->set_type ((Tensor::Tensor_type) info->type);

    for (guint j = 0; j < NNS_TENSOR_RANK_LIMIT; j++)
      tensor->add_dimension (info->dimension[j]);

    if (data_ptr + tsize > map.size) {
      ml_logw ("Setting invalid tensor data");
      break;
    }

    tensor->set_data (map.data + data_ptr, tsize);
    data_ptr += tsize;
  }

  gst_buffer_unmap (buffer, &map);
}

/** @brief create gRPC/Protobuf instance */
extern "C" void *
create_instance (gboolean server, const gchar *host, const gint port)
{
  return new ServiceImplProtobuf (server, host, port);
}
