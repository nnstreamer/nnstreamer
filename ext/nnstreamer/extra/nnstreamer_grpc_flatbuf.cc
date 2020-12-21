/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer gRPC/flatbuf support
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    nnstreamer_grpc_flatbuffer.cc
 * @date    26 Nov 2020
 * @brief   nnstreamer gRPC/Flatbuf support
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "nnstreamer_grpc_flatbuf.h"

#include <grpc++/grpc++.h>

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>

#include <thread>

#include <grpcpp/grpcpp.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <gst/base/gstdataqueue.h>

using nnstreamer::flatbuf::TensorService;
using nnstreamer::flatbuf::Tensors;
using nnstreamer::flatbuf::Tensor;
using nnstreamer::flatbuf::Empty;

using nnstreamer::flatbuf::Tensor_type;
using nnstreamer::flatbuf::frame_rate;

using flatbuffers::grpc::Message;
using flatbuffers::grpc::MessageBuilder;

using namespace grpc;

/** @brief constructor */
ServiceImplFlatbuf::ServiceImplFlatbuf (const grpc_config * config):
  NNStreamerRPC (config), client_stub_ (nullptr)
{
}

/** @brief client-to-server streaming: a client sends tensors */
Status
ServiceImplFlatbuf::SendTensors (ServerContext *context,
    ServerReader<Message<Tensors>> *reader,
    Message<Empty> *replay)
{
  return _read_tensors (reader);
}

/** @brief server-to-client streaming: a client receives tensors */
Status
ServiceImplFlatbuf::RecvTensors (ServerContext *context,
    const Message<Empty> *request,
    ServerWriter<Message<Tensors>> *writer)
{
  return _write_tensors (writer);
}

/** @brief read tensors and invoke the registered callback */
template <typename T>
Status ServiceImplFlatbuf::_read_tensors (T reader)
{
  Message<Tensors> msg;

  while (reader->Read (&msg)) {
    GstBuffer *buffer;

    _get_buffer_from_tensors (msg, &buffer);

    if (cb_)
      cb_ (cb_data_, buffer);
  }

  return Status::OK;
}

/** @brief obtain tensors from data queue and send them over gRPC */
template <typename T>
Status ServiceImplFlatbuf::_write_tensors (T writer)
{
  GstDataQueueItem *item;

  /* until flushing */
  while (gst_data_queue_pop (queue_, &item)) {
    Message<Tensors> tensors;

    _get_tensors_from_buffer (GST_BUFFER (item->object), tensors);
    writer->Write (tensors);

    GDestroyNotify destroy = (item->destroy) ? item->destroy : g_free;
    destroy (item);
  }

  return Status::OK;
}

/** @brief gRPC client thread */
void
ServiceImplFlatbuf::_client_thread ()
{
  ClientContext context;

  if (direction_ == GRPC_DIRECTION_TENSORS_TO_BUFFER) {
    Message<Empty> empty;

    /* initiate the RPC call */
    std::unique_ptr< ClientWriter<Message<Tensors>> > writer(
        client_stub_->SendTensors (&context, &empty));

    _write_tensors (writer.get ());

    writer->WritesDone ();
    /**
     * TODO: The below incurs assertion failure but it seems like a bug.
     * Let's check it later with the latest gRPC version.
     *
     * writer->Finish ();
     */
    g_usleep (G_USEC_PER_SEC / 100);
  } else if (direction_ == GRPC_DIRECTION_BUFFER_TO_TENSORS) {
    MessageBuilder builder;

    auto empty_offset = nnstreamer::flatbuf::CreateEmpty (builder);
    builder.Finish (empty_offset);

    /* initiate the RPC call */
    std::unique_ptr< ClientReader<Message<Tensors>> > reader(
        client_stub_->RecvTensors (&context, builder.ReleaseMessage <Empty> ()));

    _read_tensors (reader.get ());

    reader->Finish ();
  } else {
    g_assert (0); /* internal logic error */
  }
}

/** @brief start gRPC server handling flatbuf */
gboolean
ServiceImplFlatbuf::start_server (std::string address)
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

/** @brief start gRPC client handling flatbuf */
gboolean
ServiceImplFlatbuf::start_client (std::string address)
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
ServiceImplFlatbuf::_get_buffer_from_tensors (Message<Tensors> &msg,
    GstBuffer **buffer)
{
  const Tensors *tensors = msg.GetRoot ();
  guint num_tensor = tensors->num_tensor ();
  GstMemory *memory;

  *buffer = gst_buffer_new ();

  for (guint i = 0; i < num_tensor; i++) {
    const Tensor * tensor = tensors->tensor ()->Get (i);
    const void * data = tensor->data ()->data ();
    gsize size = VectorLength (tensor->data ());
    gpointer new_data = g_memdup (data, size);

    memory = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
        new_data, size, 0, size, new_data, g_free);
    gst_buffer_append_memory (*buffer, memory);
  }
}

/** @brief convert buffer to tensors */
void
ServiceImplFlatbuf::_get_tensors_from_buffer (GstBuffer *buffer,
    Message<Tensors> &msg)
{
  MessageBuilder builder;

  flatbuffers::Offset<flatbuffers::Vector<uint32_t>> tensor_dim;
  flatbuffers::Offset<flatbuffers::String> tensor_name;
  flatbuffers::Offset<flatbuffers::Vector<unsigned char>> tensor_data;
  flatbuffers::Offset<Tensor> tensor;
  flatbuffers::Offset<Tensors> tensors;
  std::vector<flatbuffers::Offset<Tensor>> tensor_vector;
  Tensor_type tensor_type;

  unsigned int num_tensors = config_->info.num_tensors;
  frame_rate fr = frame_rate (config_->rate_n, config_->rate_d);

  GstMapInfo map;
  gsize data_ptr = 0;

  gst_buffer_map (buffer, &map, GST_MAP_READ);

  for (guint i = 0; i < num_tensors; i++) {
    const GstTensorInfo * info = &config_->info.info[i];
    gsize tsize = gst_tensor_info_get_size (info);

    tensor_dim = builder.CreateVector (info->dimension, NNS_TENSOR_RANK_LIMIT);
    tensor_name = builder.CreateString ("Anonymous");
    tensor_type = (Tensor_type) info->type;
    tensor_data = builder.CreateVector<unsigned char> (map.data + data_ptr, tsize);

    if (data_ptr + tsize > map.size) {
      ml_logw ("Setting invalid tensor data");
      break;
    }
    data_ptr += tsize;

    tensor = CreateTensor (builder, tensor_name, tensor_type, tensor_dim, tensor_data);
    tensor_vector.push_back (tensor);
  }

  tensors = CreateTensors (builder, num_tensors, &fr, builder.CreateVector (tensor_vector));

  builder.Finish (tensors);
  msg = builder.ReleaseMessage<Tensors>();

  gst_buffer_unmap (buffer, &map);
}

/** @brief create gRPC/Flatbuf instance */
extern "C" NNStreamerRPC *
create_instance (const grpc_config * config)
{
  return new ServiceImplFlatbuf (config);
}
