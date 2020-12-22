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

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>

#include <thread>

#include <grpcpp/grpcpp.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <gst/base/gstdataqueue.h>

using nnstreamer::flatbuf::Tensor_type;
using nnstreamer::flatbuf::frame_rate;

using flatbuffers::grpc::MessageBuilder;

using namespace grpc;

/** @brief Constructor of ServiceImplFlatbuf */
ServiceImplFlatbuf::ServiceImplFlatbuf (const grpc_config * config)
  : NNStreamerRPC (config), client_stub_ (nullptr)
{
}

/** @brief parse tensors and deliver the buffer via callback */
void
ServiceImplFlatbuf::parse_tensors (Message<Tensors> &tensors)
{
  GstBuffer *buffer;

  _get_buffer_from_tensors (tensors, &buffer);

  if (cb_)
    cb_ (cb_data_, buffer);
  else
    gst_buffer_unref (buffer);
}

/** @brief fill tensors from the buffer */
gboolean
ServiceImplFlatbuf::fill_tensors (Message<Tensors> &tensors)
{
  GstDataQueueItem *item;

  if (!gst_data_queue_pop (queue_, &item))
    return FALSE;

  _get_tensors_from_buffer (GST_BUFFER (item->object), tensors);

  GDestroyNotify destroy = (item->destroy) ? item->destroy : g_free;
  destroy (item);

  return TRUE;
}

/** @brief read tensors and invoke the registered callback */
template <typename T>
Status ServiceImplFlatbuf::_read_tensors (T reader)
{
  while (1) {
    Message<Tensors> tensors;

    if (!reader->Read (&tensors))
      break;

    parse_tensors (tensors);
  }

  return Status::OK;
}

/** @brief obtain tensors from data queue and send them over gRPC */
template <typename T>
Status ServiceImplFlatbuf::_write_tensors (T writer)
{
  while (1) {
    Message<Tensors> tensors;

    /* until flushing */
    if (!fill_tensors (tensors))
      break;

    writer->Write (tensors);
  }

  return Status::OK;
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

  if (!gst_buffer_map (buffer, &map, GST_MAP_READ)) {
    ml_loge ("Unable to map the buffer\n");
    return;
  }

  for (guint i = 0; i < num_tensors; i++) {
    const GstTensorInfo * info = &config_->info.info[i];
    gsize tsize = gst_tensor_info_get_size (info);

    if (data_ptr + tsize > map.size) {
      ml_logw ("Setting invalid tensor data");
      break;
    }

    tensor_dim = builder.CreateVector (info->dimension, NNS_TENSOR_RANK_LIMIT);
    tensor_name = builder.CreateString ("Anonymous");
    tensor_type = (Tensor_type) info->type;
    tensor_data = builder.CreateVector<unsigned char> (map.data + data_ptr, tsize);

    data_ptr += tsize;

    tensor = CreateTensor (builder, tensor_name, tensor_type, tensor_dim, tensor_data);
    tensor_vector.push_back (tensor);
  }

  tensors = CreateTensors (builder, num_tensors, &fr, builder.CreateVector (tensor_vector));

  builder.Finish (tensors);
  msg = builder.ReleaseMessage<Tensors>();

  gst_buffer_unmap (buffer, &map);
}

/** @brief Constructor of SyncServiceImplFlatbuf */
SyncServiceImplFlatbuf::SyncServiceImplFlatbuf (const grpc_config * config)
  : ServiceImplFlatbuf (config)
{
}

/** @brief client-to-server streaming: a client sends tensors */
Status
SyncServiceImplFlatbuf::SendTensors (ServerContext *context,
    ServerReader<Message<Tensors>> *reader,
    Message<Empty> *replay)
{
  return _read_tensors (reader);
}

/** @brief server-to-client streaming: a client receives tensors */
Status
SyncServiceImplFlatbuf::RecvTensors (ServerContext *context,
    const Message<Empty> *request,
    ServerWriter<Message<Tensors>> *writer)
{
  return _write_tensors (writer);
}

/** @brief start gRPC server handling flatbuf */
gboolean
SyncServiceImplFlatbuf::start_server (std::string address)
{
  /* listen on the given address without any authentication mechanism */
  ServerBuilder builder;
  builder.AddListeningPort (address, grpc::InsecureServerCredentials(), &port_);
  builder.RegisterService (this);

  /* start the server */
  server_instance_ = builder.BuildAndStart ();
  if (server_instance_.get () == nullptr)
    return FALSE;

  return TRUE;
}

/** @brief start gRPC client handling flatbuf */
gboolean
SyncServiceImplFlatbuf::start_client (std::string address)
{
  /* create a gRPC channel */
  std::shared_ptr<Channel> channel = grpc::CreateChannel(
      address, grpc::InsecureChannelCredentials());

  /* connect the server */
  client_stub_ = TensorService::NewStub (channel);
  if (client_stub_.get () == nullptr)
    return FALSE;

  worker_ = std::thread ([this] { this->_client_thread (); });

  return TRUE;
}

/** @brief gRPC client thread */
void
SyncServiceImplFlatbuf::_client_thread ()
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

/** @brief Constructor of AsyncServiceImplFlatbuf */
AsyncServiceImplFlatbuf::AsyncServiceImplFlatbuf (const grpc_config * config)
  : ServiceImplFlatbuf (config), last_call_ (nullptr)
{
}

/** @brief Destructor of AsyncServiceImplFlatbuf */
AsyncServiceImplFlatbuf::~AsyncServiceImplFlatbuf ()
{
  if (last_call_)
    delete last_call_;
}


/** @brief start gRPC server handling flatbuf */
gboolean
AsyncServiceImplFlatbuf::start_server (std::string address)
{
  /* listen on the given address without any authentication mechanism */
  ServerBuilder builder;
  builder.AddListeningPort (address, grpc::InsecureServerCredentials(), &port_);
  builder.RegisterService (this);

  /* need to manually handle the completion queue */
  completion_queue_ = builder.AddCompletionQueue ();

  /* start the server */
  server_instance_ = builder.BuildAndStart ();
  if (server_instance_.get () == nullptr)
    return FALSE;

  worker_ = std::thread ([this] { this->_server_thread (); });

  return TRUE;
}

/** @brief start gRPC client handling flatbuf */
gboolean
AsyncServiceImplFlatbuf::start_client (std::string address)
{
  /* create a gRPC channel */
  std::shared_ptr<Channel> channel = grpc::CreateChannel(
      address, grpc::InsecureChannelCredentials());

  /* connect the server */
  client_stub_ = TensorService::NewStub (channel);
  if (client_stub_.get () == nullptr)
    return FALSE;

  worker_ = std::thread ([this] { this->_client_thread (); });

  return TRUE;
}

/** @brief Internal derived class for server */
class AsyncCallDataServer : public AsyncCallData {
  public:
    /** @brief Constructor of AsyncCallDataServer */
    AsyncCallDataServer (AsyncServiceImplFlatbuf *service, ServerCompletionQueue *cq)
      : AsyncCallData (service), cq_ (cq), writer_ (nullptr), reader_ (nullptr)
    {
      RunState ();
    }

    /** @brief implemented RunState () of AsyncCallDataServer */
    void RunState (bool ok = true) override
    {
      if (state_ == PROCESS && !ok) {
        if (count_ != 0) {
          if (reader_.get () != nullptr)
            service_->parse_tensors (rpc_tensors_);
          state_ = FINISH;
        } else {
          return;
        }
      }

      if (state_ == CREATE) {
        if (service_->getDirection () == GRPC_DIRECTION_BUFFER_TO_TENSORS) {
          reader_.reset (new ServerAsyncReader<Message<Empty>, Message<Tensors>> (&ctx_));
          service_->RequestSendTensors (&ctx_, reader_.get (), cq_, cq_, this);
        } else {
          writer_.reset (new ServerAsyncWriter<Message<Tensors>> (&ctx_));
          service_->RequestRecvTensors (&ctx_, &rpc_empty_, writer_.get (), cq_, cq_, this);
        }
        state_ = PROCESS;
      } else if (state_ == PROCESS) {
        if (count_ == 0) {
          /* spawn a new instance to serve new clients */
          service_->set_last_call (new AsyncCallDataServer (service_, cq_));
        }

        if (reader_.get () != nullptr) {
          if (count_ != 0)
            service_->parse_tensors (rpc_tensors_);
          reader_->Read (&rpc_tensors_, this);
          /* can't read tensors yet. use the next turn */
          count_++;
        } else if (writer_.get () != nullptr) {
          Message<Tensors> tensors;
          if (service_->fill_tensors (tensors)) {
            writer_->Write (tensors, this);
            count_++;
          } else {
            Status status;
            writer_->Finish (status, this);
            state_ = DESTROY;
          }
        }
      } else if (state_ == FINISH) {
        if (reader_.get () != nullptr) {
          MessageBuilder builder;

          auto empty_offset = nnstreamer::flatbuf::CreateEmpty (builder);
          builder.Finish (empty_offset);

          reader_->Finish (builder.ReleaseMessage <Empty> (), Status::OK, this);
        }
        if (writer_.get () != nullptr) {
          Status status;
          writer_->Finish (status, this);
        }
        state_ = DESTROY;
      } else {
        delete this;
      }
    }

  private:
    ServerCompletionQueue *cq_;
    ServerContext ctx_;

    std::unique_ptr<ServerAsyncWriter<Message<Tensors>>> writer_;
    std::unique_ptr<ServerAsyncReader<Message<Empty>, Message<Tensors>>> reader_;
};

/** @brief Internal derived class for client */
class AsyncCallDataClient : public AsyncCallData {
  public:
    /** @brief Constructor of AsyncCallDataClient */
    AsyncCallDataClient (AsyncServiceImplFlatbuf *service, TensorService::Stub * stub,
        CompletionQueue *cq)
      : AsyncCallData (service), stub_ (stub), cq_ (cq), writer_ (nullptr), reader_ (nullptr)
    {
      RunState ();
    }

    /** @brief implemented RunState () of AsyncCallDataClient */
    void RunState (bool ok = true) override
    {
      if (state_ == PROCESS && !ok) {
        if (count_ != 0) {
          if (reader_.get () != nullptr)
            service_->parse_tensors (rpc_tensors_);
          state_ = FINISH;
        } else {
          return;
        }
      }

      if (state_ == CREATE) {
        if (service_->getDirection () == GRPC_DIRECTION_BUFFER_TO_TENSORS) {
          MessageBuilder builder;

          auto empty_offset = nnstreamer::flatbuf::CreateEmpty (builder);
          builder.Finish (empty_offset);

          reader_ = stub_->AsyncRecvTensors (&ctx_,
              builder.ReleaseMessage <Empty> (), cq_, this);
        } else {
          writer_ = stub_->AsyncSendTensors (&ctx_, &rpc_empty_, cq_, this);
        }
        state_ = PROCESS;
      } else if (state_ == PROCESS) {
        if (reader_.get () != nullptr) {
          if (count_ != 0)
            service_->parse_tensors (rpc_tensors_);
          reader_->Read (&rpc_tensors_, this);
          /* can't read tensors yet. use the next turn */
          count_++;
        } else if (writer_.get () != nullptr) {
          Message<Tensors> tensors;
          if (service_->fill_tensors (tensors)) {
            writer_->Write (tensors, this);
            count_++;
          } else {
            writer_->WritesDone (this);
            state_ = FINISH;
          }
        }
      } else if (state_ == FINISH) {
        Status status;

        if (reader_.get () != nullptr)
          reader_->Finish (&status, this);
        if (writer_.get () != nullptr)
          writer_->Finish (&status, this);

        delete this;
      }
    }

  private:
    TensorService::Stub * stub_;
    CompletionQueue * cq_;
    ClientContext ctx_;

    std::unique_ptr<ClientAsyncWriter<Message<Tensors>>> writer_;
    std::unique_ptr<ClientAsyncReader<Message<Tensors>>> reader_;
};

/** @brief gRPC client thread */
void
AsyncServiceImplFlatbuf::_server_thread ()
{
  /* spawn a new instance to server new clients */
  set_last_call (new AsyncCallDataServer (this, completion_queue_.get ()));

  while (1) {
    void *tag;
    bool ok;

    /* 10 msec deadline to wait the next event */
    gpr_timespec deadline =
      gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC),
          gpr_time_from_millis(10, GPR_TIMESPAN));

    switch (completion_queue_->AsyncNext (&tag, &ok, deadline)) {
      case CompletionQueue::GOT_EVENT:
        static_cast<AsyncCallDataServer *>(tag)->RunState(ok);
        break;
      case CompletionQueue::SHUTDOWN:
        return;
      default:
        break;
    }
  }
}

/** @brief gRPC client thread */
void
AsyncServiceImplFlatbuf::_client_thread ()
{
  CompletionQueue cq;

  /* spawn a new instance to serve new clients */
  new AsyncCallDataClient (this, client_stub_.get (), &cq);

  /* until the stop is called */
  while (!stop_) {
    void *tag;
    bool ok;

    /* 10 msec deadline to wait the next event */
    gpr_timespec deadline =
      gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC),
          gpr_time_from_millis(10, GPR_TIMESPAN));

    switch (cq.AsyncNext (&tag, &ok, deadline)) {
      case CompletionQueue::GOT_EVENT:
        static_cast<AsyncCallDataClient *>(tag)->RunState(ok);
        if (ok == false)
          return;
        break;
      default:
        break;
    }
  }
}

/** @brief create gRPC/Flatbuf instance */
extern "C" NNStreamerRPC *
create_instance (const grpc_config * config)
{
  if (config->is_blocking)
    return new SyncServiceImplFlatbuf (config);
  else
    return new AsyncServiceImplFlatbuf (config);
}
