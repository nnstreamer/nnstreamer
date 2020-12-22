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

using namespace grpc;

/** @brief constructor */
ServiceImplProtobuf::ServiceImplProtobuf (const grpc_config * config):
  NNStreamerRPC (config), client_stub_ (nullptr)
{
}

/** @brief parse tensors and deliver the buffer via callback */
void
ServiceImplProtobuf::parse_tensors (Tensors &tensors)
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
ServiceImplProtobuf::fill_tensors (Tensors &tensors)
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
Status ServiceImplProtobuf::_read_tensors (T reader)
{
  while (1) {
    Tensors tensors;

    if (!reader->Read (&tensors))
      break;

    parse_tensors (tensors);
  }

  return Status::OK;
}

/** @brief obtain tensors from data queue and send them over gRPC */
template <typename T>
Status ServiceImplProtobuf::_write_tensors (T writer)
{
  while (1) {
    Tensors tensors;

    /* until flushing */
    if (!fill_tensors (tensors))
      break;

    writer->Write (tensors);
  }

  return Status::OK;
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

  tensors.set_num_tensor (config_->info.num_tensors);

  fr = tensors.mutable_fr ();
  fr->set_rate_n (config_->rate_n);
  fr->set_rate_d (config_->rate_d);

  if (!gst_buffer_map (buffer, &map, GST_MAP_READ)) {
    ml_loge ("Unable to map the buffer\n");
    return;
  }

  for (guint i = 0; i < config_->info.num_tensors; i++) {
    nnstreamer::protobuf::Tensor *tensor = tensors.add_tensor ();
    const GstTensorInfo * info = &config_->info.info[i];
    gsize tsize = gst_tensor_info_get_size (info);

    if (data_ptr + tsize > map.size) {
      ml_logw ("Setting invalid tensor data");
      break;
    }

    /* set tensor info */
    tensor->set_name ("Anonymous");
    tensor->set_type ((Tensor::Tensor_type) info->type);

    for (guint j = 0; j < NNS_TENSOR_RANK_LIMIT; j++)
      tensor->add_dimension (info->dimension[j]);

    tensor->set_data (map.data + data_ptr, tsize);
    data_ptr += tsize;
  }

  gst_buffer_unmap (buffer, &map);
}

/** @brief Constructor of SyncServiceImplProtobuf */
SyncServiceImplProtobuf::SyncServiceImplProtobuf (const grpc_config * config)
  : ServiceImplProtobuf (config)
{
}

/** @brief client-to-server streaming: a client sends tensors */
Status
SyncServiceImplProtobuf::SendTensors (ServerContext *context,
    ServerReader<Tensors> *reader, Empty *reply)
{
  return _read_tensors (reader);
}

/** @brief server-to-client streaming: a client receives tensors */
Status
SyncServiceImplProtobuf::RecvTensors (ServerContext *context,
    const Empty *request, ServerWriter<Tensors> *writer)
{
  return _write_tensors (writer);
}

/** @brief start gRPC server handling protobuf */
gboolean
SyncServiceImplProtobuf::start_server (std::string address)
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

/** @brief start gRPC client handling protobuf */
gboolean
SyncServiceImplProtobuf::start_client (std::string address)
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
SyncServiceImplProtobuf::_client_thread ()
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

/** @brief Constructor of AsyncServiceImplProtobuf */
AsyncServiceImplProtobuf::AsyncServiceImplProtobuf (const grpc_config * config)
  : ServiceImplProtobuf (config), last_call_ (nullptr)
{
}

/** @brief Destructor of AsyncServiceImplProtobuf */
AsyncServiceImplProtobuf::~AsyncServiceImplProtobuf ()
{
  if (last_call_)
    delete last_call_;
}

/** @brief start gRPC server handling protobuf */
gboolean
AsyncServiceImplProtobuf::start_server (std::string address)
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

/** @brief start gRPC client handling protobuf */
gboolean
AsyncServiceImplProtobuf::start_client (std::string address)
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
    AsyncCallDataServer (AsyncServiceImplProtobuf *service, ServerCompletionQueue *cq)
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
          reader_.reset (new ServerAsyncReader<Empty, Tensors> (&ctx_));
          service_->RequestSendTensors (&ctx_, reader_.get (), cq_, cq_, this);
        } else {
          writer_.reset (new ServerAsyncWriter<Tensors> (&ctx_));
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
          Tensors tensors;
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
        if (reader_.get () != nullptr)
          reader_->Finish (rpc_empty_, Status::OK, this);
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

    std::unique_ptr<ServerAsyncWriter<Tensors>> writer_;
    std::unique_ptr<ServerAsyncReader<Empty, Tensors>> reader_;
};

/** @brief Internal derived class for client */
class AsyncCallDataClient : public AsyncCallData {
  public:
    /** @brief Constructor of AsyncCallDataClient */
    AsyncCallDataClient (AsyncServiceImplProtobuf *service, TensorService::Stub * stub,
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
          reader_ = stub_->AsyncRecvTensors (&ctx_, rpc_empty_, cq_, this);
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
          Tensors tensors;
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

    std::unique_ptr<ClientAsyncWriter<Tensors>> writer_;
    std::unique_ptr<ClientAsyncReader<Tensors>> reader_;
};

/** @brief gRPC client thread */
void
AsyncServiceImplProtobuf::_server_thread ()
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
AsyncServiceImplProtobuf::_client_thread ()
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

/** @brief create gRPC/Protobuf instance */
extern "C" void *
create_instance (const grpc_config * config)
{
  if (config->is_blocking)
    return new SyncServiceImplProtobuf (config);
  else
    return new AsyncServiceImplProtobuf (config);
}
