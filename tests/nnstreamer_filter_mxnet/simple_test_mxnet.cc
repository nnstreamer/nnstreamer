#include "Predictor.hh"
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <mxnet-cpp/io.h>
#include <stdexcept>
#include <string.h>
#include <thread>

#define NUM_INFERENCE_BATCHES (500)

/* GStreamer app data struct */
typedef struct {
  GstElement *pipeline, *app_source, *app_sink;
  guint sourceid; /* To control the GSource */
  GMainLoop *main_loop; /* GLib's Main Loop */

  /* For nnstreamer-filter-mxnet */
  int num_batch;
  std::unique_ptr<MXDataIter> data_iter;
  std::vector<float> *log;
} AppData;

/* Get MXNet Predictor using default parameters
 * The code is from main() in MXNet Predictor example:
 *  https://github.com/apache/incubator-mxnet/blob/1.7.0/cpp-package/example/inference/imagenet_inference.cpp
 */
std::unique_ptr<Predictor>
get_predictor (MXDataIter *data_iter)
{
  std::string model_file_json ("./model/Inception-BN.json");
  std::string model_file_params ("./model/Inception-BN.params");
  Shape input_data_shape (1, 3, 224, 224); // The first 1 is the batch size, which is 1 at a time.
  std::string data_layer_type ("float32");
  bool use_gpu = false;
  bool enable_tensorrt = false;

  return std::unique_ptr<Predictor> (new Predictor (model_file_json, model_file_params,
      input_data_shape, data_layer_type, use_gpu, enable_tensorrt, data_iter));
}

/* Get MXNet RecordIO dataset iterator using default parameters
 * The code is from main() in MXNet Predictor example:
 *  https://github.com/apache/incubator-mxnet/blob/1.7.0/cpp-package/example/inference/imagenet_inference.cpp
 */
std::unique_ptr<MXDataIter>
get_dataset ()
{
  std::string dataset ("./data/val_256_q90.rec");
  std::vector<float> rgb_mean = { 123.68, 116.779, 103.939 };
  std::vector<float> rgb_std = { 1, 1, 1 };
  Shape input_data_shape (1, 3, 224, 224); // The first 1 is the batch size, which is 1 at a time.
  std::string data_layer_type ("float32");
  int seed = 48564309;
  int shuffle_chunk_seed = 3982304;
  int data_nthreads = 60;
  bool use_gpu = false;

  return std::unique_ptr<MXDataIter> (
      Predictor::CreateImageRecordIter (dataset, input_data_shape, data_layer_type,
          rgb_mean, rgb_std, shuffle_chunk_seed, seed, data_nthreads, use_gpu));
}

/* Push a dataset batch (one preprocessed image) to the pipeline */
static gboolean
push_data (AppData *data)
{
  GstFlowReturn ret;
  GstBuffer *buffer;
  GstMapInfo map;

  if (data->num_batch >= NUM_INFERENCE_BATCHES) {
    g_signal_emit_by_name (data->app_source, "end-of-stream", &ret);
    return FALSE;
  }

  /* Next batch */
  data->data_iter->Next ();
  data->num_batch++;

  const mx_uint len = Shape (1, 3, 224, 224).Size ();
  const size_t len_bytes = len * sizeof (mx_float);
  buffer = gst_buffer_new_and_alloc (len_bytes);

  /* Move batch (image) data to the buffer object */
  gst_buffer_map (buffer, &map, GST_MAP_WRITE);
  auto data_batch = data->data_iter->GetDataBatch ();
  data_batch.data.SyncCopyToCPU ((mx_float *)map.data, len);
  NDArray::WaitAll ();
  gst_buffer_unmap (buffer, &map);

  /* Push the buffer into the appsrc */
  g_signal_emit_by_name (data->app_source, "push-buffer", buffer, &ret);

  /* Free the buffer */
  gst_buffer_unref (buffer);

  if (ret != GST_FLOW_OK) {
    return FALSE;
  }
  return TRUE;
}

/* This signal callback triggers when appsrc needs data. Here, we add an idle
 * handler to the mainloop to start pushing data into the appsrc */
static void
start_feed (GstElement *source, guint size, AppData *data)
{
  if (data->sourceid == 0) {
    data->sourceid = g_idle_add ((GSourceFunc)push_data, data);
  }
}

/* This callback triggers when appsrc has enough data and we can stop sending.
 * We remove the idle handler from the mainloop */
static void
stop_feed (GstElement *source, AppData *data)
{
  if (data->sourceid != 0) {
    g_source_remove (data->sourceid);
    data->sourceid = 0;
  }
}

/* Log the result when a new sample (result) is received */
static GstFlowReturn
new_sample (GstElement *sink, AppData *data)
{
  GstSample *sample;

  /* Retrieve the buffer */
  g_signal_emit_by_name (sink, "pull-sample", &sample);
  if (sample) {
    GstMapInfo map;
    GstBuffer *buffer = gst_sample_get_buffer (sample);
    gst_buffer_map (buffer, &map, GST_MAP_READ);

    /* Log result */
    mx_float *pred_data = (mx_float *)map.data;
    data->log->push_back (pred_data[0]);

    /* Free the buffer */
    gst_buffer_unmap (buffer, &map);
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }

  return GST_FLOW_ERROR;
}

/* When the pipeline get EOS, we exit the mainloop. */
static gboolean
on_pipeline_message (GstBus *bus, GstMessage *message, AppData *data)
{
  switch (GST_MESSAGE_TYPE (message)) {
  case GST_MESSAGE_EOS:
    g_main_loop_quit (data->main_loop);
    break;
  case GST_MESSAGE_ERROR: {
    g_print ("Received error\n");

    GError *err = NULL;
    gchar *dbg_info = NULL;

    gst_message_parse_error (message, &err, &dbg_info);
    g_printerr ("ERROR from element %s: %s\n", GST_OBJECT_NAME (message->src), err->message);
    g_printerr ("Debugging info: %s\n", (dbg_info) ? dbg_info : "none");
    g_error_free (err);
    g_free (dbg_info);
  }

    g_main_loop_quit (data->main_loop);
    break;
  case GST_MESSAGE_STATE_CHANGED:
    break;
  default:
    break;
  }
  return TRUE;
}

int
main (int argc, char *argv[])
{
  std::vector<float> reference_result;
  std::vector<float> nnstreamer_result;

  /* Run the reference implementation first and save log */
  {
    std::unique_ptr<MXDataIter> dataset = get_dataset ();
    std::unique_ptr<Predictor> predictor = get_predictor (dataset.get ());
    predictor->LogInferenceResult (reference_result, NUM_INFERENCE_BATCHES);
    std::cout << "Running the reference implementation finished." << std::endl;
  }

  /* Run the NNStreamer implementation */
  {
    AppData data;
    GstBus *bus;
    memset (&data, 0, sizeof (data));

    /* Initialize dataset iterator */
    data.data_iter = get_dataset ();
    data.data_iter->Reset ();
    data.num_batch = 0;
    nnstreamer_result.reserve (NUM_INFERENCE_BATCHES);
    data.log = &nnstreamer_result;

    /* Initialize GStreamer */
    gst_init (&argc, &argv);

    /* Create the pipeline and set elements */
    data.pipeline = gst_parse_launch (" \
      appsrc name=recordio_src ! application/octet-stream \
      ! tensor_converter input-dim=1:3:224:224 input-type=float32 \
      ! tensor_filter \
          framework=mxnet \
          model=model/Inception-BN.json \
          input=1:3:224:224 \
          inputtype=float32 \
          inputname=data \
          output=1 \
          outputtype=float32 \
          outputname=argmax_channel \
          custom=input_rank=4,enable_tensorrt=false \
          accelerator=true:cpu,!npu,!gpu \
      ! appsink name=log_sink",
        NULL);

    bus = gst_element_get_bus (data.pipeline);
    gst_bus_add_watch (bus, (GstBusFunc)on_pipeline_message, &data);
    gst_object_unref (bus);

    data.app_source = gst_bin_get_by_name (GST_BIN (data.pipeline), "recordio_src");
    data.app_sink = gst_bin_get_by_name (GST_BIN (data.pipeline), "log_sink");

    if (!data.pipeline || !data.app_source || !data.app_sink) {
      g_printerr ("Not all elements could be created.\n");
      return -1;
    }

    /* Configure appsrc and appsink */
    g_signal_connect (data.app_source, "need-data", G_CALLBACK (start_feed), &data);
    g_signal_connect (data.app_source, "enough-data", G_CALLBACK (stop_feed), &data);

    g_object_set (data.app_sink, "emit-signals", TRUE, NULL);
    g_signal_connect (data.app_sink, "new-sample", G_CALLBACK (new_sample), &data);

    /* Start playing the pipeline */
    gst_element_set_state (data.pipeline, GST_STATE_PLAYING);

    /* Create a GLib Main Loop and set it to run */
    data.main_loop = g_main_loop_new (NULL, FALSE);
    g_main_loop_run (data.main_loop);

    /* Free resources */
    gst_element_set_state (data.pipeline, GST_STATE_NULL);
    gst_object_unref (data.pipeline);
  }

  /* Compare results */
  if (reference_result != nnstreamer_result) {
    std::cout << "The reference implementation and the NNStreamer implementation does not match!!"
              << std::endl;
    return 1;
  } else {
    std::cout << "Test passed" << std::endl;
  }
  return 0;
}
