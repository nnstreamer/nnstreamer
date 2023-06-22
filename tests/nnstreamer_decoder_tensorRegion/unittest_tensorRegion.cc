/**
 * @file	unittest_tensorRegion.cc
 * @date	20 June 2023
 * @brief	Unit test for tensor_decoder::tensor_region. (testcases to check data conversion or buffer transfer)
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Harsh Jain <hjain24in@gmail.com>
 * @bug		No known bugs.
 */
#include <gtest/gtest.h>
#include <glib/gstdio.h>
#include <gst/check/gstcheck.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_subplugin.h>
#include <string.h>
#include <tensor_common.h>
#include <tensor_meta.h>
#include <unistd.h>

#include "../gst/nnstreamer/elements/gsttensor_sparseutil.h"
#include "../gst/nnstreamer/elements/gsttensor_transform.h"
#include "../unittest_util.h"

#if defined(ENABLE_TENSORFLOW_LITE) || defined(ENABLE_TENSORFLOW2_LITE)
#define TEST_REQUIRE_TFLITE(Case, Name) TEST (Case, Name)
#else
#define TEST_REQUIRE_TFLITE(Case, Name) TEST (Case, DISABLED_##Name)
#endif

/**
 * @brief Call back function for tensor_region to parse outbuf
 * 
 * @param sink The sink element
 * @param user_data User data passed to the callback function
 */
static void new_data_cb(GstElement* sink, const gpointer user_data)
{
  GstSample* sample = nullptr;

  g_signal_emit_by_name(sink, "pull-sample", &sample);

  /** Expected values of cropping info for orange.png */
  guint32 expected_values[] = {58, 62, 219, 211};

  if (sample != nullptr) {
    GstBuffer* outbuf = gst_sample_get_buffer(sample);
    GstMemory* mem = gst_buffer_peek_memory(outbuf, 0);

    if (mem != nullptr) {
      GstMapInfo out_info;

      if (gst_memory_map(mem, &out_info, GST_MAP_READ)) {
        GstTensorMetaInfo map;
        int* data_ptr = nullptr;

        gst_tensor_meta_info_parse_header(&map, out_info.data);

        gsize hsize = gst_tensor_meta_info_get_header_size(&map);
        gsize dsize = gst_tensor_meta_info_get_data_size(&map);
        ASSERT_EQ(map.type, _NNS_UINT32);

        gsize esize = sizeof(guint32);

        ASSERT_EQ(hsize + dsize, out_info.size);
        ASSERT_EQ((dsize % (esize * 4)), 0);

        data_ptr = reinterpret_cast<int*>(out_info.data + hsize);

        for (int i = 0; i < 4; i++) {
          EXPECT_EQ(data_ptr[i], expected_values[i]);
        }

        gst_memory_unmap(mem, &out_info);
      }
    }

    gst_sample_unref(sample);
  }
}

/**
 * @brief Structure to hold information related to TensorRegion.
 */
struct TensorRegion {
  GstElement* pipeline;   /**< The pipeline element */
  GstElement* app_sink;   /**< The app sink element */
  GMainLoop* main_loop;   /**< The main loop */
};


/**
 * @brief Callback function to handle pipeline messages.
 *
 * @param bus The GStreamer bus.
 * @param message The GStreamer message.
 * @param data Pointer to the TensorRegion structure.
 * @return gboolean Returns TRUE to continue receiving messages.
 */
static gboolean on_pipeline_message(GstBus* bus, GstMessage* message, TensorRegion* data)
{
  switch (GST_MESSAGE_TYPE(message)) {
    case GST_MESSAGE_EOS:
      g_main_loop_quit(data->main_loop);
      break;
    case GST_MESSAGE_ERROR: {
      g_print("Received error\n");

      GError* err = NULL;
      gchar* dbg_info = NULL;

      gst_message_parse_error(message, &err, &dbg_info);
      g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(message->src), err->message);
      g_printerr("Debugging info: %s\n", (dbg_info) ? dbg_info : "none");
      g_error_free(err);
      g_free(dbg_info);
    }

    g_main_loop_quit(data->main_loop);
    break;
    case GST_MESSAGE_STATE_CHANGED:
      break;
    default:
      break;
  }

  /** Return FALSE to stop receiving messages after the callback function
  * has handled the current message. */
  return G_SOURCE_CONTINUE;
}


/**
 * @brief Test for tensor_decoder::tensor_region
 */
TEST_REQUIRE_TFLITE (TensorDecoder, TensorRegion)
{
  GstBus* bus;
  const gchar* root_path = g_getenv("NNSTREAMER_SOURCE_ROOT_PATH");
  if (root_path == nullptr)
    root_path = "..";

  const gchar* tensor_0 = g_build_filename(root_path, "tests", "nnstreamer_decoder_tensorRegion", "mobilenet_ssd_tensor.0", nullptr);
  const gchar* tensor_1 = g_build_filename(root_path, "tests", "nnstreamer_decoder_tensorRegion", "mobilenet_ssd_tensor.1", nullptr);
  const gchar* labels_path = g_build_filename(root_path, "tests", "test_models", "labels", "labels.txt", nullptr);
  const gchar* box_priors_path = g_build_filename(root_path, "tests", "nnstreamer_decoder_boundingbox", "box_priors.txt", nullptr);

  ASSERT_TRUE(g_file_test(tensor_0, G_FILE_TEST_EXISTS));
  ASSERT_TRUE(g_file_test(tensor_1, G_FILE_TEST_EXISTS));
  ASSERT_TRUE(g_file_test(labels_path, G_FILE_TEST_EXISTS));
  ASSERT_TRUE(g_file_test(box_priors_path, G_FILE_TEST_EXISTS));

  /** Create the GStreamer pipeline */
  gchar* pipeline_str = g_strdup_printf(
      "multifilesrc name=fs1 location=%s start-index=0 stop-index=1 caps=application/octet-stream ! tensor_converter name=el1 input-dim=4:1:1917:1 input-type=float32 ! mux.sink_0 \
       multifilesrc name=fs2 location=%s start-index=0 stop-index=1 caps=application/octet-stream ! tensor_converter name=el2 input-dim=91:1917:1 input-type=float32 ! mux.sink_1 \
       tensor_mux name=mux ! other/tensors,format=static ! tensor_decoder mode=tensor_region option1=1 option2=%s option3=%s ! appsink name=sinkx ",
      tensor_0, tensor_1, labels_path, box_priors_path);

  GstElement* pipeline = gst_parse_launch(pipeline_str, nullptr);
  g_free(pipeline_str);

  GstElement* app_sink = gst_bin_get_by_name(GST_BIN(pipeline), "sinkx");

  /** Create the TensorRegion structure and assign pipeline and app_sink */
  TensorRegion data;
  data.pipeline = pipeline;
  data.app_sink = app_sink;
  bus = gst_element_get_bus(data.pipeline);
  gst_bus_add_watch(bus, (GstBusFunc)on_pipeline_message, &data);
  gst_object_unref(bus);

  /** Enable signal emission from the app_sink */
  g_object_set(app_sink, "emit-signals", TRUE, NULL);

  /** Connect the new-sample callback to the app_sink */
  g_signal_connect(app_sink, "new-sample", G_CALLBACK(new_data_cb), nullptr);

  /** Start playing the pipeline */
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /** Create a GLib Main Loop and set it to run */
  data.main_loop = g_main_loop_new(NULL, FALSE);
  g_main_loop_run(data.main_loop);

  /** Free resources */
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
}


/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  int ret = -1;
  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    ret = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return ret;
}
