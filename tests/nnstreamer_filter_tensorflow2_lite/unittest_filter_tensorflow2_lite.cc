/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_tensorflow2_lite.cc
 * @date    4 Nov 2021
 * @brief   Unit test for tensorflow2-lite tensor filter sub-plugin
 * @author  Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <nnstreamer_util.h>
#include <unittest_util.h>
#include "nnstreamer_plugin_api.h"
#include "nnstreamer_plugin_api_util.h"

/**
 * @brief internal function to get model file path
 */
static gboolean
_GetModelFilePath (gchar **model_file, int option)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  std::string model_name;

  switch (option) {
    case 0:
      model_name = "mobilenet_v2_1.0_224_quant.tflite";
      break;
    case 1:
      model_name = "mobilenet_v2_1.0_224.tflite";
      break;
    case 2:
      model_name = "simple_32_in_32_out.tflite";
      break;
    default:
      break;
  }

  *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", model_name.c_str (), NULL);

  g_free (root_path);

  return g_file_test (*model_file, G_FILE_TEST_EXISTS);
}

/**
 * @brief internal function to get the orange.png
 */
static gboolean
_GetOrangePngFilePath (gchar **input_file)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  std::string input_file_name = "orange.png";

  *input_file = g_build_filename (
      root_path, "tests", "test_models", "data", input_file_name.c_str (), NULL);

  g_free (root_path);

  return g_file_test (*input_file, G_FILE_TEST_EXISTS);
}

/**
 * @brief Signal to validate the result in tensor_sink
 */
static void
check_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  UNUSED (element);

  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);

  gint is_float = (gint) * ((guint8 *) user_data);
  guint idx, max_idx = -1;

  if (is_float == 0) {
    guint8 *output = (guint8 *) info_res.data;
    guint8 max_value = 0;

    for (idx = 0; idx < info_res.size; ++idx) {
      if (output[idx] > max_value) {
        max_value = output[idx];
        max_idx = idx;
      }
    }
  } else if (is_float == 1) {
    gfloat *output = (gfloat *) info_res.data;
    gfloat max_value = G_MINFLOAT;

    for (idx = 0; idx < (info_res.size / sizeof (gfloat)); ++idx) {
      if (output[idx] > max_value) {
        max_value = output[idx];
        max_idx = idx;
      }
    }
  } else {
    ASSERT_TRUE (1 == 0);
  }

  EXPECT_EQ (max_idx, 951U);

  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);
}

/**
 * @brief Negative case to launch gst pipeline: wrong dimension
 */
TEST (nnstreamerFilterTensorFlow2Lite, launch0_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=10 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=42,height=42,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow2-lite model=\"%s\" latency=1 ! tensor_sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Negative case to launch gst pipeline: wrong data type
 */
TEST (nnstreamerFilterTensorFlow2Lite, launch1_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=10 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow2-lite model=\"%s\" latency=1 ! tensor_sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Positive case to launch gst pipeline
 */
TEST (nnstreamerFilterTensorFlow2Lite, quantModelResult)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file, *input_file;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  ASSERT_TRUE (_GetOrangePngFilePath (&input_file));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("filesrc location=\"%s\" ! pngdec ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow2-lite model=\"%s\" ! tensor_sink name=sink",
      input_file, model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sink");
  ASSERT_TRUE (sink_handle != nullptr);

  guint8 *is_float = (guint8 *) g_malloc0 (1);
  *is_float = 0;
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, is_float);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
  g_free (input_file);
  g_free (is_float);
}

/**
 * @brief Positive case to launch gst pipeline
 */
TEST (nnstreamerFilterTensorFlow2Lite, floatModelResult)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file, *input_file;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));
  ASSERT_TRUE (_GetOrangePngFilePath (&input_file));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("filesrc location=\"%s\" ! pngdec ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=tensorflow2-lite model=\"%s\" ! tensor_sink name=sink",
      input_file, model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sink");
  ASSERT_TRUE (sink_handle != nullptr);

  guint8 *is_float = (guint8 *) g_malloc0 (1);
  *is_float = 1;
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, is_float);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
  g_free (input_file);
  g_free (is_float);
}

/**
 * @brief Positive case to launch gst pipeline
 */
TEST (nnstreamerFilterTensorFlow2Lite, floatModelXNNPACKResult)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file, *input_file;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));
  ASSERT_TRUE (_GetOrangePngFilePath (&input_file));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("filesrc location=\"%s\" ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=20/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=tensorflow2-lite model=\"%s\" custom=Delegate:XNNPACK,NumThreads:4 ! tensor_sink name=sink",
      input_file, model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sink");
  ASSERT_TRUE (sink_handle != nullptr);

  guint8 *is_float = (guint8 *) g_malloc0 (1);
  *is_float = 1;
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, is_float);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);
  g_usleep (1000 * 1000 * 5); // wait for 5 seconds to check all output is valid

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
  g_free (input_file);
  g_free (is_float);
}

/**
 * @brief Signal to validate the result in tensor_sink of 32 input/output model.
 */
static void
check_output_many (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  UNUSED (element);

  guint *data_received = (guint *) user_data;
  (*data_received)++;

  for (guint i = 0; i < 32; i++) {
    mem_res = gst_tensor_buffer_get_nth_memory (buffer, i);
    mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
    ASSERT_TRUE (mapped);
    gfloat *output = (gfloat *) info_res.data;
    EXPECT_EQ (17.f, *output);
    gst_memory_unmap (mem_res, &info_res);
    gst_memory_unref (mem_res);
  }
}

/**
 * @brief Check result of tflite model with 32 input/output tensors.
 */
TEST (nnstreamerFilterTensorFlow2Lite, manyInOutModel)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 2));

  /* make 32 "t. ! queue ! mux.sink_## " */
  gchar *tee_queue_mux = g_strdup ("");
  for (int i = 0; i < 32; i++) {
    gchar *aux = g_strdup (tee_queue_mux);
    g_free (tee_queue_mux);
    tee_queue_mux = g_strdup_printf ("%s t. ! queue ! mux.sink_%d ", aux, i);
    g_free (aux);
  }

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc pattern=2 num-buffers=10 is-live=true ! "
      "videoscale ! videoconvert ! video/x-raw,format=GRAY8,width=1,height=1,framerate=30/1 ! "
      "tensor_converter ! tensor_transform mode=typecast option=float32 ! tee name=t "
      "%s"
      "tensor_mux name=mux ! other/tensors,format=static,num_tensors=32 ! "
      "tensor_filter framework=tensorflow2-lite model=\"%s\" ! tensor_sink name=sinkx",
      tee_queue_mux, model_file);

  g_free (tee_queue_mux);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  ASSERT_TRUE (sink_handle != nullptr);

  guint data_received = 0U;
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output_many, &data_received);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);
  g_usleep (1000 * 1000 * 5); // wait for 5 seconds to check all output is valid

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (10U, data_received);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Test for suspend mode.
 */
TEST (nnstreamerFilterTensorFlow2Lite, suspend)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("appsrc name=srcx ! application/octet-stream ! tensor_converter input-dim=3:224:224 input-type=uint8 ! tensor_filter suspend=2000 framework=tensorflow2-lite model=\"%s\" ! tensor_sink name=sink async=false",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *src_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
  ASSERT_TRUE (src_handle != nullptr);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  GstBuffer *buf = gst_buffer_new ();
  GstMemory *mem = gst_allocator_alloc (NULL, 3 * 224 * 224, NULL);
  gst_buffer_append_memory (buf, mem);

  buf = gst_buffer_ref (buf);
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (src_handle), buf), GST_FLOW_OK);

  /** Wait for unloading the framework. */
  g_usleep (5000000);

  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (src_handle), buf), GST_FLOW_OK);
  g_usleep (1000000);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (src_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Main gtest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
