/**
 * @file	unittest_filter_custom.cc
 * @date	11 Apr 2023
 * @brief	Unit test for tensor filter custom-easy plugin
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <stdlib.h>
#include <tensor_filter_custom_easy.h>
#include <unittest_util.h>

/** @brief User data for new_data_cb and custom filter */
typedef struct _cb_data {
  GMutex lock;
  guint filter_received;
  guint sink_received;
} cb_data;

/**
 * @brief In-Code Test Function for custom-easy filter
 */
static int
_custom_easy_filter_dynamic (void *data, const GstTensorsInfo *in_info,
    GstTensorsInfo *out_info, const GstTensorMemory *input, GstTensorMemory *output)
{
  gchar *dim_str;
  guint i;

  cb_data *cbdata = (cb_data *) data;
  if (cbdata == NULL) {
    g_printerr ("%s:%s is called with its third parameter NULL. Cannot proceed.",
        __FILE__, __func__);
    return -EINVAL;
  }

  /* Fill output tensors info */
  gst_tensors_info_init (out_info);
  out_info->info[0].type = _NNS_UINT32;

  /** Protect cbdata->* */
  g_mutex_lock (&cbdata->lock);
  dim_str = g_strdup_printf ("%u:1:1:1", ++(cbdata->filter_received));
  gst_tensor_parse_dimension (dim_str, out_info->info[0].dimension);
  out_info->num_tensors = 1;
  out_info->format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  /* Allocate and fill output memory */
  output[0].size = sizeof (guint) * cbdata->filter_received;
  output[0].data = g_malloc0 (output[0].size);

  for (i = 0; i < cbdata->filter_received; i++) {
    ((guint *) output[0].data)[i] = i;
  }
  g_mutex_unlock (&cbdata->lock);
  g_free (dim_str);
  return 0;
}

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  gsize mem_size, header_size, expected_size;
  GstTensorMetaInfo meta;
  GstMemory *mem;
  GstMapInfo map;
  guint *data;
  guint i;
  cb_data *cbdata = (cb_data *) user_data;

  if (cbdata == NULL) {
    g_printerr ("%s:%s is called with its third parameter NULL. Cannot proceed.",
        __FILE__, __func__);
    return;
  }

  /** Protect cbdata->* */
  g_mutex_lock (&cbdata->lock);

  cbdata->sink_received = cbdata->sink_received + 1;
  expected_size = sizeof (guint) * cbdata->sink_received;

  EXPECT_EQ (1U, gst_buffer_n_memory (buffer));

  mem = gst_buffer_peek_memory (buffer, 0);
  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    g_message ("Failed to map the info buffer.");
    g_mutex_unlock (&cbdata->lock);
    return;
  }

  gst_tensor_meta_info_parse_header (&meta, map.data);
  EXPECT_EQ (_NNS_TENSOR_FORMAT_FLEXIBLE, meta.format);
  EXPECT_EQ (cbdata->sink_received, meta.dimension[0]);

  mem_size = gst_memory_get_sizes (mem, NULL, NULL);
  header_size = gst_tensor_meta_info_get_header_size (&meta);
  EXPECT_EQ (expected_size, mem_size - header_size);

  data = (guint *) (map.data + header_size);
  for (i = 0; i < cbdata->filter_received; i++) {
    EXPECT_EQ (i, data[i]);
  }
  g_mutex_unlock (&cbdata->lock);

  gst_memory_unmap (mem, &map);
}

/**
 * @brief Test custom-easy filter with flexible tensor input/output.
 * @todo Enable the test after development is done.
 */
TEST (tensorFilterCustom, flexibleInvoke_p)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  GstTensorsInfo info_in;
  GstElement *sink_handle;
  int ret;

  cb_data data;
  g_mutex_init (&data.lock);
  data.filter_received = 0;
  data.sink_received = 0;

  gst_tensors_info_init (&info_in);
  info_in.num_tensors = 1U;
  info_in.info[0].name = NULL;
  info_in.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  ret = NNS_custom_easy_dynamic_register (
      "flexbible_filter", _custom_easy_filter_dynamic, &data, &info_in);
  ASSERT_EQ (ret, 0);

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_1 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_2 "
      "join name=j ! other/tensors,format=flexible ! tensor_filter framework=custom-easy invoke-dynamic=TRUE model=flexbible_filter ! other/tensors,format=flexible ! tensor_sink name=sinkx sync=true");

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, &data);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&data.sink_received, 6, TEST_TIMEOUT_LIMIT_MS));
  g_usleep (1000000);

  /** cleanup registered custom_easy filter */
  ret = NNS_custom_easy_unregister ("flexbible_filter");
  ASSERT_EQ (0, ret);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_mutex_clear (&data.lock);
}


/**
 * @brief Test custom-easy filter with static input, flexible output.
 * @todo Enable the test after development is done.
 */
TEST (tensorFilterCustom, staticFlexibleInvoke_p)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  GstTensorsInfo info_in;
  GstElement *sink_handle;
  int ret;

  cb_data data;
  g_mutex_init (&data.lock);
  data.filter_received = 0;
  data.sink_received = 0;

  gst_tensors_info_init (&info_in);
  info_in.num_tensors = 1U;
  info_in.info[0].name = NULL;
  info_in.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  ret = NNS_custom_easy_dynamic_register (
      "flexbible_filter", _custom_easy_filter_dynamic, &data, &info_in);
  ASSERT_EQ (ret, 0);

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=10/1 ! tensor_converter ! j.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240,framerate=10/1 ! tensor_converter ! j.sink_1 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480,framerate=10/1 ! tensor_converter ! j.sink_2 "
      "join name=j ! other/tensors,format=flexible ! tensor_filter framework=custom-easy invoke-dynamic=TRUE model=flexbible_filter ! other/tensors,format=flexible ! tensor_sink name=sinkx sync=true");

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, &data);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&data.sink_received, 6, TEST_TIMEOUT_LIMIT_MS));
  g_usleep (1000000);

  /** cleanup registered custom_easy filter */
  ret = NNS_custom_easy_unregister ("flexbible_filter");
  ASSERT_EQ (0, ret);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_mutex_clear (&data.lock);
}


/**
 * @brief Test dynamic invoke with invalid prop..
 * @todo Enable the test after development is done.
 */
TEST (tensorFilterCustom, flexibleInvokeInvalidProp_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  GstTensorsInfo info_in;
  GstElement *sink_handle;
  int ret;

  cb_data data;
  g_mutex_init (&data.lock);
  data.filter_received = 0;
  data.sink_received = 0;

  gst_tensors_info_init (&info_in);
  info_in.num_tensors = 1U;
  info_in.info[0].name = NULL;
  info_in.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  ret = NNS_custom_easy_dynamic_register (
      "flexbible_filter", _custom_easy_filter_dynamic, &data, &info_in);
  ASSERT_EQ (ret, 0);

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_1 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_2 "
      "join name=j ! other/tensors,format=flexible ! tensor_filter framework=custom-easy invoke-dynamic=FALSE model=flexbible_filter ! other/tensors,format=flexible ! tensor_sink name=sinkx sync=true");

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, &data);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  /** cleanup registered custom_easy filter */
  ret = NNS_custom_easy_unregister ("flexbible_filter");
  ASSERT_EQ (0, ret);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_mutex_clear (&data.lock);
}

/**
 * @brief In-Code Test Function for custom-easy filter
 */
static int
_custom_easy_filter (void *data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  guint i;
  cb_data *cbdata = (cb_data *) data;
  if (cbdata == NULL) {
    g_printerr ("%s:%s is called with its third parameter NULL. Cannot proceed.",
        __FILE__, __func__);
    return -EINVAL;
  }

  /** Protect cbdata->* */
  g_mutex_lock (&cbdata->lock);
  /* Allocate and fill output memory */
  output[0].size = sizeof (guint) * ++(cbdata->filter_received);
  output[0].data = g_malloc0 (output[0].size);

  for (i = 0; i < cbdata->filter_received; i++) {
    ((guint *) output[0].data)[i] = i;
  }
  g_mutex_unlock (&cbdata->lock);

  return 0;
}

/**
 * @brief Test custom-easy statc invoke with flexible tensor input/output.
 * @todo Enable the test after development is done.
 */
TEST (tensorFilterCustom, staticInvoke_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  GstTensorsInfo info_in;
  GstTensorsInfo info_out;
  GstElement *sink_handle;
  int ret;

  cb_data data;
  g_mutex_init (&data.lock);
  data.filter_received = 0;
  data.sink_received = 0;

  gst_tensors_info_init (&info_in);
  info_in.num_tensors = 1U;
  info_in.info[0].name = NULL;
  info_in.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  gst_tensors_info_init (&info_out);
  info_out.num_tensors = 1U;
  info_out.info[0].name = NULL;
  info_out.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  ret = NNS_custom_easy_register (
      "normal_filter", _custom_easy_filter, &data, &info_in, &info_out);
  ASSERT_EQ (ret, 0);

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_1 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_2 "
      "join name=j ! other/tensors,format=flexible ! tensor_filter framework=custom-easy model=normal_filter ! other/tensors,format=flexible ! tensor_sink name=sinkx sync=true");

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, NULL);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  /** cleanup registered custom_easy filter */
  ret = NNS_custom_easy_unregister ("normal_filter");
  ASSERT_EQ (0, ret);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_mutex_clear (&data.lock);
}

/**
 * @brief Test custom-easy filter with flexible tensor input/output without register custom easy model.
 */
TEST (tensorFilterCustom, notRegisterFlexibleInvoke_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  if (root_path == NULL)
    root_path = "..";

  gchar *model_file = g_build_filename (root_path, "build", "tests",
      "nnstreamer_example", "libnnstreamer_customfilter_passthrough.so", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=10/1 ! "
      "tensor_converter ! tensor_filter name=test_filter framework=custom invoke-dynamic=TRUE model=%s ! tensor_sink sync=true",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Test dynamic invoke with invalid param.
 * @todo Enable the test after development is done.
 */
TEST (tensorFilterCustom, dynamicRegisterInvalidParam_n)
{
  GstTensorsInfo info_in;
  int ret;

  gst_tensors_info_init (&info_in);
  info_in.num_tensors = 1U;
  info_in.info[0].name = NULL;
  info_in.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  cb_data data;
  g_mutex_init (&data.lock);
  data.filter_received = 0;
  data.sink_received = 0;

  ret = NNS_custom_easy_dynamic_register (NULL, _custom_easy_filter_dynamic, &data, &info_in);
  EXPECT_NE (0, ret);

  ret = NNS_custom_easy_dynamic_register ("temp_name", NULL, NULL, &info_in);
  EXPECT_NE (0, ret);

  ret = NNS_custom_easy_dynamic_register (
      "temp_name", _custom_easy_filter_dynamic, NULL, NULL);
  EXPECT_NE (0, ret);
  g_mutex_clear (&data.lock);
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
