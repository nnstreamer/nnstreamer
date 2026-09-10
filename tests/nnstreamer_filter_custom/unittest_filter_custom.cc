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
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <stdlib.h>
#include <string.h>
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

  g_atomic_int_inc (&cbdata->sink_received);
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
      "flexible_filter", _custom_easy_filter_dynamic, &data, &info_in);
  ASSERT_EQ (ret, 0);

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_1 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_2 "
      "join name=j ! other/tensors,format=flexible ! tensor_filter framework=custom-easy invoke-dynamic=TRUE model=flexible_filter ! other/tensors,format=flexible ! tensor_sink name=sinkx sync=true");

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, &data);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&data.sink_received, 6, TEST_TIMEOUT_LIMIT_MS));
  g_usleep (1000000);

  /** cleanup registered custom_easy filter */
  ret = NNS_custom_easy_unregister ("flexible_filter");
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
      "flexible_filter", _custom_easy_filter_dynamic, &data, &info_in);
  ASSERT_EQ (ret, 0);

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=10/1 ! tensor_converter ! j.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240,framerate=10/1 ! tensor_converter ! j.sink_1 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480,framerate=10/1 ! tensor_converter ! j.sink_2 "
      "join name=j ! other/tensors,format=flexible ! tensor_filter framework=custom-easy invoke-dynamic=TRUE model=flexible_filter ! other/tensors,format=flexible ! tensor_sink name=sinkx sync=true");

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, &data);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&data.sink_received, 6, TEST_TIMEOUT_LIMIT_MS));
  g_usleep (1000000);

  /** cleanup registered custom_easy filter */
  ret = NNS_custom_easy_unregister ("flexible_filter");
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
      "flexible_filter", _custom_easy_filter_dynamic, &data, &info_in);
  ASSERT_EQ (ret, 0);

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_1 "
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480,framerate=10/1 ! tensor_converter ! other/tensors,format=flexible ! j.sink_2 "
      "join name=j ! other/tensors,format=flexible ! tensor_filter framework=custom-easy invoke-dynamic=FALSE model=flexible_filter ! other/tensors,format=flexible ! tensor_sink name=sinkx sync=true");

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, &data);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  /** cleanup registered custom_easy filter */
  ret = NNS_custom_easy_unregister ("flexible_filter");
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
 * @brief Test custom-easy filter without the model property.
 */
TEST (tensorFilterCustom, easyOpenWithoutModel_n)
{
  GstElement *gstpipe;
  GError *err = NULL;
  const gchar *pipeline
      = "videotestsrc num-buffers=3 ! videoconvert ! "
        "video/x-raw,width=160,height=120,format=RGB,framerate=10/1 ! tensor_converter ! "
        "tensor_filter framework=custom-easy ! tensor_sink sync=true";

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
      -ESTRPIPE);

  gst_object_unref (gstpipe);
}

/**
 * @brief Test custom-easy filter with an empty model name.
 * @details "," splits into two empty names, unlike "" which yields no name at all.
 */
TEST (tensorFilterCustom, easyOpenEmptyModel_n)
{
  GstElement *gstpipe;
  GError *err = NULL;
  const gchar *pipeline
      = "videotestsrc num-buffers=3 ! videoconvert ! "
        "video/x-raw,width=160,height=120,format=RGB,framerate=10/1 ! tensor_converter ! "
        "tensor_filter framework=custom-easy model=\",\" ! tensor_sink sync=true";

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
      -ESTRPIPE);

  gst_object_unref (gstpipe);
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
 * @brief Shared state for the flexible-input regression tests below (issue #4928).
 */
typedef struct _FlexCbData {
  GMutex lock;
  guint invoke_count;
  gsize last_size[NNS_TENSOR_MEMORY_MAX];
  tensor_type dynamic_in_type;
} FlexCbData;

/**
 * @brief Non-dynamic custom-easy invoke: records the call count and each input
 * tensor size, then copies tensor 0 into the (pre-allocated) output 0.
 */
static int
_flex_record_invoke (void *data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  FlexCbData *cbdata = (FlexCbData *) data;
  guint i, n;

  g_mutex_lock (&cbdata->lock);
  cbdata->invoke_count++;

  n = MIN (prop->input_meta.num_tensors, (guint) NNS_TENSOR_MEMORY_MAX);
  for (i = 0; i < n; i++)
    cbdata->last_size[i] = input[i].size;

  memcpy (output[0].data, input[0].data, MIN (output[0].size, input[0].size));
  g_mutex_unlock (&cbdata->lock);

  return 0;
}

/**
 * @brief Dynamic custom-easy invoke: records the input type it was handed (pinning that invoke-dynamic still refreshes input info per buffer), echoes the input as output.
 */
static int
_flex_dynamic_echo (void *data, const GstTensorsInfo *in_info,
    GstTensorsInfo *out_info, const GstTensorMemory *input, GstTensorMemory *output)
{
  FlexCbData *cbdata = (FlexCbData *) data;

  g_mutex_lock (&cbdata->lock);
  cbdata->invoke_count++;
  cbdata->dynamic_in_type = in_info->info[0].type;
  g_mutex_unlock (&cbdata->lock);

  gst_tensors_info_copy (out_info, in_info);
  out_info->format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  output[0].size = input[0].size;
  output[0].data = g_malloc (output[0].size);
  memcpy (output[0].data, input[0].data, output[0].size);

  return 0;
}

/**
 * @brief Build one flexible-format GstMemory with a real header
 * (type/dimension) and a zero-filled payload sized to match that header.
 */
static GstMemory *
_flex_new_mem (tensor_type type, const gchar *dim_str)
{
  GstTensorInfo info;
  GstTensorMetaInfo meta;
  GstMemory *raw, *flex;
  GstMapInfo map;
  gsize payload;

  gst_tensor_info_init (&info);
  info.type = type;
  gst_tensor_parse_dimension (dim_str, info.dimension);

  gst_tensor_info_convert_to_meta (&info, &meta);
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  payload = gst_tensor_info_get_size (&info);
  raw = gst_allocator_alloc (NULL, payload, NULL);

  if (gst_memory_map (raw, &map, GST_MAP_WRITE)) {
    memset (map.data, 0, payload);
    gst_memory_unmap (raw, &map);
  }

  flex = gst_tensor_meta_info_append_header (&meta, raw);
  gst_memory_unref (raw);
  gst_tensor_info_free (&info);

  return flex;
}

/**
 * @brief Build a flexible-format GstBuffer wrapping a single memory block from _flex_new_mem().
 */
static GstBuffer *
_flex_new_buffer (tensor_type type, const gchar *dim_str)
{
  GstBuffer *buf = gst_buffer_new ();

  gst_buffer_append_memory (buf, _flex_new_mem (type, dim_str));
  return buf;
}

/**
 * @brief Register a single-tensor custom-easy model (same shape for input and output) for the flexible-input regression tests.
 */
static void
_flex_register_model (const gchar *model, tensor_type type, const gchar *dim_str,
    const gchar *tensor_name, NNS_custom_invoke func, void *data)
{
  GstTensorsInfo info;

  gst_tensors_info_init (&info);
  info.num_tensors = 1U;
  info.info[0].type = type;
  gst_tensor_parse_dimension (dim_str, info.info[0].dimension);
  info.info[0].name = tensor_name ? g_strdup (tensor_name) : NULL;

  ASSERT_EQ (0, NNS_custom_easy_register (model, func, data, &info, &info));
  gst_tensors_info_free (&info);
}

/**
 * @brief Build a tensor_filter element (custom-easy framework, given model)
 * wrapped in a GstHarness, with flexible sink caps already negotiated.
 */
static GstHarness *
_flex_new_harness (const gchar *model, gboolean invoke_dynamic, GstBus **bus_out)
{
  GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);
  GstBus *bus = gst_bus_new ();
  GstHarness *h;

  g_assert (filter != NULL);
  gst_element_set_bus (filter, bus);
  g_object_set (filter, "framework", "custom-easy", "model", model,
      "invoke-dynamic", invoke_dynamic, NULL);

  h = gst_harness_new_with_element (filter, "sink", "src");
  gst_object_unref (filter);

  gst_harness_set_src_caps_str (h, "other/tensors,format=flexible,framerate=(fraction)0/1");

  if (bus_out)
    *bus_out = bus;
  else
    gst_object_unref (bus);

  return h;
}

/**
 * @brief Pop and validate the single STREAM/WRONG_TYPE error posted by tensor_filter on the given bus.
 */
static void
_flex_expect_wrong_type_error (GstHarness *h, GstBus *bus)
{
  GstMessage *msg = gst_bus_pop_filtered (bus, GST_MESSAGE_ERROR);
  GError *err = NULL;

  ASSERT_TRUE (msg != NULL);
  EXPECT_EQ (GST_MESSAGE_SRC (msg), (GstObject *) h->element);

  gst_message_parse_error (msg, &err, NULL);
  EXPECT_TRUE (g_error_matches (err, GST_STREAM_ERROR, GST_STREAM_ERROR_WRONG_TYPE));

  g_error_free (err);
  gst_message_unref (msg);
}

/**
 * @brief A flexible input tensor whose header matches the model's type/dimension exactly must be invoked.
 */
TEST (tensorFilterCustom, flexibleInputMatchedType)
{
  const gchar *model = "flex_matched_type";
  FlexCbData cbdata;
  GstHarness *h;
  GstBuffer *out;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);

  _flex_register_model (model, _NNS_FLOAT32, "4:2:1:1", "in0", _flex_record_invoke, &cbdata);
  h = _flex_new_harness (model, FALSE, NULL);

  EXPECT_EQ (GST_FLOW_OK, gst_harness_push (h, _flex_new_buffer (_NNS_FLOAT32, "4:2:1:1")));
  EXPECT_EQ (1U, cbdata.invoke_count);
  EXPECT_EQ (32U, cbdata.last_size[0]);

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  gst_buffer_unref (out);

  gst_harness_teardown (h);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
}

/**
 * @brief A flexible input tensor of a lower rank than the model, with matching leading dims and trailing 1s, must be invoked.
 */
TEST (tensorFilterCustom, flexibleInputTrailingOnes)
{
  const gchar *model = "flex_trailing_ones";
  FlexCbData cbdata;
  GstHarness *h;
  GstBuffer *out;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);

  _flex_register_model (model, _NNS_FLOAT32, "4:2:1:1", "in0", _flex_record_invoke, &cbdata);
  h = _flex_new_harness (model, FALSE, NULL);

  EXPECT_EQ (GST_FLOW_OK, gst_harness_push (h, _flex_new_buffer (_NNS_FLOAT32, "4:2")));
  EXPECT_EQ (1U, cbdata.invoke_count);
  EXPECT_EQ (32U, cbdata.last_size[0]);

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  gst_buffer_unref (out);

  gst_harness_teardown (h);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
}

/**
 * @brief A flexible input tensor of a different type but the same byte count as the model input must be refused, not invoked.
 */
TEST (tensorFilterCustom, flexibleInputTypeSameSize_n)
{
  const gchar *model = "flex_type_same_size_n";
  FlexCbData cbdata;
  GstHarness *h;
  GstBus *bus;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);

  _flex_register_model (model, _NNS_FLOAT32, "4:2:1:1", "in0", _flex_record_invoke, &cbdata);
  h = _flex_new_harness (model, FALSE, &bus);

  EXPECT_EQ (GST_FLOW_ERROR, gst_harness_push (h, _flex_new_buffer (_NNS_INT32, "4:2:1:1")));
  EXPECT_EQ (0U, cbdata.invoke_count);
  _flex_expect_wrong_type_error (h, bus);

  gst_harness_teardown (h);
  gst_object_unref (bus);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
}

/**
 * @brief A flexible input tensor of a different type and a different byte count
 * than the model input must be refused, not invoked.
 */
TEST (tensorFilterCustom, flexibleInputTypeOtherSize_n)
{
  const gchar *model = "flex_type_other_size_n";
  FlexCbData cbdata;
  GstHarness *h;
  GstBus *bus;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);

  _flex_register_model (model, _NNS_FLOAT32, "4:2:1:1", "in0", _flex_record_invoke, &cbdata);
  h = _flex_new_harness (model, FALSE, &bus);

  EXPECT_EQ (GST_FLOW_ERROR, gst_harness_push (h, _flex_new_buffer (_NNS_UINT8, "4:2:1:1")));
  EXPECT_EQ (0U, cbdata.invoke_count);
  EXPECT_EQ (0U, cbdata.last_size[0]);
  _flex_expect_wrong_type_error (h, bus);

  gst_harness_teardown (h);
  gst_object_unref (bus);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
}

/**
 * @brief A flexible input tensor with a transposed dimension but the same byte
 * count as the model input must be refused, not invoked.
 */
TEST (tensorFilterCustom, flexibleInputDimSameSize_n)
{
  const gchar *model = "flex_dim_same_size_n";
  FlexCbData cbdata;
  GstHarness *h;
  GstBus *bus;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);

  _flex_register_model (model, _NNS_FLOAT32, "4:2:1:1", "in0", _flex_record_invoke, &cbdata);
  h = _flex_new_harness (model, FALSE, &bus);

  EXPECT_EQ (GST_FLOW_ERROR,
      gst_harness_push (h, _flex_new_buffer (_NNS_FLOAT32, "2:4:1:1")));
  EXPECT_EQ (0U, cbdata.invoke_count);
  _flex_expect_wrong_type_error (h, bus);

  gst_harness_teardown (h);
  gst_object_unref (bus);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
}

/**
 * @brief A flexible input tensor with both a different dimension and a
 * different byte count than the model input must be refused, not invoked.
 */
TEST (tensorFilterCustom, flexibleInputDimOtherSize_n)
{
  const gchar *model = "flex_dim_other_size_n";
  FlexCbData cbdata;
  GstHarness *h;
  GstBus *bus;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);

  _flex_register_model (model, _NNS_FLOAT32, "4:2:1:1", "in0", _flex_record_invoke, &cbdata);
  h = _flex_new_harness (model, FALSE, &bus);

  EXPECT_EQ (GST_FLOW_ERROR,
      gst_harness_push (h, _flex_new_buffer (_NNS_FLOAT32, "4:1:1:1")));
  EXPECT_EQ (0U, cbdata.invoke_count);
  _flex_expect_wrong_type_error (h, bus);

  gst_harness_teardown (h);
  gst_object_unref (bus);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
}

/**
 * @brief A flexible input buffer with an invalid (unparsable) header must be refused, not invoked.
 */
TEST (tensorFilterCustom, flexibleInputInvalidHeader_n)
{
  const gchar *model = "flex_invalid_header_n";
  const gsize size = 160; /* >= 128 (header) + 32 (payload): parse_header reads fixed offsets */
  FlexCbData cbdata;
  GstHarness *h;
  GstBus *bus;
  GstBuffer *buf;
  GstMemory *mem;
  GstMapInfo map;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);

  _flex_register_model (model, _NNS_FLOAT32, "4:2:1:1", "in0", _flex_record_invoke, &cbdata);
  h = _flex_new_harness (model, FALSE, &bus);

  mem = gst_allocator_alloc (NULL, size, NULL);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));
  memset (map.data, 0, size);
  gst_memory_unmap (mem, &map);

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, mem);

  EXPECT_EQ (GST_FLOW_ERROR, gst_harness_push (h, buf));
  EXPECT_EQ (0U, cbdata.invoke_count);
  _flex_expect_wrong_type_error (h, bus);

  gst_harness_teardown (h);
  gst_object_unref (bus);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
}

/**
 * @brief A flexible input tensor tagged sparse must be refused, even if its
 * type, dimension and byte count match the model input.
 */
TEST (tensorFilterCustom, flexibleInputSparseHeader_n)
{
  const gchar *model = "flex_sparse_header_n";
  FlexCbData cbdata;
  GstTensorInfo info;
  GstTensorMetaInfo meta;
  GstHarness *h;
  GstBus *bus;
  GstBuffer *buf;
  GstMemory *raw;
  gpointer payload;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);

  _flex_register_model (model, _NNS_FLOAT32, "4:2:1:1", "in0", _flex_record_invoke, &cbdata);
  h = _flex_new_harness (model, FALSE, &bus);

  gst_tensor_info_init (&info);
  info.type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("4:2:1:1", info.dimension);
  gst_tensor_info_convert_to_meta (&info, &meta);
  meta.format = _NNS_TENSOR_FORMAT_SPARSE;
  /* 4 x (float32 value + uint32 index) = 32 bytes, the dense size of the model input */
  meta.sparse_info.nnz = 4;
  EXPECT_EQ (32U, gst_tensor_meta_info_get_data_size (&meta));

  payload = g_malloc0 (32);
  raw = gst_memory_new_wrapped ((GstMemoryFlags) 0, payload, 32, 0, 32, payload, g_free);
  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, gst_tensor_meta_info_append_header (&meta, raw));
  gst_memory_unref (raw);

  EXPECT_EQ (GST_FLOW_ERROR, gst_harness_push (h, buf));
  EXPECT_EQ (0U, cbdata.invoke_count);
  _flex_expect_wrong_type_error (h, bus);

  gst_harness_teardown (h);
  gst_object_unref (bus);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
}

/**
 * @brief The model's input info (type/dimension/name) must survive a refused
 * flexible buffer, and a later matching buffer must still be accepted.
 */
TEST (tensorFilterCustom, flexibleInputKeepsModelInfo)
{
  const gchar *model = "flex_keeps_model_info";
  FlexCbData cbdata;
  GstHarness *h;
  GstBus *bus;
  GstBuffer *out;
  gchar *strval;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);

  _flex_register_model (model, _NNS_FLOAT32, "4:2:1:1", "in0", _flex_record_invoke, &cbdata);
  h = _flex_new_harness (model, FALSE, &bus);

  /* refused: same byte count (32) as the model input, but a different type */
  EXPECT_EQ (GST_FLOW_ERROR, gst_harness_push (h, _flex_new_buffer (_NNS_INT32, "4:2:1:1")));
  EXPECT_EQ (0U, cbdata.invoke_count);
  _flex_expect_wrong_type_error (h, bus);

  g_object_get (h->element, "inputtype", &strval, NULL);
  EXPECT_STREQ ("float32", strval);
  g_free (strval);

  g_object_get (h->element, "input", &strval, NULL);
  EXPECT_STREQ ("4:2:1:1", strval);
  g_free (strval);

  g_object_get (h->element, "inputname", &strval, NULL);
  EXPECT_STREQ ("in0", strval);
  g_free (strval);

  /* a matching buffer pushed right after the refusal must still be accepted */
  EXPECT_EQ (GST_FLOW_OK, gst_harness_push (h, _flex_new_buffer (_NNS_FLOAT32, "4:2:1:1")));
  EXPECT_EQ (1U, cbdata.invoke_count);

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  gst_buffer_unref (out);

  gst_harness_teardown (h);
  gst_object_unref (bus);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
}

/**
 * @brief With invoke-dynamic=TRUE, the dynamic invoke function must still receive input info parsed from each buffer's own header, unchanged by this fix.
 */
TEST (tensorFilterCustom, flexibleDynamicInputFollowsBuffer)
{
  const gchar *model = "flex_dynamic_follows_buffer";
  GstTensorsInfo info_in;
  FlexCbData cbdata;
  GstHarness *h;
  GstBuffer *out;
  int ret;

  memset (&cbdata, 0, sizeof (cbdata));
  g_mutex_init (&cbdata.lock);
  cbdata.dynamic_in_type = _NNS_END;

  gst_tensors_info_init (&info_in);
  info_in.num_tensors = 1U;
  info_in.info[0].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("4:2:1:1", info_in.info[0].dimension);

  ret = NNS_custom_easy_dynamic_register (model, _flex_dynamic_echo, &cbdata, &info_in);
  ASSERT_EQ (0, ret);
  gst_tensors_info_free (&info_in);

  h = _flex_new_harness (model, TRUE, NULL);
  gst_harness_set_sink_caps_str (h, "other/tensors,format=flexible,framerate=(fraction)0/1");

  /* buffer type deliberately differs from the registered in_info: dynamic path takes it as-is */
  EXPECT_EQ (GST_FLOW_OK, gst_harness_push (h, _flex_new_buffer (_NNS_INT32, "4:2:1:1")));
  EXPECT_EQ (1U, cbdata.invoke_count);
  EXPECT_EQ (_NNS_INT32, cbdata.dynamic_in_type);

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  gst_buffer_unref (out);

  gst_harness_teardown (h);
  EXPECT_EQ (0, NNS_custom_easy_unregister (model));
  g_mutex_clear (&cbdata.lock);
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
