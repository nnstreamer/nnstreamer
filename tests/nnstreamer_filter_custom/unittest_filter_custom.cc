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
#include <sys/mman.h>
#include <tensor_filter_custom_easy.h>
#include <unistd.h>
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

#define FLEX_IN_HSIZE (128U)
#define FLEX_IN_MODEL_SIZE (176U)

/**
 * @brief What the custom-easy filters of the flexible input tests have seen.
 */
typedef struct {
  guint invoked;
  gsize in_size;
  guint in_dim;
  tensor_type in_type;
  gchar *in_name;
  guint num_out;
  gint invalid_out;
} flex_in_data;

/**
 * @brief Static custom-easy filter recording its input and the configured input info.
 */
static int
_flex_in_static (void *data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  flex_in_data *d = (flex_in_data *) data;

  d->invoked++;
  d->in_size = input[0].size;
  g_free (d->in_name);
  d->in_name = g_strdup (prop->input_meta.info[0].name);
  memcpy (output[0].data, input[0].data, MIN (input[0].size, output[0].size));
  return 0;
}

/**
 * @brief Dynamic custom-easy filter recording its input info, leaving the output info of @a invalid_out unset.
 */
static int
_flex_in_dynamic (void *data, const GstTensorsInfo *in_info,
    GstTensorsInfo *out_info, const GstTensorMemory *input, GstTensorMemory *output)
{
  flex_in_data *d = (flex_in_data *) data;
  guint i;

  d->invoked++;
  d->in_size = input[0].size;
  d->in_dim = in_info->info[0].dimension[0];
  d->in_type = in_info->info[0].type;
  g_free (d->in_name);
  d->in_name = g_strdup (in_info->info[0].name);

  gst_tensors_info_free (out_info);
  gst_tensors_info_init (out_info);
  out_info->num_tensors = d->num_out;
  out_info->format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  for (i = 0; i < d->num_out; i++) {
    if ((gint) i != d->invalid_out) {
      out_info->info[i].type = _NNS_UINT8;
      out_info->info[i].dimension[0] = (guint) input[0].size;
    }
    output[i].size = input[0].size;
    output[i].data = _g_memdup (input[0].data, input[0].size);
  }
  return 0;
}

/**
 * @brief Register a custom-easy model taking one uint8 tensor named "in0".
 * @details The static model takes and returns FLEX_IN_MODEL_SIZE bytes; the dynamic one takes anything.
 */
static int
_flex_in_register (const gchar *model, gboolean dynamic, flex_in_data *d)
{
  GstTensorsInfo in_info, out_info;
  int ret;

  memset (d, 0, sizeof (*d));
  d->num_out = 1;
  d->invalid_out = -1;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  in_info.num_tensors = out_info.num_tensors = 1;
  in_info.info[0].name = g_strdup ("in0");

  if (dynamic) {
    in_info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
    ret = NNS_custom_easy_dynamic_register (model, _flex_in_dynamic, d, &in_info);
  } else {
    in_info.info[0].type = out_info.info[0].type = _NNS_UINT8;
    gst_tensor_parse_dimension ("176:1:1:1", in_info.info[0].dimension);
    gst_tensor_parse_dimension ("176:1:1:1", out_info.info[0].dimension);
    ret = NNS_custom_easy_register (model, _flex_in_static, d, &in_info, &out_info);
  }

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  return ret;
}

/**
 * @brief Harness a tensor_filter running the given custom-easy model on a flexible stream, with a bus to catch its errors.
 */
static GstHarness *
_flex_in_harness (const gchar *model, gboolean dynamic)
{
  GstHarness *h;
  GstBus *bus;
  gchar *desc;

  desc = g_strdup_printf ("tensor_filter framework=custom-easy model=%s invoke-dynamic=%s",
      model, dynamic ? "true" : "false");
  h = gst_harness_new_parse (desc);
  g_free (desc);

  bus = gst_bus_new ();
  gst_element_set_bus (h->element, bus);
  gst_object_unref (bus);

  if (dynamic)
    gst_harness_set_sink_caps_str (h, "other/tensors,format=flexible");
  gst_harness_set_src_caps_str (h, "other/tensors,format=flexible,framerate=(fraction)0/1");
  return h;
}

/**
 * @brief Fill @a raw with a flexible uint8 tensor of @a dim bytes: a meta header, then 0, 1, 2, ...
 */
static void
_flex_in_fill (guint8 *raw, guint dim)
{
  GstTensorInfo info;
  GstTensorMetaInfo meta;
  guint i;

  gst_tensor_info_init (&info);
  info.type = _NNS_UINT8;
  info.dimension[0] = dim;

  ASSERT_TRUE (gst_tensor_info_convert_to_meta (&info, &meta));
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  meta.media_type = _NNS_TENSOR;
  ASSERT_TRUE (gst_tensor_meta_info_update_header (&meta, raw));

  for (i = 0; i < dim; i++)
    raw[FLEX_IN_HSIZE + i] = (guint8) i;
}

/**
 * @brief Push one memory of @a size bytes copied from @a raw.
 */
static GstFlowReturn
_flex_in_push (GstHarness *h, const guint8 *raw, gsize size)
{
  return gst_harness_push (h, gst_buffer_new_wrapped (_g_memdup (raw, size), size));
}

/**
 * @brief Whether tensor_filter has posted the STREAM/WRONG_TYPE error that reports a refused input header.
 */
static gboolean
_flex_in_refused_header (GstHarness *h)
{
  GstBus *bus = gst_element_get_bus (h->element);
  GstMessage *msg;
  GError *err = NULL;
  gboolean refused = FALSE;

  msg = gst_bus_pop_filtered (bus, GST_MESSAGE_ERROR);
  if (msg) {
    gst_message_parse_error (msg, &err, NULL);
    refused = g_error_matches (err, GST_STREAM_ERROR, GST_STREAM_ERROR_WRONG_TYPE)
              && g_str_equal (G_OBJECT_TYPE_NAME (GST_MESSAGE_SRC (msg)), "GstTensorFilter");
    g_clear_error (&err);
    gst_message_unref (msg);
  }

  gst_object_unref (bus);
  return refused;
}

/**
 * @brief A flexible memory shorter than a meta header is refused before a header is read from it.
 * @details The memory ends where an inaccessible page starts, so reading a header from it faults.
 */
TEST (tensorFilterFlexInput, shortMemory_n)
{
  const gsize page = (gsize) sysconf (_SC_PAGESIZE);
  guint8 raw[FLEX_IN_HSIZE + FLEX_IN_MODEL_SIZE];
  flex_in_data data;
  GstHarness *h;
  guint8 *pages, *tail;

  ASSERT_EQ (_flex_in_register ("flex_in_short", FALSE, &data), 0);
  _flex_in_fill (raw, FLEX_IN_MODEL_SIZE);

  pages = (guint8 *) mmap (NULL, 2 * page, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE (pages, MAP_FAILED);
  ASSERT_EQ (mprotect (pages + page, page, PROT_NONE), 0);
  tail = pages + page - 8;
  memcpy (tail, raw, 8);

  h = _flex_in_harness ("flex_in_short", FALSE);
  EXPECT_EQ (gst_harness_push (h, gst_buffer_new_wrapped_full (GST_MEMORY_FLAG_READONLY,
                                      tail, 8, 0, 8, NULL, NULL)),
      GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 0U);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  EXPECT_TRUE (_flex_in_refused_header (h));

  gst_harness_teardown (h);
  munmap (pages, 2 * page);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_short"), 0);
  g_free (data.in_name);
}

/**
 * @brief A flexible memory not starting with a meta header is refused, though the whole memory is the size the model takes.
 */
TEST (tensorFilterFlexInput, brokenHeaderMagic_n)
{
  guint8 raw[FLEX_IN_HSIZE + FLEX_IN_MODEL_SIZE];
  flex_in_data data;
  GstHarness *h;

  ASSERT_EQ (_flex_in_register ("flex_in_magic", FALSE, &data), 0);
  _flex_in_fill (raw, FLEX_IN_MODEL_SIZE);
  ((guint32 *) raw)[0] = 0;

  h = _flex_in_harness ("flex_in_magic", FALSE);
  EXPECT_EQ (_flex_in_push (h, raw, FLEX_IN_MODEL_SIZE), GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 0U);
  EXPECT_TRUE (_flex_in_refused_header (h));

  gst_harness_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_magic"), 0);
  g_free (data.in_name);
}

/**
 * @brief A meta header with an invalid type is refused, though the data after it is the size the model takes.
 */
TEST (tensorFilterFlexInput, brokenHeaderType_n)
{
  guint8 raw[FLEX_IN_HSIZE + FLEX_IN_MODEL_SIZE];
  flex_in_data data;
  GstHarness *h;

  ASSERT_EQ (_flex_in_register ("flex_in_type", FALSE, &data), 0);
  _flex_in_fill (raw, FLEX_IN_MODEL_SIZE);
  ((guint32 *) raw)[2] = _NNS_END;

  h = _flex_in_harness ("flex_in_type", FALSE);
  EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 0U);
  EXPECT_TRUE (_flex_in_refused_header (h));

  gst_harness_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_type"), 0);
  g_free (data.in_name);
}

/**
 * @brief A header describing fewer bytes than follow it is refused, though the memory holds what the model takes.
 */
TEST (tensorFilterFlexInput, headerSizeMismatch_n)
{
  guint8 raw[FLEX_IN_HSIZE + FLEX_IN_MODEL_SIZE];
  flex_in_data data;
  GstHarness *h;

  ASSERT_EQ (_flex_in_register ("flex_in_mismatch", FALSE, &data), 0);
  _flex_in_fill (raw, FLEX_IN_MODEL_SIZE);
  ((guint32 *) raw)[3] = 16;

  h = _flex_in_harness ("flex_in_mismatch", FALSE);
  EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 0U);
  EXPECT_TRUE (_flex_in_refused_header (h));

  gst_harness_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_mismatch"), 0);
  g_free (data.in_name);
}

/**
 * @brief A well-formed flexible tensor smaller than the model's input does not reach the model.
 * @details The header itself is valid, so the refusal is the model size check's, not the header check's.
 */
TEST (tensorFilterFlexInput, smallerThanModel_n)
{
  guint8 raw[FLEX_IN_HSIZE + 16];
  flex_in_data data;
  GstHarness *h;

  ASSERT_EQ (_flex_in_register ("flex_in_smaller", FALSE, &data), 0);
  _flex_in_fill (raw, 16);

  h = _flex_in_harness ("flex_in_smaller", FALSE);
  EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 0U);
  EXPECT_FALSE (_flex_in_refused_header (h));

  gst_harness_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_smaller"), 0);
  g_free (data.in_name);
}

/**
 * @brief A sparse header is refused with and without the dynamic invoke, though its dense size matches the data.
 */
TEST (tensorFilterFlexInput, sparseHeader_n)
{
  const gchar *models[] = { "flex_in_sparse", "flex_in_sparse_dynamic" };
  guint8 raw[FLEX_IN_HSIZE + FLEX_IN_MODEL_SIZE];
  flex_in_data data;
  GstHarness *h;
  guint i;

  _flex_in_fill (raw, FLEX_IN_MODEL_SIZE);
  ((guint32 *) raw)[19] = _NNS_TENSOR_FORMAT_SPARSE;

  for (i = 0; i < G_N_ELEMENTS (models); i++) {
    ASSERT_EQ (_flex_in_register (models[i], i == 1, &data), 0);

    h = _flex_in_harness (models[i], i == 1);
    EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_ERROR);
    EXPECT_EQ (data.invoked, 0U);
    EXPECT_TRUE (_flex_in_refused_header (h));

    gst_harness_teardown (h);
    EXPECT_EQ (NNS_custom_easy_unregister (models[i]), 0);
    g_free (data.in_name);
  }
}

/**
 * @brief A header of a version that validates but cannot be sized is refused.
 * @details The dimension covers the whole memory, so taking the header size of 0 at its word would pass every size check.
 */
TEST (tensorFilterFlexInput, unknownVersion_n)
{
  guint8 raw[FLEX_IN_HSIZE + FLEX_IN_MODEL_SIZE];
  GstTensorMetaInfo meta;
  flex_in_data data;
  GstHarness *h;

  ASSERT_EQ (_flex_in_register ("flex_in_version", TRUE, &data), 0);
  _flex_in_fill (raw, FLEX_IN_MODEL_SIZE);
  ((guint32 *) raw)[1] = 0xDE002000U;
  ((guint32 *) raw)[3] = sizeof (raw);

  EXPECT_TRUE (gst_tensor_meta_info_parse_header (&meta, raw));
  EXPECT_EQ (gst_tensor_meta_info_get_header_size (&meta), 0U);

  h = _flex_in_harness ("flex_in_version", TRUE);
  EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 0U);
  EXPECT_TRUE (_flex_in_refused_header (h));

  gst_harness_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_version"), 0);
  g_free (data.in_name);
}

/**
 * @brief A flexible tensor of the model's size reaches the model with the configured input info intact.
 */
TEST (tensorFilterFlexInput, keepsConfiguredInfo)
{
  guint8 raw[FLEX_IN_HSIZE + FLEX_IN_MODEL_SIZE];
  flex_in_data data;
  GstHarness *h;
  GstElement *filter;
  GstBuffer *out;
  GstMapInfo map;
  gchar *str;

  ASSERT_EQ (_flex_in_register ("flex_in_keeps", FALSE, &data), 0);
  _flex_in_fill (raw, FLEX_IN_MODEL_SIZE);

  h = _flex_in_harness ("flex_in_keeps", FALSE);
  EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_OK);
  EXPECT_EQ (data.invoked, 1U);
  EXPECT_EQ (data.in_size, FLEX_IN_MODEL_SIZE);
  EXPECT_STREQ (data.in_name, "in0");

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  ASSERT_TRUE (gst_buffer_map (out, &map, GST_MAP_READ));
  EXPECT_EQ (map.size, FLEX_IN_MODEL_SIZE);
  EXPECT_EQ (memcmp (map.data, raw + FLEX_IN_HSIZE, MIN (map.size, FLEX_IN_MODEL_SIZE)), 0);
  gst_buffer_unmap (out, &map);
  gst_buffer_unref (out);

  filter = gst_harness_find_element (h, "tensor_filter");
  ASSERT_TRUE (filter != NULL);
  g_object_get (filter, "inputname", &str, NULL);
  EXPECT_STREQ (str, "in0");
  g_free (str);
  g_object_get (filter, "input", &str, NULL);
  EXPECT_STREQ (str, "176:1:1:1");
  g_free (str);
  gst_object_unref (filter);
  EXPECT_FALSE (_flex_in_refused_header (h));

  gst_harness_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_keeps"), 0);
  g_free (data.in_name);
}

/**
 * @brief The dynamic invoke takes the type and dimension of each incoming tensor, keeping the configured name.
 */
TEST (tensorFilterFlexInput, dynamicTakesEachHeader)
{
  guint8 raw[FLEX_IN_HSIZE + 24];
  const guint dims[] = { 16, 24 };
  GstTensorMetaInfo meta;
  flex_in_data data;
  GstHarness *h;
  GstBuffer *out;
  GstMemory *mem;
  GstMapInfo map;
  guint i;

  ASSERT_EQ (_flex_in_register ("flex_in_dynamic", TRUE, &data), 0);
  h = _flex_in_harness ("flex_in_dynamic", TRUE);

  for (i = 0; i < G_N_ELEMENTS (dims); i++) {
    _flex_in_fill (raw, dims[i]);
    EXPECT_EQ (_flex_in_push (h, raw, FLEX_IN_HSIZE + dims[i]), GST_FLOW_OK);
    EXPECT_EQ (data.invoked, i + 1);
    EXPECT_EQ (data.in_size, dims[i]);
    EXPECT_EQ (data.in_dim, dims[i]);
    EXPECT_EQ (data.in_type, _NNS_UINT8);
    EXPECT_STREQ (data.in_name, "in0");

    out = gst_harness_pull (h);
    ASSERT_TRUE (out != NULL);
    EXPECT_EQ (gst_buffer_n_memory (out), 1U);
    mem = gst_buffer_peek_memory (out, 0);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    ASSERT_EQ (map.size, FLEX_IN_HSIZE + dims[i]);
    EXPECT_TRUE (gst_tensor_meta_info_parse_header (&meta, map.data));
    EXPECT_EQ (meta.dimension[0], dims[i]);
    EXPECT_EQ (memcmp (map.data + FLEX_IN_HSIZE, raw + FLEX_IN_HSIZE, dims[i]), 0);
    gst_memory_unmap (mem, &map);
    gst_buffer_unref (out);
  }

  gst_harness_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_dynamic"), 0);
  g_free (data.in_name);
}

/**
 * @brief An output tensor info the dynamic invoke leaves invalid fails the buffer instead of pushing a broken one.
 * @details The invalid output sits between two valid ones, so both an appended and a pending output are released.
 */
TEST (tensorFilterFlexInput, dynamicInvalidOutputInfo_n)
{
  guint8 raw[FLEX_IN_HSIZE + 16];
  flex_in_data data;
  GstHarness *h;

  ASSERT_EQ (_flex_in_register ("flex_in_invalid_out", TRUE, &data), 0);
  data.num_out = 3;
  data.invalid_out = 1;
  _flex_in_fill (raw, 16);

  h = _flex_in_harness ("flex_in_invalid_out", TRUE);
  EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 1U);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  data.invalid_out = -1;
  EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);

  gst_harness_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_invalid_out"), 0);
  g_free (data.in_name);
}

static guint flex_in_async_dispatched = 0; /**< outputs dispatched by the async sub-plugin */

/**
 * @brief Framework info of the async test sub-plugin.
 */
static int
_flex_in_async_fw_info (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    GstTensorFilterFrameworkInfo *fw_info)
{
  UNUSED (self);
  UNUSED (prop);
  UNUSED (private_data);
  memset (fw_info, 0, sizeof (*fw_info));
  fw_info->name = "flex_in_async";
  fw_info->allocate_in_invoke = 1;
  fw_info->run_without_model = 1;
  return 0;
}

/**
 * @brief Model info of the async test sub-plugin: one uint8 tensor, like the custom-easy models above.
 */
static int
_flex_in_async_model_info (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    model_info_ops ops, GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  UNUSED (self);
  UNUSED (prop);
  UNUSED (private_data);
  if (ops != GET_IN_OUT_INFO)
    return -ENOENT;

  gst_tensors_info_init (in_info);
  gst_tensors_info_init (out_info);
  in_info->num_tensors = out_info->num_tensors = 1;
  in_info->format = out_info->format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  return 0;
}

/**
 * @brief Event handler of the async test sub-plugin, leaving the release of the output data to tensor-filter.
 */
static int
_flex_in_async_event (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data, event_ops ops,
    GstTensorFilterFrameworkEventData *data)
{
  UNUSED (self);
  UNUSED (prop);
  UNUSED (private_data);
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

/**
 * @brief Invoke of the async test sub-plugin: dispatches three outputs twice, the middle output info of the second left invalid.
 */
static int
_flex_in_async_invoke (const GstTensorFilterFramework *self, GstTensorFilterProperties *prop,
    void *private_data, const GstTensorMemory *input, GstTensorMemory *output)
{
  GstTensorMemory out[3];
  guint i, k;

  UNUSED (self);
  UNUSED (private_data);
  UNUSED (output);

  for (k = 0; k < 2; k++) {
    gst_tensors_info_free (&prop->output_meta);
    gst_tensors_info_init (&prop->output_meta);
    prop->output_meta.num_tensors = G_N_ELEMENTS (out);
    prop->output_meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

    for (i = 0; i < G_N_ELEMENTS (out); i++) {
      if (k == 0 || i != 1) {
        prop->output_meta.info[i].type = _NNS_UINT8;
        prop->output_meta.info[i].dimension[0] = (guint) input[0].size;
      }
      out[i].size = input[0].size;
      out[i].data = _g_memdup (input[0].data, input[0].size);
    }

    flex_in_async_dispatched++;
    nnstreamer_filter_dispatch_output_async (prop, out);
  }

  /* everything went out through the async callback, drop the buffer of this invoke */
  return 1;
}

/**
 * @brief An output info left invalid fails the async output instead of pushing a broken buffer.
 */
TEST (tensorFilterFlexInput, asyncInvalidOutputInfo_n)
{
  guint8 raw[FLEX_IN_HSIZE + 16];
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);
  GstHarness *h;
  GstBuffer *out;

  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V1;
  fw->invoke = _flex_in_async_invoke;
  fw->getFrameworkInfo = _flex_in_async_fw_info;
  fw->getModelInfo = _flex_in_async_model_info;
  fw->eventHandler = _flex_in_async_event;
  ASSERT_TRUE (nnstreamer_filter_probe (fw));

  _flex_in_fill (raw, 16);
  flex_in_async_dispatched = 0;

  h = gst_harness_new_parse (
      "tensor_filter framework=flex_in_async invoke-dynamic=true invoke-async=true");
  gst_harness_set_sink_caps_str (h, "other/tensors,format=flexible");
  gst_harness_set_src_caps_str (h, "other/tensors,format=flexible,framerate=(fraction)0/1");

  EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_OK);
  EXPECT_EQ (flex_in_async_dispatched, 2U);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (gst_buffer_n_memory (out), 3U);
  gst_buffer_unref (out);

  gst_harness_teardown (h);
  nnstreamer_filter_exit ("flex_in_async");
  g_free (fw);
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
