/**
 * @file	unittest_filter_custom.cc
 * @date	11 Apr 2023
 * @brief	Unit test for tensor filter custom-easy plugin
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <dlfcn.h>
#include <errno.h>
#include <glib/gstdio.h>
#include <gmodule.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_conf.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>
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

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

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

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

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

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

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

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

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
 * @brief Tear down a harness of _flex_in_harness(), dropping what its bus holds.
 * @details Nothing drains that bus, and a queued message keeps its source
 * element, which keeps the bus, so the harness would never be freed.
 */
static void
_flex_in_teardown (GstHarness *h)
{
  GstBus *bus = gst_element_get_bus (h->element);

  gst_bus_set_flushing (bus, TRUE);
  gst_object_unref (bus);
  gst_harness_teardown (h);
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

  _flex_in_teardown (h);
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

  _flex_in_teardown (h);
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

  _flex_in_teardown (h);
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

  _flex_in_teardown (h);
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

  _flex_in_teardown (h);
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

    _flex_in_teardown (h);
    EXPECT_EQ (NNS_custom_easy_unregister (models[i]), 0);
    g_free (data.in_name);
  }
}

/**
 * @brief A header of a version this build cannot size is refused.
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

  EXPECT_FALSE (gst_tensor_meta_info_parse_header (&meta, raw));
  EXPECT_EQ (gst_tensor_meta_info_get_header_size (&meta), 0U);

  h = _flex_in_harness ("flex_in_version", TRUE);
  EXPECT_EQ (_flex_in_push (h, raw, sizeof (raw)), GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 0U);
  EXPECT_TRUE (_flex_in_refused_header (h));

  _flex_in_teardown (h);
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

  _flex_in_teardown (h);
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

  _flex_in_teardown (h);
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

  _flex_in_teardown (h);
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
 * @brief What the custom-easy filters of the typed flexible input tests have seen.
 */
typedef struct {
  guint invoked;
  gsize in_size[2];
  tensor_type in_type;
} flex_in_typed_data;

/**
 * @brief Static custom-easy filter recording its input sizes and copying each input to its output.
 */
static int
_flex_in_typed_static (void *data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  flex_in_typed_data *d = (flex_in_typed_data *) data;
  guint i;

  d->invoked++;
  for (i = 0; i < MIN (prop->input_meta.num_tensors, G_N_ELEMENTS (d->in_size)); i++) {
    d->in_size[i] = input[i].size;
    memcpy (output[i].data, input[i].data, MIN (input[i].size, output[i].size));
  }
  return 0;
}

/**
 * @brief Dynamic custom-easy filter recording the input type it is given and echoing its input.
 */
static int
_flex_in_typed_dynamic (void *data, const GstTensorsInfo *in_info,
    GstTensorsInfo *out_info, const GstTensorMemory *input, GstTensorMemory *output)
{
  flex_in_typed_data *d = (flex_in_typed_data *) data;

  d->invoked++;
  d->in_type = in_info->info[0].type;

  gst_tensors_info_free (out_info);
  gst_tensors_info_copy (out_info, in_info);
  out_info->format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  output[0].size = input[0].size;
  output[0].data = _g_memdup (input[0].data, input[0].size);
  return 0;
}

/**
 * @brief Register a custom-easy model whose input and output tensors are given as comma-separated lists.
 */
static int
_flex_in_typed_register (const gchar *model, gboolean dynamic, const gchar *types,
    const gchar *dims, const gchar *names, flex_in_typed_data *d)
{
  GstTensorsInfo info;
  int ret;

  memset (d, 0, sizeof (*d));
  d->in_type = _NNS_END;

  gst_tensors_info_init (&info);
  info.num_tensors = gst_tensors_info_parse_types_string (&info, types);
  gst_tensors_info_parse_dimensions_string (&info, dims);
  gst_tensors_info_parse_names_string (&info, names);

  if (dynamic)
    ret = NNS_custom_easy_dynamic_register (model, _flex_in_typed_dynamic, d, &info);
  else
    ret = NNS_custom_easy_register (model, _flex_in_typed_static, d, &info, &info);

  gst_tensors_info_free (&info);
  return ret;
}

/**
 * @brief Append a flexible tensor of @a type and @a dim, with a zero-filled payload, to @a buf.
 */
static void
_flex_in_typed_append (GstBuffer *buf, tensor_type type, const gchar *dim)
{
  GstTensorInfo info;
  GstTensorMetaInfo meta;
  GstMemory *mem;
  gpointer payload;
  gsize size;

  gst_tensor_info_init (&info);
  info.type = type;
  gst_tensor_parse_dimension (dim, info.dimension);
  size = gst_tensor_info_get_size (&info);

  ASSERT_TRUE (gst_tensor_info_convert_to_meta (&info, &meta));
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  meta.media_type = _NNS_TENSOR;

  payload = g_malloc0 (size);
  mem = gst_memory_new_wrapped ((GstMemoryFlags) 0, payload, size, 0, size, payload, g_free);
  gst_buffer_append_memory (buf, gst_tensor_meta_info_append_header (&meta, mem));
  gst_memory_unref (mem);
}

/**
 * @brief Push a buffer of one flexible tensor of @a type and @a dim.
 */
static GstFlowReturn
_flex_in_typed_push (GstHarness *h, tensor_type type, const gchar *dim)
{
  GstBuffer *buf = gst_buffer_new ();

  _flex_in_typed_append (buf, type, dim);
  return gst_harness_push (h, buf);
}

/**
 * @brief A flexible tensor with the type and dimension of the configured input is invoked, also with a lower rank ending in 1s.
 */
TEST (tensorFilterFlexInput, typedMatch)
{
  const gchar *dims[] = { "4:2:1:1", "4:2" };
  flex_in_typed_data data;
  GstHarness *h;
  GstBuffer *out;
  guint i;

  ASSERT_EQ (_flex_in_typed_register ("flex_in_typed_match", FALSE, "float32",
                 "4:2:1:1", "in0", &data),
      0);
  h = _flex_in_harness ("flex_in_typed_match", FALSE);

  for (i = 0; i < G_N_ELEMENTS (dims); i++) {
    EXPECT_EQ (_flex_in_typed_push (h, _NNS_FLOAT32, dims[i]), GST_FLOW_OK);
    EXPECT_EQ (data.invoked, i + 1);
    EXPECT_EQ (data.in_size[0], 32U);

    out = gst_harness_pull (h);
    ASSERT_TRUE (out != NULL);
    gst_buffer_unref (out);
  }
  EXPECT_FALSE (_flex_in_refused_header (h));

  _flex_in_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_typed_match"), 0);
}

/**
 * @brief A flexible tensor of the configured size but of another type or shape is refused, not invoked.
 */
TEST (tensorFilterFlexInput, typedSameSizeMismatch_n)
{
  const tensor_type types[] = { _NNS_INT32, _NNS_FLOAT32 };
  const gchar *dims[] = { "4:2:1:1", "2:4:1:1" };
  flex_in_typed_data data;
  GstHarness *h;
  guint i;

  ASSERT_EQ (_flex_in_typed_register ("flex_in_typed_mismatch", FALSE,
                 "float32", "4:2:1:1", "in0", &data),
      0);
  h = _flex_in_harness ("flex_in_typed_mismatch", FALSE);

  for (i = 0; i < G_N_ELEMENTS (types); i++) {
    EXPECT_EQ (_flex_in_typed_push (h, types[i], dims[i]), GST_FLOW_ERROR);
    EXPECT_EQ (data.invoked, 0U);
    EXPECT_TRUE (_flex_in_refused_header (h));
  }

  _flex_in_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_typed_mismatch"), 0);
}

/**
 * @brief Each tensor of a two-tensor flexible input is checked against its own configured tensor.
 * @details The second tensor of the refused buffer has the configured size but another type, so only a per-index check refuses it.
 */
TEST (tensorFilterFlexInput, typedSecondTensor_n)
{
  flex_in_typed_data data;
  GstHarness *h;
  GstBuffer *buf, *out;

  ASSERT_EQ (_flex_in_typed_register ("flex_in_typed_second", FALSE,
                 "float32,int32", "4:2:1:1,4:1:1:1", "in0,in1", &data),
      0);
  h = _flex_in_harness ("flex_in_typed_second", FALSE);

  buf = gst_buffer_new ();
  _flex_in_typed_append (buf, _NNS_FLOAT32, "4:2:1:1");
  _flex_in_typed_append (buf, _NNS_INT32, "4:1:1:1");
  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_OK);
  EXPECT_EQ (data.invoked, 1U);
  EXPECT_EQ (data.in_size[0], 32U);
  EXPECT_EQ (data.in_size[1], 16U);
  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (gst_buffer_n_memory (out), 2U);
  gst_buffer_unref (out);
  EXPECT_FALSE (_flex_in_refused_header (h));

  buf = gst_buffer_new ();
  _flex_in_typed_append (buf, _NNS_FLOAT32, "4:2:1:1");
  _flex_in_typed_append (buf, _NNS_FLOAT32, "4:1:1:1");
  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 1U);
  EXPECT_TRUE (_flex_in_refused_header (h));

  _flex_in_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_typed_second"), 0);
}

/**
 * @brief A refused flexible tensor leaves the configured input info as it is, and a matching tensor after it is invoked.
 */
TEST (tensorFilterFlexInput, typedKeepsConfiguredInfo)
{
  flex_in_typed_data data;
  GstElement *filter;
  GstHarness *h;
  GstBuffer *out;
  gchar *str;

  ASSERT_EQ (_flex_in_typed_register ("flex_in_typed_keeps", FALSE, "float32",
                 "4:2:1:1", "in0", &data),
      0);
  h = _flex_in_harness ("flex_in_typed_keeps", FALSE);

  EXPECT_EQ (_flex_in_typed_push (h, _NNS_INT32, "4:2:1:1"), GST_FLOW_ERROR);
  EXPECT_EQ (data.invoked, 0U);
  EXPECT_TRUE (_flex_in_refused_header (h));

  filter = gst_harness_find_element (h, "tensor_filter");
  ASSERT_TRUE (filter != NULL);
  g_object_get (filter, "inputtype", &str, NULL);
  EXPECT_STREQ (str, "float32");
  g_free (str);
  g_object_get (filter, "input", &str, NULL);
  EXPECT_STREQ (str, "4:2:1:1");
  g_free (str);
  g_object_get (filter, "inputname", &str, NULL);
  EXPECT_STREQ (str, "in0");
  g_free (str);
  gst_object_unref (filter);

  EXPECT_EQ (_flex_in_typed_push (h, _NNS_FLOAT32, "4:2:1:1"), GST_FLOW_OK);
  EXPECT_EQ (data.invoked, 1U);
  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  gst_buffer_unref (out);

  _flex_in_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_typed_keeps"), 0);
}

/**
 * @brief The dynamic invoke is not checked against the configured type: it takes the type of each incoming tensor.
 */
TEST (tensorFilterFlexInput, typedDynamicTakesType)
{
  flex_in_typed_data data;
  GstHarness *h;
  GstBuffer *out;

  ASSERT_EQ (_flex_in_typed_register ("flex_in_typed_dynamic", TRUE, "float32",
                 "4:2:1:1", "in0", &data),
      0);
  h = _flex_in_harness ("flex_in_typed_dynamic", TRUE);

  EXPECT_EQ (_flex_in_typed_push (h, _NNS_INT32, "4:2:1:1"), GST_FLOW_OK);
  EXPECT_EQ (data.invoked, 1U);
  EXPECT_EQ (data.in_type, _NNS_INT32);
  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  gst_buffer_unref (out);
  EXPECT_FALSE (_flex_in_refused_header (h));

  _flex_in_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister ("flex_in_typed_dynamic"), 0);
}

#define OUT_COMBI_MODEL "out_combi_4_8"
#define OUT_COMBI_IN_CAPS \
  "other/tensors,num_tensors=1,dimensions=(string)4,types=uint8,format=static,framerate=(fraction)0/1"

/** @brief How the output-combination test model is invoked */
typedef enum {
  OUT_COMBI_STATIC = 0, /**< static invoke returning both outputs */
  OUT_COMBI_DYNAMIC, /**< dynamic invoke returning both outputs */
  OUT_COMBI_DYNAMIC_FEWER, /**< dynamic invoke returning the 4-byte output only */
  OUT_COMBI_DYNAMIC_INVALID, /**< dynamic invoke leaving the info of the 8-byte output invalid */
} out_combi_mode;

/**
 * @brief In-code model for output-combination tests: fills a 4-byte output with 0xA0 and an 8-byte output with 0xB0.
 */
static int
cef_func_out_combi (void *data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *in, GstTensorMemory *out)
{
  UNUSED (data);
  UNUSED (prop);
  UNUSED (in);
  memset (out[0].data, 0xA0, out[0].size);
  memset (out[1].data, 0xB0, out[1].size);
  return 0;
}

/**
 * @brief Register the custom-easy model taking a 4-byte uint8 tensor and returning a 4-byte and an 8-byte uint8 tensor.
 */
static int
_out_combi_register (void)
{
  GstTensorsInfo info_in;
  GstTensorsInfo info_out;

  gst_tensors_info_init (&info_in);
  gst_tensors_info_init (&info_out);
  info_in.num_tensors = 1U;
  info_in.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("4", info_in.info[0].dimension);

  info_out.num_tensors = 2U;
  info_out.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("4", info_out.info[0].dimension);
  info_out.info[1].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("8", info_out.info[1].dimension);

  return NNS_custom_easy_register (
      OUT_COMBI_MODEL, cef_func_out_combi, NULL, &info_in, &info_out);
}

/**
 * @brief Harness a tensor_filter running the output-combination test model, with a bus to catch its errors.
 */
static GstHarness *
_out_combi_harness (const gchar *combi, gboolean flexible)
{
  GstHarness *h;
  GstBus *bus;
  gchar *desc;

  desc = g_strdup_printf ("tensor_filter framework=custom-easy model=%s output-combination=%s",
      OUT_COMBI_MODEL, combi);
  h = gst_harness_new_parse (desc);
  g_free (desc);

  bus = gst_bus_new ();
  gst_element_set_bus (h->element, bus);
  gst_object_unref (bus);

  if (flexible)
    gst_harness_set_sink_caps_str (h, "other/tensors,format=flexible");
  gst_harness_set_src_caps_str (h, OUT_COMBI_IN_CAPS);
  return h;
}

/**
 * @brief Tear down a harness of the output-combination tests and unregister the model.
 */
static void
_out_combi_teardown (GstHarness *h)
{
  _flex_in_teardown (h);
  EXPECT_EQ (NNS_custom_easy_unregister (OUT_COMBI_MODEL), 0);
}

/**
 * @brief Dynamic variant of cef_func_out_combi(), behaving as the out_combi_mode in @a data says.
 */
static int
cef_func_out_combi_dynamic (void *data, const GstTensorsInfo *in_info,
    GstTensorsInfo *out_info, const GstTensorMemory *input, GstTensorMemory *output)
{
  out_combi_mode mode = (out_combi_mode) GPOINTER_TO_UINT (data);
  guint i, num = (mode == OUT_COMBI_DYNAMIC_FEWER) ? 1U : 2U;

  UNUSED (in_info);
  UNUSED (input);

  gst_tensors_info_free (out_info);
  gst_tensors_info_init (out_info);
  out_info->num_tensors = num;
  out_info->format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  for (i = 0; i < num; i++) {
    output[i].size = (i == 0) ? 4 : 8;
    output[i].data = g_malloc (output[i].size);
    memset (output[i].data, (i == 0) ? 0xA0 : 0xB0, output[i].size);

    if (i == 1 && mode == OUT_COMBI_DYNAMIC_INVALID)
      continue;
    out_info->info[i].type = _NNS_UINT8;
    out_info->info[i].dimension[0] = (guint) output[i].size;
  }
  return 0;
}

/**
 * @brief Harness a dynamic-invoke tensor_filter whose output info is given by properties, so an output combination negotiates.
 */
static GstHarness *
_out_combi_dynamic_harness (const gchar *combi, out_combi_mode mode)
{
  GstTensorsInfo info_in;
  GstHarness *h;
  GstBus *bus;
  gchar *desc;
  int ret;

  gst_tensors_info_init (&info_in);
  info_in.num_tensors = 1U;
  info_in.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("4", info_in.info[0].dimension);
  ret = NNS_custom_easy_dynamic_register (OUT_COMBI_MODEL,
      cef_func_out_combi_dynamic, GUINT_TO_POINTER (mode), &info_in);
  EXPECT_EQ (ret, 0);

  desc = g_strdup_printf ("tensor_filter framework=custom-easy model=%s invoke-dynamic=true "
                          "output=4,8 outputtype=uint8,uint8 output-combination=%s",
      OUT_COMBI_MODEL, combi);
  h = gst_harness_new_parse (desc);
  g_free (desc);

  bus = gst_bus_new ();
  gst_element_set_bus (h->element, bus);
  gst_object_unref (bus);

  gst_harness_set_sink_caps_str (h, "other/tensors,format=flexible");
  gst_harness_set_src_caps_str (h, OUT_COMBI_IN_CAPS);
  return h;
}

/**
 * @brief Log handler counting the critical messages it receives in the guint @a user_data.
 */
static void
_out_combi_count_critical (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer user_data)
{
  UNUSED (domain);
  UNUSED (level);
  UNUSED (message);
  (*(guint *) user_data)++;
}

/**
 * @brief Run one 0x11-filled input buffer through the model with @a combi and check the output against the negotiated caps.
 * @details No GStreamer critical may be raised on the way, e.g., by appending a memory the model did not return.
 * @param combi value of the output-combination property
 * @param flexible negotiate flexible output tensors
 * @param mode how the model is invoked; the output of a dynamic invoke is always flexible
 * @param num expected number of memories in the output buffer
 * @param bytes expected byte each tensor is filled with, in buffer order
 * @param sizes expected data size of each tensor, in buffer order
 */
static void
_out_combi_run (const gchar *combi, gboolean flexible, out_combi_mode mode,
    guint num, const guint8 *bytes, const gsize *sizes)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstCaps *caps;
  GstTensorsConfig config;
  GstTensorMetaInfo meta;
  GstMemory *mem;
  GstMapInfo map;
  gsize hsize;
  guint i, handler, critical = 0;

  if (mode != OUT_COMBI_STATIC) {
    flexible = TRUE;
    h = _out_combi_dynamic_harness (combi, mode);
  } else {
    ASSERT_EQ (_out_combi_register (), 0);
    h = _out_combi_harness (combi, flexible);
  }

  handler = g_log_set_handler ("GStreamer",
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL),
      _out_combi_count_critical, &critical);

  in_buf = gst_harness_create_buffer (h, 4);
  gst_buffer_memset (in_buf, 0, 0x11, 4);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  out_buf = gst_harness_try_pull (h);
  caps = gst_pad_get_current_caps (h->sinkpad);
  gst_tensors_config_init (&config);

  EXPECT_TRUE (out_buf != NULL);
  EXPECT_TRUE (caps != NULL);
  if (out_buf && caps) {
    EXPECT_TRUE (gst_tensors_config_from_structure (
        &config, gst_caps_get_structure (caps, 0)));
    EXPECT_EQ (gst_tensors_config_is_flexible (&config), flexible);
    if (!flexible) {
      EXPECT_EQ (config.info.num_tensors, num);
    }
    EXPECT_EQ (gst_tensor_buffer_get_count (out_buf), num);

    for (i = 0; i < num && i < gst_tensor_buffer_get_count (out_buf); i++) {
      mem = gst_tensor_buffer_get_nth_memory (out_buf, i);
      if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
        ADD_FAILURE () << "Cannot map memory " << i;
        gst_memory_unref (mem);
        continue;
      }

      hsize = 0;
      if (flexible) {
        EXPECT_TRUE (gst_tensor_meta_info_parse_header (&meta, map.data));
        hsize = gst_tensor_meta_info_get_header_size (&meta);
        EXPECT_EQ (gst_tensor_meta_info_get_data_size (&meta), sizes[i]);
      } else {
        EXPECT_EQ (map.size, gst_tensors_info_get_size (&config.info, i));
      }

      EXPECT_EQ (map.size, hsize + sizes[i]);
      if (map.size == hsize + sizes[i]) {
        EXPECT_EQ (map.data[hsize], bytes[i]);
        EXPECT_EQ (map.data[map.size - 1], bytes[i]);
      }
      gst_memory_unmap (mem, &map);
      gst_memory_unref (mem);
    }
  }

  gst_tensors_config_free (&config);
  if (caps)
    gst_caps_unref (caps);
  if (out_buf)
    gst_buffer_unref (out_buf);
  _out_combi_teardown (h);

  EXPECT_EQ (critical, 0U);
  /* The handler is live: a critical of the domain is counted */
  g_log ("GStreamer", G_LOG_LEVEL_CRITICAL, "self-check of the counter");
  EXPECT_EQ (critical, 1U);
  g_log_remove_handler ("GStreamer", handler);
}

/**
 * @brief Test output-combination selecting the model outputs in model order.
 */
TEST (tensorFilterOutputCombination, modelOrder)
{
  const guint8 bytes[] = { 0xA0, 0xB0 };
  const gsize sizes[] = { 4, 8 };

  _out_combi_run ("o0,o1", FALSE, OUT_COMBI_STATIC, 2U, bytes, sizes);
}

/**
 * @brief Test output-combination selecting a single model output.
 */
TEST (tensorFilterOutputCombination, subset)
{
  const guint8 bytes[] = { 0xB0 };
  const gsize sizes[] = { 8 };

  _out_combi_run ("o1", FALSE, OUT_COMBI_STATIC, 1U, bytes, sizes);
}

/**
 * @brief Test output-combination listing the model outputs in reverse order; the buffer must follow the caps.
 */
TEST (tensorFilterOutputCombination, reorder)
{
  const guint8 bytes[] = { 0xB0, 0xA0 };
  const gsize sizes[] = { 8, 4 };

  _out_combi_run ("o1,o0", FALSE, OUT_COMBI_STATIC, 2U, bytes, sizes);
}

/**
 * @brief Test output-combination listing a model output twice.
 */
TEST (tensorFilterOutputCombination, repeat)
{
  const guint8 bytes[] = { 0xA0, 0xA0, 0xB0 };
  const gsize sizes[] = { 4, 4, 8 };

  _out_combi_run ("o0,o0,o1", FALSE, OUT_COMBI_STATIC, 3U, bytes, sizes);
}

/**
 * @brief Test output-combination mixing an input tensor with reordered and repeated model outputs.
 */
TEST (tensorFilterOutputCombination, inputAndReorder)
{
  const guint8 bytes[] = { 0x11, 0xB0, 0xA0, 0xB0 };
  const gsize sizes[] = { 4, 8, 4, 8 };

  _out_combi_run ("o1,i0,o0,o1", FALSE, OUT_COMBI_STATIC, 4U, bytes, sizes);
}

/**
 * @brief Test output-combination reordering flexible output tensors.
 */
TEST (tensorFilterOutputCombination, reorderFlexible)
{
  const guint8 bytes[] = { 0xB0, 0xB0, 0xA0 };
  const gsize sizes[] = { 8, 8, 4 };

  _out_combi_run ("o1,o1,o0", TRUE, OUT_COMBI_STATIC, 3U, bytes, sizes);
}

/**
 * @brief Test output-combination reordering and repeating the outputs of a dynamic invoke.
 */
TEST (tensorFilterOutputCombination, reorderDynamic)
{
  const guint8 bytes[] = { 0xB0, 0xA0, 0xA0 };
  const gsize sizes[] = { 8, 4, 4 };

  _out_combi_run ("o1,o0,o0", FALSE, OUT_COMBI_DYNAMIC, 3U, bytes, sizes);
}

/**
 * @brief Test output-combination on a dynamic invoke returning fewer outputs than the list names; the missing one is left out.
 */
TEST (tensorFilterOutputCombination, dynamicFewerOutputs)
{
  const guint8 bytes[] = { 0xA0, 0xA0 };
  const gsize sizes[] = { 4, 4 };

  _out_combi_run ("o1,o0,o0", FALSE, OUT_COMBI_DYNAMIC_FEWER, 2U, bytes, sizes);
}

/**
 * @brief Test output-combination with an output index the model does not have.
 */
TEST (tensorFilterOutputCombination, invalidIndex_n)
{
  GstHarness *h;

  ASSERT_EQ (_out_combi_register (), 0);
  h = _out_combi_harness ("o0,o2", FALSE);

  EXPECT_NE (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  _out_combi_teardown (h);
}

/**
 * @brief Test output-combination on a dynamic invoke returning an invalid info after a selected output is prepared.
 */
TEST (tensorFilterOutputCombination, invalidDynamicInfo_n)
{
  GstHarness *h;

  h = _out_combi_dynamic_harness ("o1,o0", OUT_COMBI_DYNAMIC_INVALID);

  EXPECT_NE (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  _out_combi_teardown (h);
}

/**
 * @brief Path of a file the test build has put under its 'tests' directory.
 */
static gchar *
_b3_build_path (const gchar *subdir, const gchar *file_name)
{
  const gchar *build_root = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  g_autofree gchar *fallback = NULL;

  if (build_root == NULL) {
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

    fallback = g_build_filename (root_path ? root_path : "..", "build", NULL);
    build_root = fallback;
  }

  if (subdir)
    return g_build_filename (build_root, "tests", subdir, file_name, NULL);

  return g_build_filename (build_root, "tests", file_name, NULL);
}

/**
 * @brief Path of a custom filter library built for the open-failure cases.
 */
static gchar *
_b3_model_path (const gchar *variant)
{
  g_autofree gchar *lib_name = g_strdup_printf (
      "libnnscustom_open_fail_%s%s", variant, NNSTREAMER_SO_FILE_EXTENSION);

  return _b3_build_path ("nnstreamer_example", lib_name);
}

/**
 * @brief Fill the properties the custom sub-plugin reads for a single model file.
 */
static void
_b3_set_prop (GstTensorFilterProperties *prop, const gchar **models)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = "custom";
  prop->model_files = models;
  prop->num_models = models[0] ? 1 : 0;
}

/**
 * @brief Hold a reference to a custom filter library and its call counters.
 */
typedef struct {
  GModule *module;
  guint *init_count;
  guint *exit_count;
} b3_lib_counters;

/**
 * @brief Keep the library loaded while a test reads the counters it exports.
 * @details Each variant names its counters after itself: a custom filter is loaded
 *          into the global symbol scope, where equally named symbols of the variants
 *          would interpose each other and make the counts depend on the test order.
 */
static gboolean
_b3_counters_open (b3_lib_counters *counters, const gchar *path, const gchar *variant)
{
  g_autofree gchar *init_name
      = g_strdup_printf ("nnscustom_open_fail_init_count_%s", variant);
  g_autofree gchar *exit_name
      = g_strdup_printf ("nnscustom_open_fail_exit_count_%s", variant);
  gpointer init_sym, exit_sym;

  counters->module = g_module_open (path, (GModuleFlags) 0);
  if (!counters->module)
    return FALSE;

  if (!g_module_symbol (counters->module, init_name, &init_sym)
      || !g_module_symbol (counters->module, exit_name, &exit_sym)) {
    g_module_close (counters->module);
    return FALSE;
  }

  counters->init_count = (guint *) init_sym;
  counters->exit_count = (guint *) exit_sym;
  return TRUE;
}

/**
 * @brief Release the reference taken by _b3_counters_open().
 */
static void
_b3_counters_close (b3_lib_counters *counters)
{
  g_module_close (counters->module);
}

/**
 * @brief Check whether a library is loaded, without loading it.
 * @details This answers for the process, not for one caller: it reports the
 * library as loaded while anything else still holds a reference to it.
 */
static gboolean
_b3_is_loaded (const gchar *path)
{
  void *handle = dlopen (path, RTLD_LAZY | RTLD_NOLOAD);

  if (handle == NULL)
    return FALSE;

  dlclose (handle);
  return TRUE;
}

/**
 * @brief Open a custom filter library that the sub-plugin must refuse after loading it.
 * @details The sub-plugin has to undo what it has done so far: the handle is left
 *          empty and whatever initfunc returned is released by exitfunc (#4920 B3).
 */
static void
_b3_expect_refused_open (const gchar *variant, gboolean expect_init)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("custom");
  GstTensorFilterProperties prop;
  b3_lib_counters counters;
  const gchar *models[2];
  guint init_before, exit_before;
  void *data = NULL;
  g_autofree gchar *path = _b3_model_path (variant);

  ASSERT_TRUE (sp != nullptr);
  ASSERT_TRUE (g_file_test (path, G_FILE_TEST_EXISTS));
  ASSERT_TRUE (_b3_counters_open (&counters, path, variant));

  init_before = *counters.init_count;
  exit_before = *counters.exit_count;

  models[0] = path;
  models[1] = NULL;
  _b3_set_prop (&prop, models);

  EXPECT_EQ (sp->open (&prop, &data), -EINVAL);
  EXPECT_TRUE (data == nullptr);
  EXPECT_EQ (*counters.init_count - init_before, expect_init ? 1U : 0U);
  EXPECT_EQ (*counters.exit_count - exit_before, expect_init ? 1U : 0U);

  _b3_counters_close (&counters);

  /**
   * The counters cannot tell whether the module itself has been closed, because
   * the reference above keeps the library loaded. Repeat the refused open with
   * no reference of our own: the library is then left loaded if, and only if,
   * the sub-plugin has not closed the module.
   */
  data = NULL;

  EXPECT_EQ (sp->open (&prop, &data), -EINVAL);
  EXPECT_TRUE (data == nullptr);
  EXPECT_TRUE (_b3_is_loaded (path) == FALSE)
      << "the refused open left " << variant
      << " loaded: the sub-plugin did not close the module, or the loader was told "
         "to keep every module resident";
}

/**
 * @brief Open a custom filter library with no initfunc.
 */
TEST (tensorFilterCustomOpenFail, missingInit_n)
{
  _b3_expect_refused_open ("no_init", FALSE);
}

/**
 * @brief Open a custom filter library with no input/output dimension callback.
 */
TEST (tensorFilterCustomOpenFail, missingDimension_n)
{
  _b3_expect_refused_open ("no_dim", TRUE);
}

/**
 * @brief Open a custom filter library with no invoke callback.
 */
TEST (tensorFilterCustomOpenFail, missingInvoke_n)
{
  _b3_expect_refused_open ("no_invoke", TRUE);
}

/**
 * @brief Open a custom filter library that does not exist.
 */
TEST (tensorFilterCustomOpenFail, invalidPath_n)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("custom");
  GstTensorFilterProperties prop;
  const gchar *models[2];
  void *data = NULL;
  g_autofree gchar *path = _b3_model_path ("not_built");

  ASSERT_TRUE (sp != nullptr);

  models[0] = path;
  models[1] = NULL;
  _b3_set_prop (&prop, models);

  EXPECT_EQ (sp->open (&prop, &data), -EINVAL);
  EXPECT_TRUE (data == nullptr);
}

/**
 * @brief Open a custom filter library that gives both invoke callbacks.
 */
TEST (tensorFilterCustomOpenFail, bothInvoke_n)
{
  _b3_expect_refused_open ("both_invoke", TRUE);
}

/**
 * @brief Close a custom filter library that has no exitfunc.
 * @details exitfunc is not one of the callbacks the sub-plugin requires, so a
 * library without it is opened and has to be closed without calling it.
 */
TEST (tensorFilterCustomOpenFail, closeWithoutExit)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("custom");
  GstTensorFilterProperties prop;
  const gchar *models[2];
  void *data = NULL;
  g_autofree gchar *path = _b3_model_path ("no_exit");

  ASSERT_TRUE (sp != nullptr);
  ASSERT_TRUE (g_file_test (path, G_FILE_TEST_EXISTS));

  models[0] = path;
  models[1] = NULL;
  _b3_set_prop (&prop, models);

  ASSERT_EQ (sp->open (&prop, &data), 0);

  sp->close (&prop, &data);
  EXPECT_TRUE (data == nullptr);
}

/**
 * @brief Open a file that is not a loadable library.
 */
TEST (tensorFilterCustomOpenFail, notALibrary_n)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("custom");
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  const gchar *models[2];
  void *data = NULL;
  g_autofree gchar *path = NULL;

  ASSERT_TRUE (sp != nullptr);

  if (root_path == NULL)
    root_path = "..";

  path = g_build_filename (root_path, "tests", "test_models", "labels", "labels.txt", NULL);
  ASSERT_TRUE (g_file_test (path, G_FILE_TEST_EXISTS));

  models[0] = path;
  models[1] = NULL;
  _b3_set_prop (&prop, models);

  EXPECT_EQ (sp->open (&prop, &data), -EINVAL);
  EXPECT_TRUE (data == nullptr);
}

/**
 * @brief Open a library that is not a custom filter.
 */
TEST (tensorFilterCustomOpenFail, missingSymbol_n)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("custom");
  GstTensorFilterProperties prop;
  const gchar *models[2];
  void *data = NULL;
  g_autofree gchar *lib_name = g_strdup_printf (
      "libnnstreamer_unittest_util%s", NNSTREAMER_SO_FILE_EXTENSION);
  g_autofree gchar *path = _b3_build_path (NULL, lib_name);

  ASSERT_TRUE (sp != nullptr);
  ASSERT_TRUE (g_file_test (path, G_FILE_TEST_EXISTS));

  models[0] = path;
  models[1] = NULL;
  _b3_set_prop (&prop, models);

  EXPECT_EQ (sp->open (&prop, &data), -EINVAL);
  EXPECT_TRUE (data == nullptr);
}

/**
 * @brief Open the custom sub-plugin without a model file.
 */
TEST (tensorFilterCustomOpenFail, noModel_n)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("custom");
  GstTensorFilterProperties prop;
  const gchar *models[2] = { NULL, NULL };
  void *data = NULL;

  ASSERT_TRUE (sp != nullptr);

  _b3_set_prop (&prop, models);

  EXPECT_EQ (sp->open (&prop, &data), -EINVAL);
  EXPECT_TRUE (data == nullptr);
}

/**
 * @brief Open a handle that is already open.
 */
TEST (tensorFilterCustomOpenFail, alreadyOpened_n)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("custom");
  GstTensorFilterProperties prop;
  const gchar *models[2];
  void *data = NULL;
  g_autofree gchar *path = _b3_model_path ("ok");

  ASSERT_TRUE (sp != nullptr);
  ASSERT_TRUE (g_file_test (path, G_FILE_TEST_EXISTS));

  models[0] = path;
  models[1] = NULL;
  _b3_set_prop (&prop, models);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_EQ (sp->open (&prop, &data), -EINVAL);

  sp->close (&prop, &data);
  EXPECT_TRUE (data == nullptr);
}

/**
 * @brief Open a working custom filter library with the handle a refused open has left behind.
 */
TEST (tensorFilterCustomOpenFail, reopenAfterFailure)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("custom");
  const gchar *variants[] = { "no_init", "no_dim", "no_invoke" };
  GstTensorFilterProperties prop;
  const gchar *models[2];
  void *data = NULL;
  g_autofree gchar *good_path = _b3_model_path ("ok");

  ASSERT_TRUE (sp != nullptr);
  ASSERT_TRUE (g_file_test (good_path, G_FILE_TEST_EXISTS));

  models[1] = NULL;

  for (guint i = 0; i < G_N_ELEMENTS (variants); i++) {
    g_autofree gchar *path = _b3_model_path (variants[i]);

    ASSERT_TRUE (g_file_test (path, G_FILE_TEST_EXISTS));

    models[0] = path;
    _b3_set_prop (&prop, models);
    ASSERT_EQ (sp->open (&prop, &data), -EINVAL);

    models[0] = good_path;
    _b3_set_prop (&prop, models);
    EXPECT_EQ (sp->open (&prop, &data), 0);
    EXPECT_TRUE (data != nullptr);

    sp->close (&prop, &data);
    EXPECT_TRUE (data == nullptr);
  }
}

/**
 * @brief Run a pipeline with a custom filter library that has no invoke callback.
 * @details This uses a library of its own: the pipeline releases the sub-plugin when
 *          its own threads tear it down, so a test case that watches the very same
 *          library would depend on when that happens.
 */
TEST (tensorFilterCustomOpenFail, pipelineMissingInvoke_n)
{
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *pipeline;
  b3_lib_counters counters;
  guint init_before, exit_before;
  g_autofree gchar *path = _b3_model_path ("pipeline");

  ASSERT_TRUE (g_file_test (path, G_FILE_TEST_EXISTS));
  ASSERT_TRUE (_b3_counters_open (&counters, path, "pipeline"));
  init_before = *counters.init_count;
  exit_before = *counters.exit_count;

  pipeline = g_strdup_printf ("videotestsrc num-buffers=3 ! videoconvert ! "
                              "video/x-raw,width=16,height=16,format=RGB,framerate=10/1 ! "
                              "tensor_converter ! tensor_filter framework=custom model=%s ! "
                              "tensor_sink sync=false",
      path);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (err == nullptr);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  /* The pipeline keeps retrying the refused open until it is stopped. */
  setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  EXPECT_GE (*counters.init_count - init_before, 1U)
      << "the pipeline did not reach the custom filter, so it failed for another reason";
  EXPECT_EQ (*counters.exit_count - exit_before, *counters.init_count - init_before)
      << "an open the pipeline retried did not release what initfunc returned";

  gst_object_unref (gstpipe);
  g_free (pipeline);
  _b3_counters_close (&counters);
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
