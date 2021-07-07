/**
 * @file        unittest_converter.cc
 * @date        18 Mar 2021
 * @brief       Unit test for tensor_converter
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include <tensor_converter_custom.h>
#include <flatbuffers/flexbuffers.h>
#include <gst/app/gstappsrc.h>
#include <nnstreamer_plugin_api_converter.h>

#define TEST_TIMEOUT_MS (10000U)

/**
 * @brief custom callback function
 */
static GstBuffer *
tensor_converter_custom_cb (GstBuffer *in_buf,
    void *data, GstTensorsConfig *config) {
  GstMemory *in_mem, *out_mem;
  GstBuffer *out_buf = NULL;
  GstMapInfo in_info;
  guint mem_size;
  gpointer mem_data;
  guint *received = (guint *) data;

  if (!in_buf || !config)
    return NULL;

  if (received)
    *received = *received + 1;

  in_mem = gst_buffer_peek_memory (in_buf, 0);
  if (!gst_memory_map (in_mem, &in_info, GST_MAP_READ)) {
    ml_loge ("Cannot map input memory / tensor_converter::flexbuf.\n");
    return NULL;
  }
  flexbuffers::Map tensors = flexbuffers::GetRoot (in_info.data, in_info.size).AsMap ();
  config->info.num_tensors = tensors["num_tensors"].AsUInt32 ();

  if (config->info.num_tensors > NNS_TENSOR_SIZE_LIMIT) {
    nns_loge ("The number of tensors is limited to %d", NNS_TENSOR_SIZE_LIMIT);
    goto done;
  }
  config->rate_n = tensors["rate_n"].AsInt32 ();
  config->rate_d = tensors["rate_d"].AsInt32 ();

  out_buf = gst_buffer_new ();
  for (guint i = 0; i < config->info.num_tensors; i++) {
    gchar * tensor_key = g_strdup_printf ("tensor_%d", i);
    flexbuffers::Vector tensor = tensors[tensor_key].AsVector ();
    config->info.info[i].name = g_strdup (tensor[0].AsString ().c_str ());
    config->info.info[i].type = (tensor_type) tensor[1].AsInt32 ();

    flexbuffers::TypedVector dim = tensor[2].AsTypedVector ();
    for (guint j = 0; j < NNS_TENSOR_RANK_LIMIT; j++) {
      config->info.info[i].dimension[j] = dim[j].AsInt32 ();
    }
    flexbuffers::Blob tensor_data = tensor[3].AsBlob ();
    mem_size = tensor_data.size ();
    mem_data = g_memdup (tensor_data.data (), mem_size);

    out_mem = gst_memory_new_wrapped ((GstMemoryFlags) 0, mem_data, mem_size,
        0, mem_size, mem_data, g_free);

    gst_buffer_append_memory (out_buf, out_mem);
    g_free (tensor_key);
  }

  /** copy timestamps */
  gst_buffer_copy_into (
      out_buf, in_buf, (GstBufferCopyFlags)GST_BUFFER_COPY_METADATA, 0, -1);
done:
  gst_memory_unmap (in_mem, &in_info);

  return out_buf;
}

/**
 * @brief Test behavior: custom callback
 */
TEST (tensorConverterCustom, normal0)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1, len2;
  char *tmp_tensor_raw = getTempFilename ();
  char *tmp_flex_raw = getTempFilename ();
  char *tmp_flex_to_tensor = getTempFilename ();
  guint *received = (guint *) g_malloc0 (sizeof (guint));

  EXPECT_NE (tmp_tensor_raw, nullptr);
  EXPECT_NE (tmp_flex_raw, nullptr);
  EXPECT_NE (tmp_flex_to_tensor, nullptr);
  EXPECT_NE (received, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=12 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=320,height=240 ! tensor_converter ! tee name=t "
      "t. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_decoder mode=flexbuf ! "
      "filesink location=%s buffer-mode=unbuffered sync=false async=false ",
      tmp_tensor_raw, tmp_flex_raw);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, TEST_TIMEOUT_MS), 0);
  g_usleep (1000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, TEST_TIMEOUT_MS), 0);
  g_usleep (100000);

  gst_object_unref (pipeline);
  g_free (str_pipeline);

  str_pipeline = g_strdup_printf (
      "filesrc location=%s blocksize=-1 ! application/octet-stream ! tensor_converter mode=custom-code:tconv ! "
      "filesink location=%s buffer-mode=unbuffered sync=false async=false ",
      tmp_flex_raw, tmp_flex_to_tensor);

  EXPECT_EQ (0, nnstreamer_converter_custom_register ("tconv", tensor_converter_custom_cb, received));

  pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, TEST_TIMEOUT_MS), 0);
  g_usleep (1000000);

  EXPECT_EQ (1U, *received);
  _wait_pipeline_save_files (tmp_tensor_raw, content1, len1, 230400, TEST_TIMEOUT_MS);
  _wait_pipeline_save_files (tmp_flex_to_tensor, content2, len2, 230400, TEST_TIMEOUT_MS);
  EXPECT_EQ (len1, len2);
  EXPECT_EQ (memcmp (content1, content2, 230400), 0);
  g_free (content1);
  g_free (content2);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, TEST_TIMEOUT_MS), 0);
  g_usleep (100000);

  EXPECT_EQ (0, nnstreamer_converter_custom_unregister ("tconv"));

  gst_object_unref (pipeline);
  g_free (str_pipeline);
  g_free (tmp_tensor_raw);
  g_free (tmp_flex_raw);
  g_free (tmp_flex_to_tensor);
  g_free (received);
}

/**
 * @brief Register custom callback with NULL parameter
 */
TEST (tensorConverterCustom, invalidParam0_n)
{
  EXPECT_NE (0, nnstreamer_converter_custom_register (NULL, tensor_converter_custom_cb, NULL));
  EXPECT_NE (0, nnstreamer_converter_custom_register ("tconv", NULL, NULL));
}

/**
 * @brief Register custom callback twice with same name
 */
TEST (tensorConverterCustom, invalidParam1_n)
{
  EXPECT_EQ (0, nnstreamer_converter_custom_register ("tconv", tensor_converter_custom_cb, NULL));
  EXPECT_NE (0, nnstreamer_converter_custom_register ("tconv", tensor_converter_custom_cb, NULL));
  EXPECT_EQ (0, nnstreamer_converter_custom_unregister ("tconv"));
}

/**
 * @brief Unregister custom callback with NULL parameter
 */
TEST (tensorConverterCustom, invalidParam2_n)
{
  EXPECT_NE (0, nnstreamer_converter_custom_unregister (NULL));
}

/**
 * @brief Unregister custom callback which is not registered
 */
TEST (tensorConverterCustom, invalidParam3_n)
{
  EXPECT_NE (0, nnstreamer_converter_custom_unregister ("tconv"));
}

/** @brief tensor converter plugin's query caps callback */
static GstCaps *
conv_query_caps (const GstTensorsConfig *config)
{
  return NULL;
}

/** @brief tensor converter plugin's get out caps callback */
static gboolean
conv_get_out_config (const GstCaps *in_cap, GstTensorsConfig *config)
{
  return TRUE;
}

/** @brief tensor converter plugin's convert callback
 */
static GstBuffer *
conv_convert (GstBuffer *in_buf, GstTensorsConfig *config, void *priv_data)
{
  return NULL;
}

/**
 * @brief Get default external converter
 */
static NNStreamerExternalConverter *
get_default_external_converter (const gchar *name)
{
  NNStreamerExternalConverter *sub = g_try_new0 (NNStreamerExternalConverter, 1);
  g_assert (sub);

  sub->name = (char *) g_strdup (name);
  sub->query_caps = conv_query_caps;
  sub->get_out_config = conv_get_out_config;
  sub->convert = conv_convert;

  return sub;
}

/**
 * @brief Free converter subplugin
 */
static void
free_default_external_converter (NNStreamerExternalConverter *sub)
{
  g_free ((char *)sub->name);
  g_free (sub);
}

/**
 * @brief Test for plugin registration
 */
TEST (tensorConverter, subpluginNoraml)
{
  NNStreamerExternalConverter *sub = get_default_external_converter ("mode");

  EXPECT_TRUE (registerExternalConverter (sub));

  unregisterExternalConverter ("mode");
  free_default_external_converter (sub);
}

/**
 * @brief Test for plugin registration with invalid param
 */
TEST (tensorConverter, subpluginInvalidParam0_n)
{
  EXPECT_FALSE (registerExternalConverter (NULL));
}

/**
 * @brief Test for plugin registration with invalid param
 */
TEST (tensorConverter, subpluginInvalidParam1_n)
{
  NNStreamerExternalConverter *sub = get_default_external_converter (NULL);

  EXPECT_FALSE (registerExternalConverter (sub));

  free_default_external_converter (sub);
}

/**
 * @brief Test for plugin registration with invalid param
 */
TEST (tensorConverter, subpluginInvalidParam2_n)
{
  NNStreamerExternalConverter *sub = get_default_external_converter ("mode");

  sub->query_caps = NULL;
  EXPECT_FALSE (registerExternalConverter (sub));

  free_default_external_converter (sub);
}

/**
 * @brief Test for plugin registration with invalid param
 */
TEST (tensorConverter, subpluginInvalidParam3_n)
{
  NNStreamerExternalConverter *sub = get_default_external_converter ("mode");

  sub->get_out_config = NULL;
  EXPECT_FALSE (registerExternalConverter (sub));

  free_default_external_converter (sub);
}

/**
 * @brief Test for plugin registration with invalid param
 */
TEST (tensorConverter, subpluginInvalidParam4_n)
{
  NNStreamerExternalConverter *sub = get_default_external_converter ("mode");

  sub->convert = NULL;
  EXPECT_FALSE (registerExternalConverter (sub));

  free_default_external_converter (sub);
}

/**
 * @brief Test for plugin registration with invalid param
 */
TEST (tensorConverter, subpluginInvalidParam5_n)
{
  NNStreamerExternalConverter *sub = get_default_external_converter ("any");

  EXPECT_FALSE (registerExternalConverter (sub));

  free_default_external_converter (sub);
}

/**
 * @brief Test for plugin registration with invalid param
 */
TEST (tensorConverter, subpluginFindInvalidParam_n)
{
  EXPECT_FALSE (nnstreamer_converter_find (NULL));
}

/**
 * @brief Test data for tensor_conveter::flexbuf (dimension 24:1:1:1)
 */
const gint _test_frames1[24]
    = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
            1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124};

/**
 * @brief Test data for tensor_conveter::flexbuf  (dimension 48:1:1:1)
 */
const gint _test_frames2[48]
    = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
            1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
            1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212,
            1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224 };

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  (*(guint *)user_data)++;
}

/**
 * @brief Test for dynamic dimension of the custom converter
 */
TEST (tensorConverterPython, dynamicDimension)
{
  GstBuffer *buf_0, *buf_1, *buf_2;
  GstElement *appsrc_handle, *sink_handle;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;
  GstPad *sink_pad;
  GstCaps *caps;
  GstStructure *structure;
  GstTensorsConfig config;
  guint *data_received = NULL;

  /** supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "custom_converter_test.py", NULL);

  gchar *str_pipeline = g_strdup_printf ("appsrc name=srcx ! application/octet-stream ! tensor_converter silent=false mode=custom-script:%s ! tensor_sink name=sinkx async=false", test_model);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (test_model);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  appsrc_handle = gst_bin_get_by_name (GST_BIN (pipeline), "srcx");
  EXPECT_NE (appsrc_handle, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  data_received = (guint *) g_malloc0 (sizeof (guint));
  g_signal_connect (sink_handle, "new-data", (GCallback)new_data_cb, data_received);

  buf_0 = gst_buffer_new_wrapped (g_memdup (_test_frames1, 96), 96);
  buf_1 = gst_buffer_new_wrapped (g_memdup (_test_frames2, 192), 192);
  buf_2 = gst_buffer_copy (buf_0);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, TEST_TIMEOUT_MS), 0);
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_0), GST_FLOW_OK);
  EXPECT_TRUE (wait_pipeline_process_buffers (data_received, 1, TEST_TIMEOUT_MS));

  sink_pad = gst_element_get_static_pad (sink_handle, "sink");
  EXPECT_NE (sink_pad, nullptr);
  caps = gst_pad_get_current_caps (sink_pad);
  EXPECT_NE (caps, nullptr);
  structure = gst_caps_get_structure (caps, 0);
  EXPECT_NE (structure, nullptr);
  gst_tensors_config_from_structure (&config, structure);
  EXPECT_EQ (1U, config.info.num_tensors);
  EXPECT_EQ (24U, config.info.info[0].dimension[0]);
  EXPECT_EQ (1U, config.info.info[0].dimension[1]);
  EXPECT_EQ (1U, config.info.info[0].dimension[2]);
  EXPECT_EQ (1U, config.info.info[0].dimension[3]);
  gst_tensors_config_free (&config);
  gst_caps_unref (caps);

  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_1), GST_FLOW_OK);
  EXPECT_TRUE (wait_pipeline_process_buffers (data_received, 2, TEST_TIMEOUT_MS));

  caps = gst_pad_get_current_caps (sink_pad);
  EXPECT_NE (caps, nullptr);
  structure = gst_caps_get_structure (caps, 0);
  EXPECT_NE (structure, nullptr);
  gst_tensors_config_from_structure (&config, structure);
  EXPECT_EQ (1U, config.info.num_tensors);
  EXPECT_EQ (48U, config.info.info[0].dimension[0]);
  EXPECT_EQ (1U, config.info.info[0].dimension[1]);
  EXPECT_EQ (1U, config.info.info[0].dimension[2]);
  EXPECT_EQ (1U, config.info.info[0].dimension[3]);
  gst_tensors_config_free (&config);
  gst_caps_unref (caps);

  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_2), GST_FLOW_OK);
  EXPECT_TRUE (wait_pipeline_process_buffers (data_received, 3, TEST_TIMEOUT_MS));

  sink_pad = gst_element_get_static_pad (sink_handle, "sink");
  EXPECT_NE (sink_pad, nullptr);
  caps = gst_pad_get_current_caps (sink_pad);
  EXPECT_NE (caps, nullptr);
  structure = gst_caps_get_structure (caps, 0);
  EXPECT_NE (structure, nullptr);
  gst_tensors_config_from_structure (&config, structure);
  EXPECT_EQ (1U, config.info.num_tensors);
  EXPECT_EQ (24U, config.info.info[0].dimension[0]);
  EXPECT_EQ (1U, config.info.info[0].dimension[1]);
  EXPECT_EQ (1U, config.info.info[0].dimension[2]);
  EXPECT_EQ (1U, config.info.info[0].dimension[3]);
  gst_tensors_config_free (&config);
  gst_caps_unref (caps);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, TEST_TIMEOUT_MS), 0);
  g_usleep (100000);

  EXPECT_EQ (3U, *data_received);
  g_free (data_received);
  gst_object_unref (sink_handle);
  gst_object_unref (appsrc_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb_json (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  guint num_tensors;
  GstMapInfo info_res;
  gboolean mapped;
  GstMemory *mem_res;
  gchar * output;
  (*(guint *)user_data)++;

  num_tensors = gst_buffer_n_memory (buffer);
  EXPECT_EQ (2U, num_tensors);

  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);
  output = (gchar *)info_res.data;
  EXPECT_STREQ ("string_example", output);
  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);
}

/**
 * @brief Test for python json parser of the custom converter
 */
TEST (tensorConverterPython, jsonParser)
{
  GstElement *sink_handle;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model, *data_file;
  guint *data_received = NULL;
  /** supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "custom_converter_json.py", NULL);
  data_file = g_build_filename (root_path, "tests", "test_models", "data",
      "example.json", NULL);

  gchar *str_pipeline = g_strdup_printf ("filesrc location=%s ! application/octet-stream ! tensor_converter silent=false mode=custom-script:%s ! tensor_sink name=sinkx", data_file, test_model);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (data_file);
  g_free (test_model);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  data_received = (guint *) g_malloc0 (sizeof (guint));
  g_signal_connect (sink_handle, "new-data", (GCallback)new_data_cb_json, data_received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, TEST_TIMEOUT_MS), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, TEST_TIMEOUT_MS), 0);
  g_usleep (100000);

  EXPECT_EQ (1U, *data_received);

  g_free (data_received);
  gst_object_unref (sink_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Test for python custom converter with invalid param
 */
TEST (tensorConverterPython, openTwice)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const NNStreamerExternalConverter *ex;
  void *py_core = NULL;
  gchar *test_model1, *test_model2;

  /** supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  ex = nnstreamer_converter_find ("python3");
  ASSERT_NE (nullptr, ex);

  test_model1 = g_build_filename (root_path, "tests", "test_models", "models",
      "custom_converter_test.py", NULL);
  test_model2 = g_build_filename (root_path, "tests", "test_models", "models",
      "custom_converter_json.py", NULL);

  EXPECT_EQ (0, ex->open (test_model1, &py_core));
  /** Open with same python script */
  EXPECT_EQ (1, ex->open (test_model1, &py_core));
  /** Open with different python script */
  EXPECT_EQ (0, ex->open (test_model2, &py_core));

  ex->close (&py_core);
  g_free (test_model1);
  g_free (test_model2);
}

/**
 * @brief Test for python custom converter with invalid param
 */
TEST (tensorConverterPython, invalidParam_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model, *data_file;
  GstStateChangeReturn ret;

  /** supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "NOT_EXIST.py", NULL);
  data_file = g_build_filename (root_path, "tests", "test_models", "data",
      "example.json", NULL);

  gchar *str_pipeline = g_strdup_printf ("filesrc location=%s blocksize=-1 ! application/octet-stream ! tensor_converter silent=false mode=custom-script:%s ! tensor_sink name=sinkx", data_file, test_model);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (test_model);
  g_free (data_file);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  g_usleep (100000);
  ret = gst_element_get_state (pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_TRUE (ret == GST_STATE_CHANGE_FAILURE);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for python custom converter with invalid param
 */
TEST (tensorConverterPython, invalidParam2_n)
{
  const NNStreamerExternalConverter *ex;
  GstTensorsConfig config;

  gst_tensors_config_init (&config);
  ex = nnstreamer_converter_find ("python3");
  ASSERT_NE (nullptr, ex);

  EXPECT_EQ (FALSE, ex->get_out_config (NULL, &config));
}

/**
 * @brief Test for python custom converter with invalid param
 */
TEST (tensorConverterPython, invalidParam3_n)
{
  const NNStreamerExternalConverter *ex;
  GstCaps *caps;

  caps = gst_caps_from_string ("application/octet-stream");
  ex = nnstreamer_converter_find ("python3");
  ASSERT_NE (nullptr, ex);

  EXPECT_EQ (FALSE, ex->get_out_config (caps, NULL));
  g_usleep (100000);
  gst_caps_unref (caps);
}

/**
 * @brief Test for python custom converter with invalid param
 */
TEST (tensorConverterPython, invalidParam4_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const NNStreamerExternalConverter *ex;
  void *py_core = NULL;
  GstTensorsConfig config;
  gchar *test_model;

  gst_tensors_config_init (&config);
  /** supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  ex = nnstreamer_converter_find ("python3");
  ASSERT_NE (nullptr, ex);

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "custom_converter_test.py", NULL);

  EXPECT_EQ (0, ex->open (test_model, &py_core));
  EXPECT_FALSE (ex->convert (NULL, &config, py_core));

  ex->close (&py_core);
  g_free (test_model);
}

/**
 * @brief Test for python custom converter with invalid param
 */
TEST (tensorConverterPython, invalidParam5_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const NNStreamerExternalConverter *ex;
  void *py_core = NULL;
  gchar *test_model;
  GstBuffer *buf;

  /** supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  ex = nnstreamer_converter_find ("python3");
  ASSERT_NE (nullptr, ex);

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "custom_converter_test.py", NULL);
  buf = gst_buffer_new ();
  EXPECT_EQ (0, ex->open (test_model, &py_core));
  EXPECT_FALSE (ex->convert (buf, NULL, py_core));

  ex->close (&py_core);
  g_free (test_model);
  gst_buffer_unref (buf);
}

/**
 * @brief Test for python custom converter with invalid param
 */
TEST (tensorConverterPython, invalidParam6_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const NNStreamerExternalConverter *ex;
  void *py_core = NULL;
  gchar *test_model;

  /** supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  ex = nnstreamer_converter_find ("python3");
  ASSERT_NE (nullptr, ex);

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "invalid_class_custom_converter.py", NULL);
  EXPECT_NE (0, ex->open (test_model, &py_core));

  g_free (test_model);
}

/**
 * @brief Main GTest
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
