/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_lua.cc
 * @date    14 Jun 2021
 * @brief   Unit test for Lua tensor filter sub-plugin
 * @author  Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <unittest_util.h>

#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>

/**
 * @brief Set tensor filter properties
 */
static void
_SetFilterProp (GstTensorFilterProperties *prop, const gchar *name, const gchar **models)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = name;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = g_strv_length ((gchar **) models);
}

/**
 * @brief Simple lua model
 */
static const char *simple_lua_script = R""""(
inputTensorsInfo = {
  num = 2,
  dim = {{3, 100, 100, 1}, {3, 24, 24, 1},},
  type = {'uint8', 'uint8',}
}

outputTensorsInfo = {
  num = 2,
  dim = {{3, 100, 100, 1}, {2, 1, 1, 1},},
  type = {'uint8', 'float32',}
}

function nnstreamer_invoke()
  input = input_tensor(1) --[[ get the first input tensor --]]
  output = output_tensor(1) --[[ get the first output tensor --]]

  for i=1,3*100*100*1 do
    output[i] = input[i]
  end

  input = input_tensor(2) --[[ get the second input tensor --]]
  output = output_tensor(2) --[[ get the second output tensor --]]

  for i=1,2 do
    output[i] = i * 11
  end

end
)"""";


/**
 * @brief Signal to validate new output data
 */
static void
check_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  gfloat *output;
  UNUSED (element);
  UNUSED (user_data);

  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);
  output = (gfloat *) info_res.data;

  for (guint i = 1; i <= 2; i++) {
    EXPECT_EQ (i * 11, output[i - 1]);
  }
}

/**
 * @brief Negative test case with invalid model file path
 */
TEST (nnstreamerFilterLua, openClose00_n)
{
  int ret;
  void *data = NULL;
  const gchar *model_files[] = {
    "some/invalid/model/path.lua",
    NULL,
  };
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);

  _SetFilterProp (&prop, "lua", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Negative test case with invalid script
 */
TEST (nnstreamerFilterLua, openClose01_n)
{
  int ret;
  void *data = NULL;
  const char *invalid_lua_script = R""""(
invalid LUA script...
)"""";
  const gchar *model_files[] = {
    invalid_lua_script,
    NULL,
  };
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);

  _SetFilterProp (&prop, "lua", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Negative test case with invalid script: invalid TensorsInfo
 */
TEST (nnstreamerFilterLua, openClose02_n)
{
  int ret;
  void *data = NULL;
  const char *invalid_lua_script = R""""(
inputTensorsInfo = {
  num = 2,
  dim = {{3, 100, 100, 1}, }, --[[ Should provide 2-length dim --]]
  type = {'uint8', } --[[ Should provide 2-length type --]]
}
outputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
function nnstreamer_invoke()
  input = input_tensor(1)
  output = output_tensor(1)

  for i=1,3*100*100*1 do
    output[i] = input[i]
  end

end
)"""";
  const gchar *model_files[] = {
    invalid_lua_script,
    NULL,
  };
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);

  _SetFilterProp (&prop, "lua", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Negative test case with invalid script: double quotes in script
 */
TEST (nnstreamerFilterLua, openClose03_n)
{
  int ret;
  void *data = NULL;
  const char *invalid_lua_script = R""""(
inputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {\"uint8\", } --[[ In script mode, should not use double quotes --]]
}
outputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
function nnstreamer_invoke()
  input = input_tensor(1)
  output = output_tensor(1)

  for i=1,3*100*100*1 do
    output[i] = input[i]
  end

end
)"""";
  const gchar *model_files[] = {
    invalid_lua_script,
    NULL,
  };
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);

  _SetFilterProp (&prop, "lua", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Positive case with open/close (script mode)
 */
TEST (nnstreamerFilterLua, openClose04)
{
  int ret;
  void *data = NULL;
  const char *lua_script = R""""(
inputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
outputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
function nnstreamer_invoke()
  input = input_tensor(1)
  output = output_tensor(1)

  for i=1,3*100*100*1 do
    output[i] = input[i]
  end

end
)"""";
  const gchar *model_files[] = {
    lua_script,
    NULL,
  };
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);

  _SetFilterProp (&prop, "lua", model_files);
  /* close before open */
  sp->close (&prop, &data);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /* double close */
  sp->close (&prop, &data);;
}


/**
 * @brief Positive case with open/close (file mode)
 */
TEST (nnstreamerFilterLua, openClose05)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.lua", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  /* close before open */
  sp->close (&prop, &data);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /* double close */
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Positive case with successful getModelInfo
 */
TEST (nnstreamerFilterLua, getModelInfo00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.lua", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], 3U);
  EXPECT_EQ (in_info.info[0].dimension[1], 640U);
  EXPECT_EQ (in_info.info[0].dimension[2], 480U);
  EXPECT_EQ (in_info.info[0].dimension[3], 1U);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], 3U);
  EXPECT_EQ (out_info.info[0].dimension[1], 640U);
  EXPECT_EQ (out_info.info[0].dimension[2], 480U);
  EXPECT_EQ (out_info.info[0].dimension[3], 1U);
  EXPECT_EQ (out_info.info[0].type, _NNS_UINT8);

  sp->close (&prop, &data);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  g_free (model_file);
}

/**
 * @brief Negative case calling getModelInfo before open
 */
TEST (nnstreamerFilterLua, getModelInfo01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.lua", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);
  _SetFilterProp (&prop, "lua", model_files);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative case with invalid argument
 */
TEST (nnstreamerFilterLua, getModelInfo02_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.lua", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  GstTensorsInfo in_info, out_info;

  /* not supported */
  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative test case with invoke before open
 */
TEST (nnstreamerFilterLua, invoke00_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.lua", NULL);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case with invalid input/output
 */
TEST (nnstreamerFilterLua, invoke01_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;

  const gchar *model_files[] = {
    simple_lua_script,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, nullptr);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);
  ((float *) input.data)[0] = 10.0;

  /* catching assertion error */
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, NULL, &output), "");
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, &input, NULL), "");

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative test case with invalid private_data
 */
TEST (nnstreamerFilterLua, invoke02_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.lua", NULL);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, nullptr);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);
  ((float *) input.data)[0] = 10.0;

  /* unsuccessful invoke with NULL priv_data */
  ret = sp->invoke (NULL, NULL, NULL, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case with invoke for lua model
 */
TEST (nnstreamerFilterLua, invoke03)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.lua", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  output.size = input.size = sizeof (uint8_t) * 3 * 640 * 480 * 1;

  /* alloc input data without alignment */
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  memset (input.data, 0, input.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);
  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<uint8_t *> (output.data)[0], 0U);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case with invoke for lua model (script mode)
 */
TEST (nnstreamerFilterLua, invoke04)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  const char *lua_script = R""""(
inputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
outputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
function nnstreamer_invoke()
  input = input_tensor(1)
  output = output_tensor(1)

  for i=1,3*100*100*1 do
    output[i] = input[i]
  end

end
)"""";
  const gchar *model_files[] = {
    lua_script,
    NULL,
  };

  output.size = input.size = sizeof (uint8_t) * 3 * 100 * 100 * 1;

  /* alloc input data without alignment */
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  memset (input.data, 0, input.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);
  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<uint8_t *> (output.data)[0], 0U);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case with invoke for lua model: invalid index for tensor
 */
TEST (nnstreamerFilterLua, invoke05_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  const char *invalid_lua_script = R""""(
inputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
outputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
function nnstreamer_invoke()
  input = input_tensor(1)
  output = output_tensor(1)

  for i=1,3*100*100*2 do --[[ invalid index for tensor here --]]
    output[i] = input[i]
  end

end
)"""";
  const gchar *model_files[] = {
    invalid_lua_script,
    NULL,
  };

  output.size = input.size = sizeof (uint8_t) * 3 * 100 * 100 * 1;

  /* alloc input data without alignment */
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  memset (input.data, 0, input.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);
  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_NE (ret, 0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case with reload lua model file
 */
TEST (nnstreamerFilterLua, reload00)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorsInfo in_info, out_info;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.lua", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  gchar *model_file2 = g_build_filename (
      root_path, "tests", "test_models", "models", "scaler.lua", NULL);
  ASSERT_TRUE (g_file_test (model_file2, G_FILE_TEST_EXISTS));

  const gchar *model_files2[] = {
    model_file2,
    NULL,
  };

  output.size = input.size = sizeof (uint8_t) * 3 * 640 * 480 * 1;

  /* alloc input data without alignment */
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  memset (input.data, 0, input.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);
  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<uint8_t *> (output.data)[0], 0U);

  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], 3U);
  EXPECT_EQ (in_info.info[0].dimension[1], 640U);
  EXPECT_EQ (in_info.info[0].dimension[2], 480U);
  EXPECT_EQ (in_info.info[0].dimension[3], 1U);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], 3U);
  EXPECT_EQ (out_info.info[0].dimension[1], 640U);
  EXPECT_EQ (out_info.info[0].dimension[2], 480U);
  EXPECT_EQ (out_info.info[0].dimension[3], 1U);
  EXPECT_EQ (out_info.info[0].type, _NNS_UINT8);

  /** reload model */
  _SetFilterProp (&prop, "lua", model_files2);
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);
  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<uint8_t *> (output.data)[0], 0);

  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], 3U);
  EXPECT_EQ (in_info.info[0].dimension[1], 640U);
  EXPECT_EQ (in_info.info[0].dimension[2], 480U);
  EXPECT_EQ (in_info.info[0].dimension[3], 1U);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], 3U);
  EXPECT_EQ (out_info.info[0].dimension[1], 320U);
  EXPECT_EQ (out_info.info[0].dimension[2], 240U);
  EXPECT_EQ (out_info.info[0].dimension[3], 1U);
  EXPECT_EQ (out_info.info[0].type, _NNS_UINT8);


  g_free (model_file);
  g_free (model_file2);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case with reload lua script
 */
TEST (nnstreamerFilterLua, reload01)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  const char *lua_script = R""""(
inputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
outputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
function nnstreamer_invoke()
  input = input_tensor(1)
  output = output_tensor(1)

  for i=1,3*100*100*1 do
    output[i] = input[i]
  end

end
)"""";

  const gchar *model_files[] = {
    lua_script,
    NULL,
  };

  const char *lua_script2 = R""""(
inputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
outputTensorsInfo = {
  num = 1,
  dim = {{3, 100, 100, 1}, },
  type = {'uint8', }
}
function nnstreamer_invoke()
  input = input_tensor(1)
  output = output_tensor(1)

  for i=1,3*100*100*1 do
    output[i] = 77
  end

end
)"""";
  const gchar *model_files2[] = {
    lua_script2,
    NULL,
  };

  output.size = input.size = sizeof (uint8_t) * 3 * 100 * 100 * 1;

  /* alloc input data without alignment */
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  memset (input.data, 0, input.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "lua", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);
  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<uint8_t *> (output.data)[0], 0U);

  /** reload */
  _SetFilterProp (&prop, "lua", model_files2);
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);
  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<uint8_t *> (output.data)[0], 77U);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case with lua script with invalid data type
 */
TEST (nnstreamerFilterLua, dataType00_n)
{
  /**
   * Invalid data type `uint128`. Type should be one of
   * { float32, float64, int64, uint64, int32, uint32, int16, uint16, int8, uint8 }
   */
  const char *invalid_data_type = "uint128";
  int ret;
  void *data = NULL;
  gchar *model;
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);

  model = g_strdup_printf ("\
  inputTensorsInfo={num=1,dim={{1,2,2,1},},type={'%s',}} \
  outputTensorsInfo={num=1,dim={{1,2,2,1},},type={'%s',}} \
  function nnstreamer_invoke() \
  for i=1,1*2*2*1 do output_tensor(1)[i] = input_tensor(1)[i] end \
  end", invalid_data_type, invalid_data_type);

  const gchar *model_files[] = {
    model,
    NULL,
  };

  _SetFilterProp (&prop, "lua", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);

  g_free (model);
}

/**
 * @brief Positive case with lua script for all data types
 */
TEST (nnstreamerFilterLua, dataType01)
{
  int ret;
  gchar *model;
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;

  output.size = input.size = sizeof (int64_t) * 1 * 2 * 2 * 1;

  /* alloc input data without alignment */
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("lua");
  EXPECT_NE (sp, nullptr);

  for (int i = 0; i < _NNS_END; ++i) {
    tensor_type ttype = (tensor_type) i;
    model = g_strdup_printf ("\
    inputTensorsInfo={num=1,dim={{1,2,2,1},},type={'%s',}} \
    outputTensorsInfo={num=1,dim={{1,2,2,1},},type={'%s',}} \
    function nnstreamer_invoke() \
    for i=1,1*2*2*1 do output_tensor(1)[i] = input_tensor(1)[i] end \
    end", gst_tensor_get_type_string (ttype), gst_tensor_get_type_string (ttype));

    const gchar *model_files[] = {
      model,
      NULL,
    };

    _SetFilterProp (&prop, "lua", model_files);
    ret = sp->open (&prop, &data);
    EXPECT_EQ (ret, 0);
    EXPECT_NE (data, (void *) NULL);

    ret = sp->invoke (NULL, NULL, data, &input, &output);
    EXPECT_EQ (ret, 0);

    g_free (model);
  }

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case to launch gst pipeline: wrong dimension
 */
TEST (nnstreamerFilterLua, launch00_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;

  /**
   * Create a invalid pipeline the input dimension should be 3 : 24 : 24 : 1
   * Given 3 : 11 : 11 : 1
  */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! video/x-raw,width=100,height=100,format=RGB ! tensor_converter ! mux.sink_0 videotestsrc num-buffers=1 ! video/x-raw,width=11,height=11,format=RGB ! tensor_converter ! mux.sink_1 tensor_mux name=mux sync_mode=nosync ! tensor_filter framework=lua model=\"%s\" ! tensor_demux name=demux demux.src_0 ! tensor_decoder mode=direct_video ! video/x-raw,width=100,height=100,format=RGB ! videoconvert ! autovideosink demux.src_1 ! tensor_sink name=sinkx async=false sync=false",
      simple_lua_script);

  gstpipe = gst_parse_launch (pipeline, &err);
  EXPECT_NE (gstpipe, nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, NULL);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Positive case to launch gst pipeline
 */
TEST (nnstreamerFilterLua, launch01)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! video/x-raw,width=100,height=100,format=RGB ! tensor_converter ! mux.sink_0 videotestsrc num-buffers=1 ! video/x-raw,width=24,height=24,format=RGB ! tensor_converter ! mux.sink_1 tensor_mux name=mux sync_mode=nosync ! tensor_filter framework=lua model=\"%s\" ! tensor_demux name=demux demux.src_0 ! tensor_decoder mode=direct_video ! video/x-raw,width=100,height=100,format=RGB ! videoconvert ! autovideosink demux.src_1 ! tensor_sink name=sinkx async=false sync=false",
      simple_lua_script);

  gstpipe = gst_parse_launch (pipeline, &err);
  EXPECT_NE (gstpipe, nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, NULL);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT * 10), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
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
