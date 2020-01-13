/**
 * @file        unittest_tizen_EXT_NAME.cc
 * @date        19 Dec 2019
 * @brief       Failure Unit tests for tensor filter extension (EXT_NAME).
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>        /* GStatBuf */
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

/**
 * @brief Test EXT_NICK_NAME subplugin existence.
 */
TEST (nnstreamer_EXT_NICK_NAME_basic_functions, check_existence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("EXT_NAME");
  EXPECT_NE (sp, (void *) NULL);
}

/**
 * @brief Test EXT_NICK_NAME subplugin with failing open/close (no model file)
 */
TEST (nnstreamer_EXT_NICK_NAME_basic_functions, open_close_00_n)
{
  int ret;
  const gchar *model_files[] = {
    "null.EXT_NICK_NAME", NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "EXT_NAME",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };
  void *data = NULL;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("EXT_NAME");
  EXPECT_NE (sp, (void *) NULL);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Get model file after validation checks
 * @returns model file path, NULL on error
 * @note caller has to be free the returned model file path
 */
static
gchar ** get_model_files ()
{
  gchar *model_file, *model_filepath;
  gchar *model_path;
  const gchar *model_filenames = "MODEL_FILE";
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar **model_files, **model_files_iterator;
  gchar *dirname;

  if (root_path == NULL)
    root_path = "..";

  model_path = g_build_filename (root_path, "tests", "test_models", "models",
      NULL);

  model_files = g_strsplit (model_filenames, ",", 0);
  if (model_files != NULL) {
    model_files_iterator = model_files;
    for (model_file = *model_files_iterator; model_file != NULL;
        model_file = *++model_files_iterator) {

      /** If input is already path, then dont add path */
      dirname = g_path_get_dirname (model_file);
      if (g_strcmp0 (dirname, ".") != 0)
        model_filepath = g_strdup (model_file);
      else
        model_filepath = g_build_filename (model_path, model_file, NULL);
      g_free (dirname);

      if (!g_file_test (model_filepath, G_FILE_TEST_EXISTS)) {
        g_free (model_filepath);
        g_free (model_path);
        g_strfreev (model_files);
        model_files = NULL;
        goto ret;
      }

      g_free (*model_files_iterator);
      *model_files_iterator = model_filepath;
    }

    g_message ("%s\n", *model_files);
  }

  g_free (model_path);

ret:
  return model_files;
}

/**
 * @brief Test EXT_NICK_NAME subplugin with successful open/close
 */
TEST (nnstreamer_EXT_NICK_NAME_basic_functions, open_close_01_n)
{
  int ret;
  void *data = NULL;
  gchar **model_files;

  model_files = get_model_files ();
  ASSERT_TRUE (model_files != nullptr);

  GstTensorFilterProperties prop = {
    .fwname = "EXT_NAME",
    .fw_opened = 0,
    .model_files = const_cast<const char **>(model_files),
    .num_models = (int) g_strv_length (model_files),
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("EXT_NAME");
  EXPECT_NE (sp, (void *) NULL);

  /** close without open, should not crash */
  sp->close (&prop, &data);

  /** open and close successfully */
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /** close twice, should not crash */
  sp->close (&prop, &data);
  g_strfreev (model_files);
}

/**
 * @brief Get input/output dimensions with EXT_NICK_NAME subplugin
 */
TEST (nnstreamer_EXT_NICK_NAME_basic_functions, get_dimension_fail_n)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo res;
  gchar **model_files;

  model_files = get_model_files ();
  ASSERT_TRUE (model_files != nullptr);

  GstTensorFilterProperties prop = {
    .fwname = "EXT_NAME",
    .fw_opened = 0,
    .model_files = const_cast<const char **>(model_files),
    .num_models = (int) g_strv_length (model_files),
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("EXT_NAME");
  EXPECT_NE (sp, (void *) NULL);

  /** get input/output dimension without open */
  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_NE (ret, 0);
  ret = sp->getOutputDimension (&prop, &data, &res);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** get input/output dimension unsuccessfully */
  ret = sp->getInputDimension (&prop, &data, NULL);
  EXPECT_NE (ret, 0);

  ret = sp->getOutputDimension (&prop, &data, NULL);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);
  g_strfreev (model_files);
}

/**
 * @brief Get input/output dimensions with EXT_NICK_NAME subplugin
 */
TEST (nnstreamer_EXT_NICK_NAME_basic_functions, get_dimension)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo res;
  gchar **model_files;

  model_files = get_model_files ();
  ASSERT_TRUE (model_files != nullptr);

  GstTensorFilterProperties prop = {
    .fwname = "EXT_NAME",
    .fw_opened = 0,
    .model_files = const_cast<const char **>(model_files),
    .num_models = (int) g_strv_length (model_files),
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("EXT_NAME");
  EXPECT_NE (sp, (void *) NULL);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** get input/output dimension successfully */
  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  ret = sp->getOutputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  sp->close (&prop, &data);
  g_strfreev (model_files);
}

/**
 * @brief Test EXT_NICK_NAME subplugin with unsuccessful invoke
 */
TEST (nnstreamer_EXT_NICK_NAME_basic_functions, invoke_fail_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar **model_files;

  model_files = get_model_files ();
  ASSERT_TRUE (model_files != nullptr);

  GstTensorFilterProperties prop = {
    .fwname = "EXT_NAME",
    .fw_opened = 0,
    .model_files = const_cast<const char **>(model_files),
    .num_models = (int) g_strv_length (model_files),
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("EXT_NAME");
  EXPECT_NE (sp, (void *) NULL);

  /** invoke without open */
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  input.type = output.type = _NNS_FLOAT32;
  input.size = output.size = 1 * gst_tensor_get_element_size (output.type);
  input.data = g_malloc(input.size);
  output.data = g_malloc(output.size);

  /** invoke unsuccessful */
  ret = sp->invoke_NN (&prop, &data, NULL, NULL);
  EXPECT_NE (ret, 0);
  ret = sp->invoke_NN (&prop, &data, &input, NULL);
  EXPECT_NE (ret, 0);
  ret = sp->invoke_NN (&prop, &data, NULL, &output);
  EXPECT_NE (ret, 0);

  g_free (input.data);
  g_free (output.data);

  sp->close (&prop, &data);
  g_strfreev (model_files);
}

/**
 * @brief Test EXT_NICK_NAME subplugin with successful invoke
 */
TEST (nnstreamer_EXT_NICK_NAME_basic_functions, invoke)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar **model_files;
  GstTensorsInfo res;
  int num_inputs, num_outputs;

  model_files = get_model_files ();
  ASSERT_TRUE (model_files != nullptr);

  GstTensorFilterProperties prop = {
    .fwname = "EXT_NAME",
    .fw_opened = 0,
    .model_files = const_cast<const char **>(model_files),
    .num_models = (int) g_strv_length (model_files),
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("EXT_NAME");
  EXPECT_NE (sp, (void *) NULL);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  ret = sp->getOutputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);
  output.size = gst_tensor_info_get_size (&res.info[0]);
  output.type = res.info[0].type;
  num_outputs = res.num_tensors;

  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);
  input.size = gst_tensor_info_get_size (&res.info[0]);
  input.type = res.info[0].type;
  num_inputs = res.num_tensors;

  input.data = g_malloc(input.size);
  output.data = g_malloc(output.size);

  /** should never crash */
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  /** should be successful for single input/output case */
  if (num_inputs == 1 && num_outputs == 1) {
    EXPECT_EQ (ret, 0);
  }

  g_free (input.data);
  g_free (output.data);

  sp->close (&prop, &data);
  g_strfreev (model_files);
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

    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("Catched exception, GTest failed.");
  }

  return result;
}
