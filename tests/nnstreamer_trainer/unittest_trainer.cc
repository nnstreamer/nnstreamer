/**
 * @file        unittest_trainer.cc
 * @date        21 Apr 2023
 * @brief       Unit test for tensor_trainer
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Hyunil Park <hyunil46.park@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <unittest_util.h>

static const gchar filename[] = "mnist.data";
static const gchar json[] = "mnist.json";
static const gchar model_config[] = "mnist.ini";

/**
 * @brief Get file path
 */
static gchar *
get_file_path (const gchar *filename)
{
  const gchar *root_path = NULL;
  gchar *file_path = NULL;

  root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  /** supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  file_path = g_build_filename (
      root_path, "tests", "test_models", "data", "datarepo", filename, NULL);

  return file_path;
}


/**
 * @brief Model training test using mnist.data (MNIST Test), model.bin is
 * created.
 *
 * framework: framework to use for training the model
 * model-config: model configuration file path. models are limited to creating
 * with configuration files. model-save-path: model save path by query in MLOps
 * num-inputs: sub-plugin supports multiple inputs, in case of MNIST, num-inputs
 * is 1. num-labels: sub-plugin supports multiple labels, in case of MNIST,
 * num-labels is 1. num-training-samples: Number of training samples, A sample
 * can consist of multiple inputs and labels in tensors(in case of MNIST, all is
 * 1), set how many samples are taken for training model.
 * num-validation-samples: num-validation-samples, A sample can consist of
 * multiple inputs and labels in tensors(in case of MNIST, all is 1), set how
 * many samples are taken for validation model. epochs : epochs are repetitions
 * of training samples and validation smaples. number of samples received for
 * model training is (num-training-samples + num-validation-samples) * epochs
 */

TEST (tensor_trainer, SetParams)
{
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *model_config_path = NULL;
  guint get_value;
  gchar *get_str;
  GstElement *tensor_trainer = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  model_config_path = get_file_path (model_config);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=1 ! "
      "tensor_trainer name=tensor_trainer framework=nntrainer model-config=%s "
      "model-save-path=model.bin num-inputs=1 num-labels=1 "
      "num-training-samples=100 num-validation-samples=100 epochs=1 ! "
      "tensor_sink",
      file_path, json_path, model_config_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  EXPECT_NE (tensor_trainer, nullptr);

  setPipelineStateSync (pipeline, GST_STATE_PAUSED, UNITTEST_STATECHANGE_TIMEOUT);

  g_object_get (tensor_trainer, "model-config", &get_str, NULL);
  EXPECT_STREQ (get_str, model_config_path);

  g_object_get (tensor_trainer, "model-save-path", &get_str, NULL);
  EXPECT_STREQ (get_str, "model.bin");

  g_object_get (tensor_trainer, "num-inputs", &get_value, NULL);
  ASSERT_EQ (get_value, 1U);

  g_object_get (tensor_trainer, "num-labels", &get_value, NULL);
  ASSERT_EQ (get_value, 1U);

  g_object_get (tensor_trainer, "num-training-samples", &get_value, NULL);
  ASSERT_EQ (get_value, 100U);

  g_object_get (tensor_trainer, "num-validation-samples", &get_value, NULL);
  ASSERT_EQ (get_value, 100U);

  g_object_get (tensor_trainer, "epochs", &get_value, NULL);
  ASSERT_EQ (get_value, 1U);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  gst_object_unref (pipeline);
  g_free (file_path);
  g_free (json_path);
  g_free (model_config_path);
}

/**
 * @brief Model training test with invalid param (framework)
 */
TEST (tensor_trainer, invalidFramework0_n)
{
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *model_config_path = NULL;
  GstElement *tensor_trainer = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  model_config_path = get_file_path (model_config);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=5 ! "
      "tensor_trainer name=tensor_trainer model-config=%s "
      "model-save-path=model.bin num-inputs=1 num-labels=1 "
      "num-training-samples=100 num-validation-samples=100 epochs=5 ! tensor_sink",
      file_path, json_path, model_config_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);

  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  g_free (model_config_path);
  ASSERT_NE (pipeline, nullptr);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  ASSERT_NE (tensor_trainer, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (tensor_trainer), "framework", NULL, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Model training test with invalid param (framework)
 */
TEST (tensor_trainer, invalidFramework1_n)
{
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *model_config_path = NULL;
  GstElement *tensor_trainer = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  model_config_path = get_file_path (model_config);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=5 ! "
      "tensor_trainer name=tensor_trainer model-config=%s "
      "model-save-path=model.bin num-inputs=1 num-labels=1 "
      "num-training-samples=100 num-validation-samples=100 epochs=5 ! tensor_sink",
      file_path, json_path, model_config_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  g_free (file_path);
  g_free (json_path);
  g_free (model_config_path);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  ASSERT_NE (tensor_trainer, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (tensor_trainer), "framework", "no_framework", NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Model training test with invalid param (model-config)
 */
TEST (tensor_trainer, invalidModelConfig0_n)
{
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  GstElement *tensor_trainer = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s"
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=5 ! "
      "tensor_trainer name=tensor_trainer framework=nntrainer"
      "model-save-path=model.bin num-inputs=1 num-labels=1 "
      "num-training-samples=100 num-validation-samples=100 epochs=5 ! tensor_sink",
      file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  ASSERT_NE (pipeline, nullptr);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  ASSERT_NE (tensor_trainer, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (tensor_trainer), "model-config", NULL, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Model training test with invalid param (model-save-path)
 */
TEST (tensor_trainer, invalidModelSavePath0_n)
{
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *model_config_path = NULL;
  GstElement *tensor_trainer = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  model_config_path = get_file_path (model_config);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=5 ! "
      "tensor_trainer name=tensor_trainer framework=nntrainer model-config=%s "
      "num-inputs=1 num-labels=1 num-training-samples=100 num-validation-samples=100 "
      "epochs=5 ! tensor_sink",
      file_path, json_path, model_config_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  g_free (model_config_path);
  ASSERT_NE (pipeline, nullptr);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  ASSERT_NE (tensor_trainer, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (tensor_trainer), "model-save-path", NULL, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Model training test with invalid param (num-training-samples)
 */
TEST (tensor_trainer, invalidModelNumTrainingSamples0_n)
{
  GstElement *tensor_trainer = NULL;
  gint invalid_value = -1;
  guint get_value;
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *model_config_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  model_config_path = get_file_path (model_config);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=5 ! "
      "tensor_trainer name=tensor_trainer framework=nntrainer model-config=%s "
      "model-save-path=model.bin num-inputs=1 num-labels=1 "
      "num-validation-samples=100 epochs=5 ! tensor_sink",
      file_path, json_path, model_config_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  g_free (model_config_path);
  ASSERT_NE (pipeline, nullptr);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  ASSERT_NE (tensor_trainer, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (tensor_trainer), "num-training-samples", invalid_value, NULL);
  /** value "-1" is out of range for property 'num-training-samples' of type
     'guint' default value is set */
  g_object_get (GST_OBJECT (tensor_trainer), "num-training-samples", &get_value, NULL);
  /* state chagne failure is expected */
  EXPECT_EQ (get_value, 0U);

  gst_object_unref (pipeline);
}

/**
 * @brief Model training test with invalid param (num-validation-samples)
 */
TEST (tensor_trainer, invalidModelNumValidationSamples0_n)
{
  GstElement *tensor_trainer = NULL;
  gint invalid_value = -1;
  guint get_value;
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *model_config_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  model_config_path = get_file_path (model_config);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=5 ! "
      "tensor_trainer name=tensor_trainer framework=nntrainer model-config=%s "
      "model-save-path=model.bin num-inputs=1 num-labels=1 "
      "num-training-samples=100 epochs=5 ! tensor_sink",
      file_path, json_path, model_config_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  g_free (model_config_path);
  ASSERT_NE (pipeline, nullptr);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  ASSERT_NE (tensor_trainer, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (tensor_trainer), "num-validation-samples", invalid_value, NULL);
  /** value "-1" is out of range for property 'num-validation-samples' of type
     'guint' default value is set */
  g_object_get (GST_OBJECT (tensor_trainer), "num-validation-samples", &get_value, NULL);
  /* state chagne failure is expected */
  EXPECT_EQ (get_value, 0U);

  gst_object_unref (pipeline);
}

/**
 * @brief Model training test with invalid param (epochs)
 */
TEST (tensor_trainer, invalidEpochs0_n)
{
  GstElement *tensor_trainer = NULL;
  gint invalid_value = -1;
  guint get_value;
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *model_config_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  model_config_path = get_file_path (model_config);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=5 ! "
      "tensor_trainer name=tensor_trainer framework=nntrainer model-config=%s "
      "model-save-path=model.bin num-inputs=1 num-labels=1 "
      "num-training-samples=100 num-validation-samples=100 ! tensor_sink",
      file_path, json_path, model_config_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  g_free (model_config_path);
  ASSERT_NE (pipeline, nullptr);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  ASSERT_NE (tensor_trainer, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (tensor_trainer), "epochs", invalid_value, NULL);
  /** value "-1" is out of range for property 'epochs' of type 'guint'
     default value is set */
  g_object_get (GST_OBJECT (tensor_trainer), "epochs", &get_value, NULL);
  /* state chagne failure is expected */
  EXPECT_EQ (get_value, 1U);

  gst_object_unref (pipeline);
}

/**
 * @brief Model training test with invalid param (num-inputs)
 */
TEST (tensor_trainer, invalidNumInputs0_n)
{
  GstElement *tensor_trainer = NULL;
  gint invalid_value = -1;
  guint get_value;
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *model_config_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  model_config_path = get_file_path (model_config);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=5 ! "
      "tensor_trainer name=tensor_trainer framework=nntrainer model-config=%s "
      "model-save-path=model.bin num-labels=1 num-training-samples=100 "
      "num-validation-samples=100 epochs=5 ! tensor_sink",
      file_path, json_path, model_config_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  g_free (model_config_path);
  ASSERT_NE (pipeline, nullptr);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  ASSERT_NE (tensor_trainer, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (tensor_trainer), "num-inputs", invalid_value, NULL);
  /** value "-1" is out of range for property 'num-inputs' of type 'guint'
     default value is set */
  g_object_get (GST_OBJECT (tensor_trainer), "num-inputs", &get_value, NULL);
  /* state chagne failure is expected */
  EXPECT_EQ (get_value, 1U);

  gst_object_unref (pipeline);
}

/**
 * @brief Model training test with invalid param (num-labels)
 */
TEST (tensor_trainer, invalidNumLabels0_n)
{
  GstElement *tensor_trainer = NULL;
  gint invalid_value = -1;
  guint get_value;
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *model_config_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  model_config_path = get_file_path (model_config);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=5 ! "
      "tensor_trainer name=tensor_trainer framework=nntrainer model-config=%s"
      "model-save-path=model.bin num-inputs=1 num-validation-samples=100 epochs=5 ! "
      "tensor_sink",
      file_path, json_path, model_config_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  g_free (model_config_path);
  ASSERT_NE (pipeline, nullptr);

  tensor_trainer = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_trainer");
  ASSERT_NE (tensor_trainer, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (tensor_trainer), "num-labels", invalid_value, NULL);
  /** value "-1" of type 'gint64' is invalid or out of range for property
     'num-labels' of type 'guint' default value is set */
  g_object_get (GST_OBJECT (tensor_trainer), "num-labels", &get_value, NULL);
  /* state chagne failure is expected */
  EXPECT_EQ (get_value, 1U);

  gst_object_unref (pipeline);
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
