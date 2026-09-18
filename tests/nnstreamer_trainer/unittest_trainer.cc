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
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_trainer.h>
#include <sys/mman.h>
#include <unistd.h>
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
 * of training samples and validation samples. number of samples received for
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
      "datareposrc location=%s json=%s "
      "start-sample-index=3 stop-sample-index=202 tensors-sequence=0,1 epochs=1 ! "
      "tensor_trainer name=tensor_trainer framework=nntrainer model-config=%s "
      "model-save-path=new_model.bin model-load-path=old_model.bin num-inputs=1 num-labels=1 "
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
  g_free (get_str);

  g_object_get (tensor_trainer, "model-save-path", &get_str, NULL);
  EXPECT_STREQ (get_str, "new_model.bin");
  g_free (get_str);

  g_object_get (tensor_trainer, "model-load-path", &get_str, NULL);
  EXPECT_STREQ (get_str, "old_model.bin");
  g_free (get_str);

  /* set nullable param */
  g_object_set (GST_OBJECT (tensor_trainer), "model-load-path", NULL, NULL);

  g_object_get (tensor_trainer, "model-load-path", &get_str, NULL);
  EXPECT_STREQ (get_str, NULL);
  g_free (get_str);

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

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
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
      "datareposrc location=%s json=%s "
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

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
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
      "datareposrc location=%s json=%s "
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

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
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
      "datareposrc location=%s json=%s"
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

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
}

/**
 * @brief Model training test with invalid param (model-config)
 */
TEST (tensor_trainer, invalidModelConfig1_n)
{
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gchar *non_existent_path = NULL;
  GstElement *tensor_trainer = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);
  non_existent_path = get_file_path ("non_existent_file.ini");

  gchar *str_pipeline = g_strdup_printf (
      "datareposrc location=%s json=%s"
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
  g_object_set (GST_OBJECT (tensor_trainer), "model-config", non_existent_path, NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_free (non_existent_path);
  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
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
      "datareposrc location=%s json=%s "
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

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
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
      "datareposrc location=%s json=%s "
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
  /* state change failure is expected */
  EXPECT_EQ (get_value, 0U);

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
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
      "datareposrc location=%s json=%s "
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
  /* state change failure is expected */
  EXPECT_EQ (get_value, 0U);

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
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
      "datareposrc location=%s json=%s "
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
  /* state change failure is expected */
  EXPECT_EQ (get_value, 1U);

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
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
      "datareposrc location=%s json=%s "
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
  /* state change failure is expected */
  EXPECT_EQ (get_value, 1U);

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
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
      "datareposrc location=%s json=%s "
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
  /* state change failure is expected */
  EXPECT_EQ (get_value, 1U);

  gst_object_unref (GST_OBJECT (tensor_trainer));
  gst_object_unref (GST_OBJECT (pipeline));
}

/**
 * @brief What the fake trainer sub-plugin has been asked to do.
 */
static struct {
  gint stop_calls; /**< number of stop() calls, one per dummy-data thread */
  gint push_calls; /**< number of push_data() calls entered */
  gint push_done; /**< number of push_data() calls returned */
  gulong first_push_delay; /**< microseconds the first push_data() sleeps */
  gsize pushed_size; /**< size of the first input tensor last pushed */
  guint8 pushed_first; /**< first byte of the first input tensor last pushed */
  gchar expect_config[512]; /**< model-config every push_data() should see */
  gboolean config_matched; /**< whether the last push_data() saw it */
  gint push_ret; /**< what push_data() returns */
} fake_stat;

/**
 * @brief Fake sub-plugin callback that creates nothing.
 */
static int
fake_trainer_create (const GstTensorTrainerFramework *,
    const GstTensorTrainerProperties *, void **)
{
  return 0;
}

/**
 * @brief Fake sub-plugin callback that destroys nothing.
 */
static int
fake_trainer_destroy (const GstTensorTrainerFramework *,
    const GstTensorTrainerProperties *, void **)
{
  return 0;
}

/**
 * @brief Fake sub-plugin callback that starts nothing.
 */
static int
fake_trainer_start (const GstTensorTrainerFramework *,
    const GstTensorTrainerProperties *, GstTensorTrainerEventNotifier *, void *)
{
  return 0;
}

/**
 * @brief Fake sub-plugin callback counting stop() calls.
 */
static int
fake_trainer_stop (const GstTensorTrainerFramework *,
    const GstTensorTrainerProperties *, void **)
{
  g_atomic_int_inc (&fake_stat.stop_calls);
  return 0;
}

/**
 * @brief Fake sub-plugin callback recording the pushed data.
 */
static int
fake_trainer_push_data (const GstTensorTrainerFramework *,
    const GstTensorTrainerProperties *prop, void *, const GstTensorMemory *input)
{
  if (g_atomic_int_add (&fake_stat.push_calls, 1) == 0 && fake_stat.first_push_delay > 0)
    g_usleep (fake_stat.first_push_delay);

  /* the properties must outlive this call, even when it runs during finalize */
  if (prop->model_config && fake_stat.expect_config[0]) {
    guint i;

    fake_stat.config_matched = TRUE;
    for (i = 0; fake_stat.expect_config[i]; i++) {
      if (prop->model_config[i] != fake_stat.expect_config[i]) {
        fake_stat.config_matched = FALSE;
        break;
      }
    }
  }

  fake_stat.pushed_size = input[0].size;
  if (input[0].data && input[0].size > 0)
    fake_stat.pushed_first = ((guint8 *) input[0].data)[0];

  g_atomic_int_inc (&fake_stat.push_done);
  return fake_stat.push_ret;
}

/**
 * @brief Fake sub-plugin callback reporting a fixed status.
 */
static int
fake_trainer_get_status (
    const GstTensorTrainerFramework *, GstTensorTrainerProperties *prop, void *)
{
  prop->training_loss = 1.0;
  return 0;
}

/**
 * @brief Fake sub-plugin callback naming the framework.
 * @note It takes the name nntrainer, the only framework tensor_trainer
 *       starts a dummy-data thread for.
 */
static int
fake_trainer_get_framework_info (const GstTensorTrainerFramework *,
    const GstTensorTrainerProperties *, void *, GstTensorTrainerFrameworkInfo *fw_info)
{
  fw_info->name = "nntrainer";
  return 0;
}

/**
 * @brief The fake trainer sub-plugin.
 */
static GstTensorTrainerFramework fake_trainer_fw = { GST_TENSOR_TRAINER_FRAMEWORK_V1,
  fake_trainer_create, fake_trainer_destroy, fake_trainer_start, fake_trainer_stop,
  fake_trainer_push_data, fake_trainer_get_status, fake_trainer_get_framework_info };

/**
 * @brief Test fixture registering the fake trainer sub-plugin.
 */
class TensorTrainerFakeFw : public ::testing::Test
{
  protected:
  /**
   * @brief Reset the counters and register the fake sub-plugin.
   */
  void SetUp () override
  {
    memset (&fake_stat, 0, sizeof (fake_stat));
    ASSERT_TRUE (nnstreamer_trainer_probe (&fake_trainer_fw));
  }

  /**
   * @brief Unregister the fake sub-plugin.
   */
  void TearDown () override
  {
    nnstreamer_trainer_exit (&fake_trainer_fw);
  }
};

/**
 * @brief Create a tensor_trainer that uses the fake sub-plugin.
 */
static GstElement *
make_fake_trainer (void)
{
  gchar *config_path = get_file_path (model_config);
  GstElement *trainer = gst_element_factory_make ("tensor_trainer", NULL);

  if (trainer)
    g_object_set (trainer, "framework", "nntrainer", "model-config",
        config_path, "model-save-path", "c31_model.bin", NULL);
  g_free (config_path);

  return trainer;
}

/**
 * @brief Wait until the fake sub-plugin has seen @a count stop() calls.
 */
static gboolean
wait_for_stop_calls (gint count)
{
  guint i;

  for (i = 0; i < 500; i++) {
    if (g_atomic_int_get (&fake_stat.stop_calls) >= count)
      return TRUE;
    g_usleep (10000);
  }

  return FALSE;
}

/**
 * @brief Pausing twice waits for the first dummy-data thread.
 *
 * The first thread is still inside push_data() when the element pauses
 * again. The element must join it before starting the second one; if the
 * handle is overwritten instead, finalize joins only the second thread and
 * the first one outlives the element.
 */
TEST_F (TensorTrainerFakeFw, pauseTwiceJoinsDummyThread)
{
  GstElement *trainer = make_fake_trainer ();
  ASSERT_NE (trainer, nullptr);

  fake_stat.first_push_delay = G_USEC_PER_SEC;

  EXPECT_EQ (gst_element_set_state (trainer, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (gst_element_set_state (trainer, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);
  ASSERT_TRUE (wait_for_stop_calls (1));

  EXPECT_EQ (gst_element_set_state (trainer, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (gst_element_set_state (trainer, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (gst_element_set_state (trainer, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (trainer);

  EXPECT_EQ (g_atomic_int_get (&fake_stat.stop_calls), 2);
  EXPECT_EQ (g_atomic_int_get (&fake_stat.push_done), 2);
}

/**
 * @brief Finalizing the element waits for a running dummy-data thread.
 *
 * The thread is inside push_data() while the element is finalized, and it
 * reads the properties there. Finalize must therefore join it before it
 * frees them; if the join stays below the frees, the sub-plugin reads
 * freed memory, which the Valgrind job reports as an error of ours.
 */
TEST_F (TensorTrainerFakeFw, finalizeJoinsDummyThread)
{
  GstElement *trainer = make_fake_trainer ();
  gchar *config_path = get_file_path (model_config);

  ASSERT_NE (trainer, nullptr);
  ASSERT_LT (strlen (config_path), sizeof (fake_stat.expect_config));

  fake_stat.first_push_delay = G_USEC_PER_SEC / 2;
  g_strlcpy (fake_stat.expect_config, config_path, sizeof (fake_stat.expect_config));

  EXPECT_EQ (gst_element_set_state (trainer, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (gst_element_set_state (trainer, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);
  ASSERT_TRUE (wait_for_stop_calls (1));

  EXPECT_EQ (gst_element_set_state (trainer, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (trainer);

  EXPECT_EQ (g_atomic_int_get (&fake_stat.stop_calls), 1);
  EXPECT_EQ (g_atomic_int_get (&fake_stat.push_done), 1);
  EXPECT_TRUE (fake_stat.config_matched);

  g_free (config_path);
}

/**
 * @brief Create a harness around a playing tensor_trainer with the fake sub-plugin.
 */
static GstHarness *
make_fake_trainer_harness (void)
{
  GstHarness *h;
  GstElement *trainer = make_fake_trainer ();

  if (!trainer)
    return NULL;

  h = gst_harness_new_with_element (trainer, "sink", "src");
  gst_object_unref (trainer);

  return h;
}

/**
 * @brief Build static caps with @a num uint8 tensors of one element each.
 */
static gchar *
make_static_caps_string (guint num, gint rate_n)
{
  GString *caps = g_string_new (NULL);
  guint i;

  g_string_append_printf (caps,
      "other/tensors,format=static,num_tensors=%u,framerate=%d/1,dimensions=(string)\"",
      num, rate_n);
  for (i = 0; i < num; i++)
    g_string_append (caps, i == 0 ? "1:1:1:1" : ".1:1:1:1");
  g_string_append (caps, "\",types=(string)\"");
  for (i = 0; i < num; i++)
    g_string_append (caps, i == 0 ? "uint8" : ",uint8");
  g_string_append (caps, "\"");

  return g_string_free (caps, FALSE);
}

/**
 * @brief Renegotiating caps with more than 16 tensors frees the previous config.
 *
 * Each GstTensorsInfo with more than 16 tensors owns a heap block for the
 * extra tensors. The copy in the trainer properties, the negotiated input
 * config and the last copy at finalize must all be released; the leak
 * gate of the valgrind CI job is what fails if one is dropped.
 */
TEST_F (TensorTrainerFakeFw, renegotiateManyTensors)
{
  GstHarness *h = make_fake_trainer_harness ();
  GstCaps *caps;
  GstStructure *s;
  gchar *caps_str;
  gint rate_n = 0, rate_d = 0;

  ASSERT_NE (h, nullptr);

  caps_str = make_static_caps_string (17, 10);
  gst_harness_set_src_caps_str (h, caps_str);
  g_free (caps_str);

  caps_str = make_static_caps_string (18, 30);
  gst_harness_set_src_caps_str (h, caps_str);
  g_free (caps_str);

  caps = gst_pad_get_current_caps (GST_PAD_PEER (h->sinkpad));
  ASSERT_NE (caps, nullptr);
  s = gst_caps_get_structure (caps, 0);
  EXPECT_TRUE (gst_structure_get_fraction (s, "framerate", &rate_n, &rate_d));
  EXPECT_EQ (rate_n, 30);
  EXPECT_EQ (rate_d, 1);
  gst_caps_unref (caps);

  gst_harness_teardown (h);
}

/**
 * @brief Push a flexible tensor with a valid header and 4 bytes of data.
 */
TEST_F (TensorTrainerFakeFw, flexibleInput)
{
  GstHarness *h = make_fake_trainer_harness ();
  GstTensorMetaInfo meta;
  GstMemory *data_mem, *mem;
  GstBuffer *buf;
  guint8 data[4] = { 7, 8, 9, 10 };

  ASSERT_NE (h, nullptr);
  gst_harness_set_src_caps_str (h, "other/tensors,format=flexible,framerate=0/1");

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT8;
  meta.dimension[0] = 4;
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  data_mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY, data,
      sizeof (data), 0, sizeof (data), NULL, NULL);
  mem = gst_tensor_meta_info_append_header (&meta, data_mem);
  gst_memory_unref (data_mem);
  ASSERT_NE (mem, nullptr);

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, mem);

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_OK);
  EXPECT_EQ (g_atomic_int_get (&fake_stat.push_done), 1);
  EXPECT_EQ (fake_stat.pushed_size, sizeof (data));
  EXPECT_EQ ((guint) fake_stat.pushed_first, 7U);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);

  gst_harness_teardown (h);
}

/**
 * @brief Push a flexible tensor whose memory is shorter than the tensor header.
 *
 * The 8 bytes end right before an inaccessible page, so reading the
 * 128-byte header out of them faults instead of silently reading the
 * neighbouring heap.
 */
TEST_F (TensorTrainerFakeFw, flexibleHeaderTruncated_n)
{
  const gsize page = (gsize) sysconf (_SC_PAGESIZE);
  const gsize size = 8;
  GstHarness *h;
  GstBuffer *buf;
  gpointer region;

  region = mmap (NULL, page * 2, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE (region, MAP_FAILED);
  memset (region, 0, page);
  if (mprotect ((guint8 *) region + page, page, PROT_NONE) != 0) {
    munmap (region, page * 2);
    FAIL () << "Cannot protect the guard page.";
  }

  h = make_fake_trainer_harness ();
  if (!h) {
    munmap (region, page * 2);
    FAIL () << "Cannot create the harness.";
  }
  gst_harness_set_src_caps_str (h, "other/tensors,format=flexible,framerate=0/1");

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
                                     region, page, page - size, size, NULL, NULL));

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (g_atomic_int_get (&fake_stat.push_calls), 0);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
  munmap (region, page * 2);
}

/**
 * @brief Push a flexible tensor whose header is long enough but invalid.
 */
TEST_F (TensorTrainerFakeFw, flexibleHeaderInvalid_n)
{
  GstHarness *h = make_fake_trainer_harness ();
  GstBuffer *buf;

  ASSERT_NE (h, nullptr);
  gst_harness_set_src_caps_str (h, "other/tensors,format=flexible,framerate=0/1");

  buf = gst_buffer_new_allocate (NULL, 256, NULL);
  gst_buffer_memset (buf, 0, 0, 256);

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (g_atomic_int_get (&fake_stat.push_calls), 0);

  gst_harness_teardown (h);
}

/**
 * @brief Push a flexible tensor whose header has a version the build cannot size.
 *
 * The header passes validation but its size is unknown. The declared data
 * size equals the whole memory, so only refusing the header keeps the
 * header bytes from reaching the sub-plugin as data.
 */
TEST_F (TensorTrainerFakeFw, flexibleHeaderUnknownVersion_n)
{
  GstHarness *h = make_fake_trainer_harness ();
  GstTensorMetaInfo meta;
  GstBuffer *buf;
  GstMapInfo map;

  ASSERT_NE (h, nullptr);
  gst_harness_set_src_caps_str (h, "other/tensors,format=flexible,framerate=0/1");

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT8;
  meta.dimension[0] = 256;
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  buf = gst_buffer_new_allocate (NULL, 256, NULL);
  gst_buffer_memset (buf, 0, 0, 256);
  ASSERT_TRUE (gst_buffer_map (buf, &map, GST_MAP_WRITE));
  EXPECT_TRUE (gst_tensor_meta_info_update_header (&meta, map.data));
  ((uint32_t *) map.data)[1] = 0xDE002000U;
  gst_buffer_unmap (buf, &map);

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (g_atomic_int_get (&fake_stat.push_calls), 0);

  gst_harness_teardown (h);
}

/**
 * @brief Memory of the allocator below.
 */
typedef struct {
  GstMemory mem;
  guint8 data[8];
} TrainerMapTestMemory;

/**
 * @brief Allocator counting its maps and unmaps, which can refuse to map.
 */
typedef struct {
  GstAllocator parent;
  gboolean refuse_map; /**< refuse every map */
  guint maps; /**< number of maps granted */
  guint unmaps; /**< number of unmaps */
  guint freed; /**< number of memories freed */
} TrainerMapTestAllocator;

/**
 * @brief Class of TrainerMapTestAllocator.
 */
typedef struct {
  GstAllocatorClass parent_class;
} TrainerMapTestAllocatorClass;

G_DEFINE_TYPE (TrainerMapTestAllocator, trainer_map_test_allocator, GST_TYPE_ALLOCATOR);

/**
 * @brief Map a memory of TrainerMapTestAllocator unless it refuses to.
 */
static gpointer
trainer_map_test_memory_map (GstMemory *mem, gsize, GstMapFlags)
{
  TrainerMapTestAllocator *self = (TrainerMapTestAllocator *) mem->allocator;

  if (self->refuse_map)
    return NULL;

  self->maps++;
  return ((TrainerMapTestMemory *) mem)->data;
}

/**
 * @brief Unmap a memory of TrainerMapTestAllocator.
 */
static void
trainer_map_test_memory_unmap (GstMemory *mem)
{
  ((TrainerMapTestAllocator *) mem->allocator)->unmaps++;
}

/**
 * @brief Free a memory of TrainerMapTestAllocator.
 */
static void
trainer_map_test_allocator_free (GstAllocator *allocator, GstMemory *mem)
{
  ((TrainerMapTestAllocator *) allocator)->freed++;
  g_free (mem);
}

/**
 * @brief Initialize the class of TrainerMapTestAllocator.
 */
static void
trainer_map_test_allocator_class_init (TrainerMapTestAllocatorClass *klass)
{
  GST_ALLOCATOR_CLASS (klass)->free = trainer_map_test_allocator_free;
}

/**
 * @brief Initialize a TrainerMapTestAllocator.
 */
static void
trainer_map_test_allocator_init (TrainerMapTestAllocator *self)
{
  GstAllocator *allocator = GST_ALLOCATOR_CAST (self);

  allocator->mem_type = "TrainerMapTest";
  allocator->mem_map = trainer_map_test_memory_map;
  allocator->mem_unmap = trainer_map_test_memory_unmap;
  GST_OBJECT_FLAG_SET (allocator, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
}

/**
 * @brief Create a TrainerMapTestAllocator.
 */
static TrainerMapTestAllocator *
trainer_map_test_allocator_new (gboolean refuse_map)
{
  TrainerMapTestAllocator *self = (TrainerMapTestAllocator *) g_object_new (
      trainer_map_test_allocator_get_type (), NULL);

  self->refuse_map = refuse_map;
  return self;
}

/**
 * @brief Append a memory of @a allocator, @a size bytes of @a value, to @a buf.
 */
static void
trainer_map_test_append_memory (
    GstBuffer *buf, TrainerMapTestAllocator *allocator, gsize size, guint8 value)
{
  TrainerMapTestMemory *mem = g_new0 (TrainerMapTestMemory, 1);

  gst_memory_init (GST_MEMORY_CAST (mem), GST_MEMORY_FLAG_NO_SHARE,
      GST_ALLOCATOR_CAST (allocator), NULL, sizeof (mem->data), 0, 0, size);
  memset (mem->data, value, size);
  gst_buffer_append_memory (buf, GST_MEMORY_CAST (mem));
}

/**
 * @brief Count the critical messages of GStreamer.
 */
static void
trainer_count_critical_log (const gchar *, GLogLevelFlags, const gchar *, gpointer user_data)
{
  (*(guint *) user_data)++;
}

/**
 * @brief What one push of the two tensors below did.
 */
typedef struct {
  GstFlowReturn ret; /**< flow return of the push */
  gint push_calls; /**< push_data() calls the buffer caused */
  gsize pushed_size; /**< size of the first tensor the sub-plugin saw */
  guint8 pushed_first; /**< its first byte */
  guint num_critical; /**< critical messages GStreamer logged during the push */
  TrainerMapTestAllocator *first; /**< allocator of the first tensor */
  TrainerMapTestAllocator *second; /**< allocator of the second tensor */
} TrainerMapTestResult;

/**
 * @brief Push @a num_mems tensors of @a mem_size bytes, each of which may refuse to map.
 *
 * The caps always announce two tensors of 4 bytes, so a different count or
 * size is what the element refuses. The counters are read before the harness
 * is torn down, because the dummy-data thread the element starts when it
 * pauses pushes as well. Both allocators outlive the harness and are unreffed
 * by the caller, even the second one that a single-memory push leaves unused.
 */
static void
trainer_map_test_push (guint num_mems, gsize mem_size, gboolean refuse_first,
    gboolean refuse_second, TrainerMapTestResult *result)
{
  GstHarness *h = make_fake_trainer_harness ();
  GstBuffer *buf;
  guint handler;

  memset (result, 0, sizeof (*result));
  result->first = trainer_map_test_allocator_new (refuse_first);
  result->second = trainer_map_test_allocator_new (refuse_second);
  ASSERT_NE (h, nullptr);

  gst_harness_set_src_caps_str (h,
      "other/tensors,format=static,num_tensors=2,framerate=0/1,"
      "dimensions=(string)\"4:1:1:1.4:1:1:1\",types=(string)\"uint8,uint8\"");

  buf = gst_buffer_new ();
  trainer_map_test_append_memory (buf, result->first, mem_size, 7);
  if (num_mems > 1)
    trainer_map_test_append_memory (buf, result->second, mem_size, 8);

  handler = g_log_set_handler ("GStreamer", G_LOG_LEVEL_CRITICAL,
      trainer_count_critical_log, &result->num_critical);
  result->ret = gst_harness_push (h, buf);
  g_log_remove_handler ("GStreamer", handler);

  result->push_calls = g_atomic_int_get (&fake_stat.push_calls);
  result->pushed_size = fake_stat.pushed_size;
  result->pushed_first = fake_stat.pushed_first;

  /* the dummy-data thread below repeats push_data () until it succeeds */
  fake_stat.push_ret = 0;
  gst_harness_teardown (h);
}

/**
 * @brief Release the allocators of a push.
 */
static void
trainer_map_test_result_clear (TrainerMapTestResult *result)
{
  gst_object_unref (result->first);
  gst_object_unref (result->second);
}

/**
 * @brief Push tensors of a custom allocator; each is mapped once and unmapped once.
 */
TEST_F (TensorTrainerFakeFw, customAllocatorInput)
{
  TrainerMapTestResult result;

  trainer_map_test_push (2, 4, FALSE, FALSE, &result);
  EXPECT_EQ (result.ret, GST_FLOW_OK);
  EXPECT_EQ (result.push_calls, 1);
  EXPECT_EQ (result.pushed_size, 4U);
  EXPECT_EQ ((guint) result.pushed_first, 7U);
  EXPECT_EQ (result.num_critical, 0U);
  EXPECT_EQ (result.first->maps, 1U);
  EXPECT_EQ (result.first->unmaps, 1U);
  EXPECT_EQ (result.first->freed, 1U);
  EXPECT_EQ (result.second->maps, 1U);
  EXPECT_EQ (result.second->unmaps, 1U);
  EXPECT_EQ (result.second->freed, 1U);

  trainer_map_test_result_clear (&result);
}

/**
 * @brief Push a tensor that cannot be mapped.
 *
 * The element must not unmap the memory it could not map: GStreamer refuses
 * an unmap whose map info does not belong to the memory with a critical
 * message, and hands it to the allocator when the stack happens to hold the
 * memory itself.
 */
TEST_F (TensorTrainerFakeFw, unmappableInput_n)
{
  TrainerMapTestResult result;

  trainer_map_test_push (2, 4, TRUE, FALSE, &result);
  EXPECT_EQ (result.ret, GST_FLOW_ERROR);
  EXPECT_EQ (result.push_calls, 0);
  EXPECT_EQ (result.num_critical, 0U);
  EXPECT_EQ (result.first->unmaps, 0U);
  EXPECT_EQ (result.first->freed, 1U);
  EXPECT_EQ (result.second->maps, 0U);
  EXPECT_EQ (result.second->unmaps, 0U);
  EXPECT_EQ (result.second->freed, 1U);

  trainer_map_test_result_clear (&result);
}

/**
 * @brief Push two tensors, the second of which cannot be mapped.
 *
 * The first tensor is mapped by then and must still be unmapped, while the
 * second one must not be.
 */
TEST_F (TensorTrainerFakeFw, unmappableSecondInput_n)
{
  TrainerMapTestResult result;

  trainer_map_test_push (2, 4, FALSE, TRUE, &result);
  EXPECT_EQ (result.ret, GST_FLOW_ERROR);
  EXPECT_EQ (result.push_calls, 0);
  EXPECT_EQ (result.num_critical, 0U);
  EXPECT_EQ (result.first->maps, 1U);
  EXPECT_EQ (result.first->unmaps, 1U);
  EXPECT_EQ (result.first->freed, 1U);
  EXPECT_EQ (result.second->unmaps, 0U);
  EXPECT_EQ (result.second->freed, 1U);

  trainer_map_test_result_clear (&result);
}

/**
 * @brief Push fewer memories than the caps announce; nothing is mapped.
 *
 * The element refuses the buffer before the mapping loop, so the cleanup
 * runs with no memory taken at all.
 */
TEST_F (TensorTrainerFakeFw, fewerMemoriesThanTensors_n)
{
  TrainerMapTestResult result;

  trainer_map_test_push (1, 4, FALSE, FALSE, &result);
  EXPECT_EQ (result.ret, GST_FLOW_ERROR);
  EXPECT_EQ (result.push_calls, 0);
  EXPECT_EQ (result.num_critical, 0U);
  EXPECT_EQ (result.first->maps, 0U);
  EXPECT_EQ (result.first->unmaps, 0U);
  EXPECT_EQ (result.first->freed, 1U);

  trainer_map_test_result_clear (&result);
}

/**
 * @brief Push memories larger than the caps announce; what was mapped is unmapped.
 */
TEST_F (TensorTrainerFakeFw, tensorSizeMismatch_n)
{
  TrainerMapTestResult result;

  trainer_map_test_push (2, 8, FALSE, FALSE, &result);
  EXPECT_EQ (result.ret, GST_FLOW_ERROR);
  EXPECT_EQ (result.push_calls, 0);
  EXPECT_EQ (result.num_critical, 0U);
  EXPECT_EQ (result.first->maps, 1U);
  EXPECT_EQ (result.first->unmaps, 1U);
  EXPECT_EQ (result.first->freed, 1U);
  EXPECT_EQ (result.second->maps, 0U);
  EXPECT_EQ (result.second->freed, 1U);

  trainer_map_test_result_clear (&result);
}

/**
 * @brief A sub-plugin that refuses the data still gets every memory released.
 */
TEST_F (TensorTrainerFakeFw, subpluginPushFailure_n)
{
  TrainerMapTestResult result;

  fake_stat.push_ret = -1;
  trainer_map_test_push (2, 4, FALSE, FALSE, &result);
  EXPECT_EQ (result.ret, GST_FLOW_ERROR);
  EXPECT_EQ (result.push_calls, 1);
  EXPECT_EQ (result.num_critical, 0U);
  EXPECT_EQ (result.first->maps, 1U);
  EXPECT_EQ (result.first->unmaps, 1U);
  EXPECT_EQ (result.first->freed, 1U);
  EXPECT_EQ (result.second->maps, 1U);
  EXPECT_EQ (result.second->unmaps, 1U);
  EXPECT_EQ (result.second->freed, 1U);

  trainer_map_test_result_clear (&result);
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
