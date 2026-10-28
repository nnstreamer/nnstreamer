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
#include <dlfcn.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#ifdef __GLIBC__
#include <malloc.h>
#if __GLIBC_PREREQ(2, 33)
#define HAVE_MALLINFO2 1
#endif
#endif
#ifndef HAVE_MALLINFO2
#define HAVE_MALLINFO2 0
#endif

#include <nnstreamer_util.h>
#include <unittest_util.h>
#include "nnstreamer_plugin_api.h"
#include "nnstreamer_plugin_api_filter.h"
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
    case 3:
      model_name = "mobilenet_v1_1.0_224_quant.tflite";
      break;
    case 4:
      model_name = "add.tflite";
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
  gsize idx, max_idx = 0U;

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

    max_idx = argmax_float (output, info_res.size / sizeof (gfloat));
  } else {
    ASSERT_TRUE (1 == 0);
  }

  EXPECT_EQ (max_idx, 951U);

  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);
}

/**
 * @brief Reload the model files in prop the way tensor_filter does.
 */
static int
_ReloadModel (const GstTensorFilterFramework *sp, GstTensorFilterProperties *prop, void *data)
{
  GstTensorFilterFrameworkEventData event_data;

  memset (&event_data, 0, sizeof (event_data));
  event_data.model_files = prop->model_files;
  event_data.num_models = prop->num_models;

  return sp->eventHandler (sp, prop, data, RELOAD_MODEL, &event_data);
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
  guint count = 0U;
  *is_float = 0;
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, is_float);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_output, &count);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&count, 1U, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_GE (count, 1U);

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
  guint count = 0U;
  *is_float = 1;
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, is_float);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_output, &count);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&count, 1U, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_GE (count, 1U);

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

#define LIFETIME_DELEGATE_LIB \
  "libnnstreamer_unittest_tflite_lifetime_delegate.so"

/**
 * @brief Fixture with the external delegate checking the lifetime of the
 *        delegate and the model against the interpreter using them.
 */
class nnstreamerFilterTensorFlow2LiteLifetime : public ::testing::Test
{
  protected:
  void *lib;
  gchar *lib_path;
  gchar *custom;
  int (*get_count) (const char *name);
  const GstTensorFilterFramework *sp;

  /**
   * @brief Find and load the delegate library before the sub-plugin does.
   * @note Holding a reference keeps the library mapped whatever the interpreter
   *       does, so a wrong destruction order is counted instead of crashing.
   */
  void SetUp () override
  {
    const gchar *build_root = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
    void (*reset) (void);

    lib = NULL;
    custom = NULL;
    get_count = NULL;
    lib_path = NULL;

#ifndef TFLITE_EXTERNAL_DELEGATE_SUPPORTED
    GTEST_SKIP () << "Built without the tensorflow-lite external delegate";
#endif

    if (build_root)
      lib_path = g_build_filename (build_root, "tests", LIFETIME_DELEGATE_LIB, NULL);

    if (!lib_path || !g_file_test (lib_path, G_FILE_TEST_EXISTS)) {
      gchar *exe = g_file_read_link ("/proc/self/exe", NULL);
      gchar *dir = exe ? g_path_get_dirname (exe) : g_get_current_dir ();

      g_free (lib_path);
      lib_path = g_build_filename (dir, LIFETIME_DELEGATE_LIB, NULL);
      g_free (dir);
      g_free (exe);
    }
    ASSERT_TRUE (g_file_test (lib_path, G_FILE_TEST_EXISTS)) << lib_path;

    lib = dlopen (lib_path, RTLD_NOW | RTLD_LOCAL);
    ASSERT_TRUE (lib != NULL) << dlerror ();

    get_count = (int (*) (const char *)) dlsym (lib, "nns_tflite_lifetime_delegate_get_count");
    reset = (void (*) (void)) dlsym (lib, "nns_tflite_lifetime_delegate_reset");
    ASSERT_TRUE (get_count != NULL && reset != NULL);
    reset ();

    custom = g_strdup_printf ("Delegate:External,ExtDelegateLib:%s", lib_path);

    sp = nnstreamer_filter_find ("tensorflow2-lite");
    ASSERT_TRUE (sp != NULL);
  }

  /**
   * @brief Release the delegate library.
   */
  void TearDown () override
  {
    if (lib)
      dlclose (lib);
    g_free (custom);
    g_free (lib_path);
  }

  /**
   * @brief Fill the filter properties for the model of the given option.
   */
  void fillProp (GstTensorFilterProperties *prop, const gchar **model_files,
      const gchar *custom_prop, gchar *shared_key)
  {
    memset (prop, 0, sizeof (GstTensorFilterProperties));
    prop->fwname = "tensorflow2-lite";
    prop->model_files = model_files;
    prop->num_models = 1;
    prop->custom_properties = custom_prop;
    prop->shared_tensor_filter_key = shared_key;
  }
};

/**
 * @brief Closing a filter must destroy the interpreter before its delegate and model.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, openClose)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file;
  void *data = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  model_files[0] = model_file;
  fillProp (&prop, model_files, custom, NULL);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_GE (get_count ("prepared"), 1);
  EXPECT_EQ (get_count ("live"), 1);

  sp->close (&prop, &data);
  EXPECT_EQ (get_count ("freed"), get_count ("prepared"));
  EXPECT_GE (get_count ("model_checks"), 1);
  EXPECT_EQ (get_count ("violations"), 0);
  EXPECT_EQ (get_count ("live"), 0);

  g_free (model_file);
}

/**
 * @brief Reloading a model must destroy the old interpreter before its delegate and model.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, reloadModel)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file, *model_file2;
  void *data = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  ASSERT_TRUE (_GetModelFilePath (&model_file2, 3));
  model_files[0] = model_file;
  fillProp (&prop, model_files, custom, NULL);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_EQ (get_count ("prepared"), 1);

  model_files[0] = model_file2;
  EXPECT_EQ (_ReloadModel (sp, &prop, data), 0);
  EXPECT_EQ (get_count ("prepared"), 2);
  EXPECT_EQ (get_count ("freed"), 1);
  EXPECT_EQ (get_count ("violations"), 0);

  sp->close (&prop, &data);
  EXPECT_EQ (get_count ("freed"), 2);
  EXPECT_EQ (get_count ("violations"), 0);

  g_free (model_file);
  g_free (model_file2);
}

/**
 * @brief A refused reload must destroy the unused interpreter before its delegate and model.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, reloadUnmatchedModel_n)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file, *model_file2;
  void *data = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  ASSERT_TRUE (_GetModelFilePath (&model_file2, 1));
  model_files[0] = model_file;
  fillProp (&prop, model_files, custom, NULL);

  ASSERT_EQ (sp->open (&prop, &data), 0);

  model_files[0] = model_file2;
  EXPECT_NE (_ReloadModel (sp, &prop, data), 0);
  EXPECT_EQ (get_count ("prepared"), 2);
  EXPECT_EQ (get_count ("freed"), 1);
  EXPECT_EQ (get_count ("violations"), 0);

  model_files[0] = model_file;
  sp->close (&prop, &data);
  EXPECT_EQ (get_count ("freed"), 2);
  EXPECT_EQ (get_count ("violations"), 0);

  g_free (model_file);
  g_free (model_file2);
}

#define SHARED_BRANCH                                                                                                 \
  "appsrc name=src%d ! other/tensors,num_tensors=1,dimensions=3:224:224:1,types=uint8,format=static,framerate=0/1 ! " \
  "tensor_filter framework=tensorflow2-lite model=\"%s\" custom=\"%s\" shared-tensor-filter-key=tflite_lifetime_f2f3 ! fakesink "

/**
 * @brief Starting a second filter with the same shared key reloads the shared
 *        interpreter; the old interpreter must go before the model it was built from.
 * @note Only the elements create the shared model table, so this case uses a
 *       pipeline. PAUSED starts (opens) both filters; no buffer is pushed.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, sharedKeyReopen)
{
  GstElement *gstpipe;
  gchar *model_file, *pipeline;
  gchar *branch1, *branch2;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  branch1 = g_strdup_printf (SHARED_BRANCH, 1, model_file, custom);
  branch2 = g_strdup_printf (SHARED_BRANCH, 2, model_file, custom);
  pipeline = g_strconcat (branch1, branch2, NULL);

  gstpipe = gst_parse_launch (pipeline, NULL);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  EXPECT_EQ (get_count ("prepared"), 2);
  EXPECT_EQ (get_count ("freed"), 1);
  EXPECT_EQ (get_count ("violations"), 0);

  EXPECT_EQ (gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (get_count ("freed"), 2);
  EXPECT_EQ (get_count ("violations"), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (branch1);
  g_free (branch2);
  g_free (model_file);
}

/**
 * @brief A reload of a path that is not a regular file loads nothing at all.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, reloadMissingModel_n)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file, *missing;
  void *data = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  model_files[0] = model_file;
  fillProp (&prop, model_files, custom, NULL);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_EQ (get_count ("prepared"), 1);

  missing = g_strdup_printf ("%s.missing", model_file);
  model_files[0] = missing;
  EXPECT_NE (_ReloadModel (sp, &prop, data), 0);
  EXPECT_EQ (get_count ("prepared"), 1);
  EXPECT_EQ (get_count ("freed"), 0);

  model_files[0] = model_file;
  sp->close (&prop, &data);
  EXPECT_EQ (get_count ("freed"), 1);
  EXPECT_EQ (get_count ("violations"), 0);

  g_free (missing);
  g_free (model_file);
}

/**
 * @brief A reload of a file that is not a model releases the interpreter that failed to load it.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, reloadUnloadableModel_n)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file, *not_a_model;
  void *data = NULL;
  gint fd;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  model_files[0] = model_file;
  fillProp (&prop, model_files, custom, NULL);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_EQ (get_count ("prepared"), 1);

  not_a_model = NULL;
  fd = g_file_open_tmp ("nnsb16XXXXXX.tflite", &not_a_model, NULL);
  ASSERT_GE (fd, 0);
  g_close (fd, NULL);
  ASSERT_TRUE (g_file_set_contents (not_a_model, "not a flatbuffer", 16, NULL));

  model_files[0] = not_a_model;
  EXPECT_NE (_ReloadModel (sp, &prop, data), 0);
  EXPECT_EQ (get_count ("prepared"), 1);
  EXPECT_EQ (get_count ("freed"), 0);

  model_files[0] = model_file;
  sp->close (&prop, &data);
  EXPECT_EQ (get_count ("freed"), 1);
  EXPECT_EQ (get_count ("violations"), 0);

  g_remove (not_a_model);
  g_free (not_a_model);
  g_free (model_file);
}

#define SHARED_UPDATABLE_BRANCH                                                                                       \
  "appsrc name=src%d ! other/tensors,num_tensors=1,dimensions=3:224:224:1,types=uint8,format=static,framerate=0/1 ! " \
  "tensor_filter name=filter%d framework=tensorflow2-lite model=\"%s\" custom=\"%s\" is-updatable=true "              \
  "shared-tensor-filter-key=tflite_lifetime_b16 ! fakesink "

/**
 * @brief A model the sharing cores cannot take leaves the shared interpreter in use.
 * @details The cores compare the reloaded model with their own tensors info and
 *          refuse it, so the reload fails and the old model files stay. Only the
 *          interpreter nobody took is released.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, sharedKeyRefusedReload_n)
{
  GstElement *gstpipe, *filter;
  gchar *model_file, *model_file2, *pipeline, *readback;
  gchar *branch1, *branch2;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  ASSERT_TRUE (_GetModelFilePath (&model_file2, 1));
  branch1 = g_strdup_printf (SHARED_UPDATABLE_BRANCH, 1, 1, model_file, custom);
  branch2 = g_strdup_printf (SHARED_UPDATABLE_BRANCH, 2, 2, model_file, custom);
  pipeline = g_strconcat (branch1, branch2, NULL);

  gstpipe = gst_parse_launch (pipeline, NULL);
  ASSERT_TRUE (gstpipe != nullptr);

  ASSERT_NE (gst_element_set_state (gstpipe, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  ASSERT_EQ (get_count ("prepared"), 2);
  ASSERT_EQ (get_count ("freed"), 1);

  filter = gst_bin_get_by_name (GST_BIN (gstpipe), "filter1");
  ASSERT_TRUE (filter != nullptr);

  /* the tensors info of this model differs, so both cores refuse it */
  g_object_set (filter, "model", model_file2, NULL);

  EXPECT_EQ (get_count ("prepared"), 3);
  EXPECT_EQ (get_count ("freed"), 2);
  EXPECT_EQ (get_count ("violations"), 0);

  readback = NULL;
  g_object_get (filter, "model", &readback, NULL);
  EXPECT_STREQ (readback, model_file);
  g_free (readback);

  EXPECT_EQ (gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (get_count ("freed"), 3);
  EXPECT_EQ (get_count ("live"), 0);
  EXPECT_EQ (get_count ("violations"), 0);

  gst_object_unref (filter);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (branch1);
  g_free (branch2);
  g_free (model_file);
  g_free (model_file2);
}

/**
 * @brief The last ExtDelegateLib given in the custom option is the one loaded.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, extDelegateLibGivenTwice)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file, *custom_twice;
  void *data = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  model_files[0] = model_file;
  custom_twice = g_strdup_printf (
      "Delegate:External,ExtDelegateLib:/nonexistent/libdelegate.so,ExtDelegateLib:%s", lib_path);
  fillProp (&prop, model_files, custom_twice, NULL);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_EQ (get_count ("prepared"), 1);

  sp->close (&prop, &data);
  EXPECT_EQ (get_count ("violations"), 0);

  g_free (custom_twice);
  g_free (model_file);
}

/**
 * @brief ExtDelegateKeyVal entries reach the delegate; malformed ones are dropped.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, extDelegateKeyValOptions)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file, *custom_kv;
  void *data = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  model_files[0] = model_file;
  custom_kv = g_strdup_printf ("Delegate:External,ExtDelegateLib:%s,"
                               "ExtDelegateKeyVal:mode#ok;no_separator,ExtDelegateKeyVal:extra#1",
      lib_path);
  fillProp (&prop, model_files, custom_kv, NULL);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_EQ (get_count ("options"), 2);
  EXPECT_EQ (get_count ("prepared"), 1);

  sp->close (&prop, &data);
  EXPECT_EQ (get_count ("violations"), 0);

  g_free (custom_kv);
  g_free (model_file);
}

/**
 * @brief A delegate that refuses to be applied must fail the open.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteLifetime, extDelegateRefusesToApply_n)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file, *custom_fail;
  void *data = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  model_files[0] = model_file;
  custom_fail = g_strdup_printf (
      "Delegate:External,ExtDelegateLib:%s,ExtDelegateKeyVal:mode#fail", lib_path);
  fillProp (&prop, model_files, custom_fail, NULL);

  EXPECT_NE (sp->open (&prop, &data), 0);
  EXPECT_EQ (get_count ("prepared"), 0);
  EXPECT_EQ (get_count ("violations"), 0);
  /* no kernel is built when the delegate refuses, but it must still be destroyed */
  EXPECT_EQ (get_count ("live"), 0);
  if (data)
    sp->close (&prop, &data);

  g_free (custom_fail);
  g_free (model_file);
}

/**
 * @brief The external delegate is dropped when no library is given.
 */
TEST (nnstreamerFilterTensorFlow2Lite, extDelegateWithoutLib)
{
  GstTensorFilterProperties prop = {};
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file;
  void *data = NULL;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tensorflow2-lite");
  ASSERT_TRUE (sp != NULL);
  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));

  model_files[0] = model_file;
  prop.fwname = "tensorflow2-lite";
  prop.model_files = model_files;
  prop.num_models = 1;
  prop.custom_properties = "Delegate:External";

  EXPECT_EQ (sp->open (&prop, &data), 0);
  sp->close (&prop, &data);

  g_free (model_file);
}

/**
 * @brief Every other custom option is parsed, and an unknown one is ignored.
 */
TEST (nnstreamerFilterTensorFlow2Lite, customOptionsParsed)
{
  GstTensorFilterProperties prop = {};
  const gchar *model_files[] = { NULL, NULL };
  gchar *model_file;
  void *data = NULL;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tensorflow2-lite");
  ASSERT_TRUE (sp != NULL);
  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));

  model_files[0] = model_file;
  prop.fwname = "tensorflow2-lite";
  prop.model_files = model_files;
  prop.num_models = 1;
  prop.custom_properties = "NumThreads:2,Delegate:Unknown,QNNBackend:HTP,"
                           "QNNPerformanceMode:powersaver,ExtDelegateKeyVal:k#v,UnknownOption:1,"
                           "no_separator";

  EXPECT_EQ (sp->open (&prop, &data), 0);
  sp->close (&prop, &data);

  g_free (model_file);
}

/**
 * @brief Every delegate name and QNN option is parsed, whatever this build supports.
 * @note The model is missing, so the open fails before the parsed delegate is
 *       built; that keeps the case free of any accelerator this host lacks.
 */
TEST (nnstreamerFilterTensorFlow2Lite, customOptionsDelegateNames_n)
{
  static const gchar *customs[] = {
    "Delegate:NNAPI",
    "Delegate:GPU",
    "Delegate:QNN,QNNBackend:DSP,QNNPerformanceMode:default",
    "Delegate:QNN,QNNBackend:GPU,QNNPerformanceMode:highperformance",
    "Delegate:QNN,QNNBackend:Unknown,QNNPerformanceMode:Unknown",
    NULL,
  };
  const gchar *model_files[] = { "/nonexistent/model.tflite", NULL };
  GstTensorFilterProperties prop = {};
  void *data = NULL;
  guint i;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tensorflow2-lite");
  ASSERT_TRUE (sp != NULL);

  prop.fwname = "tensorflow2-lite";
  prop.model_files = model_files;
  prop.num_models = 1;

  for (i = 0; customs[i]; i++) {
    prop.custom_properties = customs[i];
    EXPECT_NE (sp->open (&prop, &data), 0) << customs[i];
    EXPECT_TRUE (data == NULL);
  }
}

/**
 * @brief The open must fail when no model file is given.
 */
TEST (nnstreamerFilterTensorFlow2Lite, openWithoutModelFile_n)
{
  const gchar *model_files[] = { NULL, NULL };
  GstTensorFilterProperties prop = {};
  void *data = NULL;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tensorflow2-lite");
  ASSERT_TRUE (sp != NULL);

  prop.fwname = "tensorflow2-lite";
  prop.model_files = model_files;
  prop.num_models = 1;

  EXPECT_NE (sp->open (&prop, &data), 0);
  EXPECT_TRUE (data == NULL);
}

/**
 * @brief ExtDelegateLib given twice must not leak the first path.
 * @note The paths are large so that a leak is measurable in the heap usage
 *       reported by glibc; the model is missing so open fails before tflite
 *       allocates anything that it may keep.
 */
TEST (nnstreamerFilterTensorFlow2Lite, extDelegateLibGivenTwiceNoLeak_n)
{
#if HAVE_MALLINFO2
  const gsize path_len = 64 * 1024;
  const guint repeat = 10;
  const gchar *model_files[] = { "/nonexistent/model.tflite", NULL };
  GstTensorFilterProperties prop;
  gchar *long_path, *custom_twice;
  struct mallinfo2 before, after;
  gsize used_before, used_after;
  void *data = NULL;
  guint i;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tensorflow2-lite");
  ASSERT_TRUE (sp != NULL);

  long_path = (gchar *) g_malloc (path_len + 1);
  memset (long_path, 'a', path_len);
  long_path[path_len] = '\0';
  custom_twice = g_strdup_printf ("ExtDelegateLib:%s,ExtDelegateLib:%s", long_path, long_path);

  memset (&prop, 0, sizeof (prop));
  prop.fwname = "tensorflow2-lite";
  prop.model_files = model_files;
  prop.num_models = 1;
  prop.custom_properties = custom_twice;

  /* warm up whatever the sub-plugin initializes once */
  EXPECT_NE (sp->open (&prop, &data), 0);
  EXPECT_NE (sp->open (&prop, &data), 0);

  before = mallinfo2 ();
  for (i = 0; i < repeat; i++) {
    EXPECT_NE (sp->open (&prop, &data), 0);
    EXPECT_TRUE (data == NULL);
  }
  after = mallinfo2 ();

  used_before = before.uordblks + before.hblkhd;
  used_after = after.uordblks + after.hblkhd;
  EXPECT_LT (used_after, used_before + path_len);

  g_free (custom_twice);
  g_free (long_path);
#else
  GTEST_SKIP () << "mallinfo2 () is not available";
#endif
}

/**
 * @brief Fixture opening the sub-plugin directly through its v1 callbacks.
 */
class nnstreamerFilterTensorFlow2LiteV1 : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  GstTensorFilterProperties prop;
  const gchar *model_files[2];
  gchar *model_file;
  void *data;

  /**
   * @brief Find the sub-plugin and fill the properties for the default model.
   */
  void SetUp () override
  {
    data = NULL;
    model_file = NULL;
    sp = nnstreamer_filter_find ("tensorflow2-lite");
    ASSERT_TRUE (sp != NULL);
    ASSERT_TRUE (_GetModelFilePath (&model_file, 3));

    model_files[0] = model_file;
    model_files[1] = NULL;
    memset (&prop, 0, sizeof (prop));
    prop.fwname = "tensorflow2-lite";
    prop.model_files = model_files;
    prop.num_models = 1;
  }

  /**
   * @brief Close the sub-plugin if a case left it open.
   */
  void TearDown () override
  {
    if (sp && data)
      sp->close (&prop, &data);
    g_free (model_file);
  }

  /**
   * @brief Allocate the memory for the given tensors info.
   */
  static void allocMemory (const GstTensorsInfo *info, GstTensorMemory *mem)
  {
    for (guint i = 0; i < info->num_tensors; i++) {
      mem[i].size = gst_tensors_info_get_size (info, i);
      mem[i].data = g_malloc0 (mem[i].size);
    }
  }

  /**
   * @brief Free the memory allocated by allocMemory ().
   */
  static void freeMemory (const GstTensorsInfo *info, GstTensorMemory *mem)
  {
    for (guint i = 0; i < info->num_tensors; i++)
      g_free (mem[i].data);
  }
};

/**
 * @brief The sub-plugin is registered as a v1 framework and describes itself.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, frameworkInfo)
{
  GstTensorFilterFrameworkInfo info;
  gboolean has_gpu = FALSE;

  EXPECT_EQ (sp->version, GST_TENSOR_FILTER_FRAMEWORK_V1);

  memset (&info, 0, sizeof (info));
  ASSERT_EQ (sp->getFrameworkInfo (sp, &prop, NULL, &info), 0);
  EXPECT_STREQ (info.name, "tensorflow2-lite");
  EXPECT_FALSE (info.allow_in_place);
  EXPECT_FALSE (info.allocate_in_invoke);
  EXPECT_FALSE (info.run_without_model);
  EXPECT_TRUE (info.verify_model_path);
  EXPECT_TRUE (info.statistics != NULL);
  EXPECT_EQ (info.accl_default, ACCL_CPU);
  ASSERT_GT (info.num_hw, 0);
  for (int i = 0; i < info.num_hw; i++)
    has_gpu |= (info.hw_list[i] == ACCL_GPU);
  EXPECT_TRUE (has_gpu);

  /* an opened instance reports the same */
  ASSERT_EQ (sp->open (&prop, &data), 0);
  memset (&info, 0, sizeof (info));
  ASSERT_EQ (sp->getFrameworkInfo (sp, &prop, data, &info), 0);
  EXPECT_STREQ (info.name, "tensorflow2-lite");
}

/**
 * @brief Closing without open or twice must not crash.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, openCloseTwice)
{
  sp->close (&prop, &data);
  EXPECT_TRUE (data == NULL);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_TRUE (data != NULL);
  sp->close (&prop, &data);
  EXPECT_TRUE (data == NULL);
  sp->close (&prop, &data);
}

/**
 * @brief The open fails for a model file that does not exist.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, openMissingModel_n)
{
  model_files[0] = "null.tflite";
  EXPECT_NE (sp->open (&prop, &data), 0);
  EXPECT_TRUE (data == NULL);
}

/**
 * @brief The model info is the input and output of the model.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, getModelInfo)
{
  GstTensorsInfo in_info, out_info;
  GstTensorInfo *info;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  EXPECT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info), 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  info = gst_tensors_info_get_nth_info (&in_info, 0);
  EXPECT_EQ (info->type, _NNS_UINT8);
  EXPECT_EQ (info->dimension[0], 3U);
  EXPECT_EQ (info->dimension[1], 224U);
  EXPECT_EQ (info->dimension[2], 224U);
  EXPECT_EQ (info->dimension[3], 1U);

  EXPECT_EQ (out_info.num_tensors, 1U);
  info = gst_tensors_info_get_nth_info (&out_info, 0);
  EXPECT_EQ (info->type, _NNS_UINT8);
  EXPECT_EQ (info->dimension[0], 1001U);
  EXPECT_EQ (info->dimension[1], 1U);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief The model info is refused without an opened instance.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, getModelInfoInvalid_n)
{
  GstTensorsInfo in_info, out_info;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  EXPECT_NE (sp->getModelInfo (sp, &prop, NULL, GET_IN_OUT_INFO, &in_info, &out_info), 0);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Setting the input info resizes the model and returns the new output info.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, setInputInfo)
{
  GstTensorsInfo in_info, out_info, set_out, get_in, get_out;
  gchar *add_model;

  ASSERT_TRUE (_GetModelFilePath (&add_model, 4));
  model_files[0] = add_model;
  ASSERT_EQ (sp->open (&prop, &data), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info), 0);

  /* the same input info keeps the output info */
  gst_tensors_info_init (&set_out);
  EXPECT_EQ (sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in_info, &set_out), 0);
  EXPECT_TRUE (gst_tensors_info_is_equal (&set_out, &out_info));
  gst_tensors_info_free (&set_out);

  /* a larger input is taken and reflected by the model info */
  in_info.info[0].dimension[0] *= 2;
  gst_tensors_info_init (&set_out);
  EXPECT_EQ (sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in_info, &set_out), 0);

  gst_tensors_info_init (&get_in);
  gst_tensors_info_init (&get_out);
  ASSERT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &get_in, &get_out), 0);
  EXPECT_TRUE (gst_tensors_info_is_equal (&get_in, &in_info));
  EXPECT_TRUE (gst_tensors_info_is_equal (&get_out, &set_out));
  EXPECT_EQ (get_out.info[0].dimension[0], out_info.info[0].dimension[0] * 2);

  gst_tensors_info_free (&get_in);
  gst_tensors_info_free (&get_out);
  gst_tensors_info_free (&set_out);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);

  sp->close (&prop, &data);
  g_free (add_model);
}

/**
 * @brief An input info the model cannot take is refused and the model is kept as it was.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, setInputInfoInvalid_n)
{
  GstTensorsInfo in_info, out_info, set_info, set_out, get_in, get_out;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info), 0);

  /* another type */
  gst_tensors_info_init (&set_info);
  gst_tensors_info_copy (&set_info, &in_info);
  set_info.info[0].type = _NNS_FLOAT32;
  gst_tensors_info_init (&set_out);
  EXPECT_NE (sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &set_info, &set_out), 0);
  gst_tensors_info_free (&set_out);

  /* another number of tensors */
  set_info.info[0].type = in_info.info[0].type;
  set_info.num_tensors = 2;
  gst_tensor_info_copy (gst_tensors_info_get_nth_info (&set_info, 1),
      gst_tensors_info_get_nth_info (&in_info, 0));
  gst_tensors_info_init (&set_out);
  EXPECT_NE (sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &set_info, &set_out), 0);
  gst_tensors_info_free (&set_out);

  gst_tensors_info_init (&get_in);
  gst_tensors_info_init (&get_out);
  ASSERT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &get_in, &get_out), 0);
  EXPECT_TRUE (gst_tensors_info_is_equal (&get_in, &in_info));
  EXPECT_TRUE (gst_tensors_info_is_equal (&get_out, &out_info));

  gst_tensors_info_free (&get_in);
  gst_tensors_info_free (&get_out);
  gst_tensors_info_free (&set_info);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Invoke runs the model through the v1 callback.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, invoke)
{
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT] = {};
  GstTensorMemory output[NNS_TENSOR_SIZE_LIMIT] = {};
  GstTensorsInfo in_info, out_info;
  GstTensorFilterFrameworkInfo info;

  ASSERT_EQ (sp->open (&prop, &data), 0);
  memset (&info, 0, sizeof (info));
  ASSERT_EQ (sp->getFrameworkInfo (sp, &prop, data, &info), 0);
  ASSERT_TRUE (info.statistics != NULL);
  const int64_t invoked = info.statistics->total_invoke_num;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info), 0);
  allocMemory (&in_info, input);
  allocMemory (&out_info, output);

  EXPECT_EQ (sp->invoke (sp, &prop, data, input, output), 0);
  EXPECT_EQ (sp->invoke (sp, &prop, data, input, output), 0);
  EXPECT_EQ (info.statistics->total_invoke_num, invoked + 2);

  freeMemory (&in_info, input);
  freeMemory (&out_info, output);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Invoke fails without an opened instance or without the tensors.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, invokeInvalid_n)
{
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT] = {};
  GstTensorMemory output[NNS_TENSOR_SIZE_LIMIT] = {};

  EXPECT_NE (sp->invoke (sp, &prop, NULL, input, output), 0);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_NE (sp->invoke (sp, &prop, data, NULL, output), 0);
  EXPECT_NE (sp->invoke (sp, &prop, data, input, NULL), 0);
  EXPECT_NE (sp->invoke (sp, &prop, data, NULL, NULL), 0);
}

/**
 * @brief Invoke-dynamic is refused instead of aborting the process.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, invokeDynamic_n)
{
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT] = {};
  GstTensorMemory output[NNS_TENSOR_SIZE_LIMIT] = {};
  GstTensorsInfo in_info, out_info;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info), 0);
  allocMemory (&in_info, input);
  allocMemory (&out_info, output);

  prop.invoke_dynamic = TRUE;
  EXPECT_NE (sp->invoke (sp, &prop, data, input, output), 0);
  prop.invoke_dynamic = FALSE;

  freeMemory (&in_info, input);
  freeMemory (&out_info, output);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Reloading a model is supported; the is-updatable probe has no model.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, reloadModel)
{
  gchar *model_file2;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  /* tensor_filter probes is-updatable with no event data */
  EXPECT_NE (sp->eventHandler (sp, &prop, data, RELOAD_MODEL, NULL), -ENOENT);

  ASSERT_TRUE (_GetModelFilePath (&model_file2, 0));
  model_files[0] = model_file2;
  EXPECT_EQ (_ReloadModel (sp, &prop, data), 0);

  model_files[0] = model_file;
  EXPECT_EQ (_ReloadModel (sp, &prop, data), 0);
  g_free (model_file2);
}

/**
 * @brief A reload is refused without an opened instance or with other than one model.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, reloadModelInvalid_n)
{
  EXPECT_NE (_ReloadModel (sp, &prop, NULL), 0);

  ASSERT_EQ (sp->open (&prop, &data), 0);
  prop.num_models = 2;
  EXPECT_NE (_ReloadModel (sp, &prop, data), 0);
  prop.num_models = 0;
  EXPECT_NE (_ReloadModel (sp, &prop, data), 0);
  prop.num_models = 1;
}

/**
 * @brief Any other event is left to tensor_filter.
 */
TEST_F (nnstreamerFilterTensorFlow2LiteV1, eventNotHandled_n)
{
  GstTensorFilterFrameworkEventData event_data;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  memset (&event_data, 0, sizeof (event_data));
  event_data.custom_properties = "NumThreads:2";
  EXPECT_EQ (sp->eventHandler (sp, &prop, data, CUSTOM_PROP, &event_data), -ENOENT);

  memset (&event_data, 0, sizeof (event_data));
  EXPECT_EQ (sp->eventHandler (sp, &prop, data, SET_ACCELERATOR, &event_data), -ENOENT);
  EXPECT_EQ (sp->eventHandler (sp, &prop, data, DESTROY_NOTIFY, &event_data), -ENOENT);
}

/**
 * @brief The accelerator property reads back the value the application set.
 * @note End-to-end check with tensorflow2-lite; tensor_filter's own cases are in unittest_filter_common.
 */
TEST (nnstreamerFilterTensorFlow2Lite, acceleratorReadBack)
{
  static const gchar *accelerators[] = {
    "true:cpu",
    "true:!npu,gpu",
    "true:cpu,gpu",
    "false",
    NULL,
  };
  gchar *model_file, *pipeline, *readback;
  GstElement *gstpipe, *filter;
  guint i;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 3));

  for (i = 0; accelerators[i]; i++) {
    /* both orders of the framework and accelerator properties */
    for (int order = 0; order < 2; order++) {
      if (order == 0)
        pipeline = g_strdup_printf ("tensor_filter name=tfilter framework=tensorflow2-lite model=\"%s\" accelerator=%s ! fakesink",
            model_file, accelerators[i]);
      else
        pipeline = g_strdup_printf ("tensor_filter name=tfilter accelerator=%s framework=tensorflow2-lite model=\"%s\" ! fakesink",
            accelerators[i], model_file);

      gstpipe = gst_parse_launch (pipeline, NULL);
      ASSERT_TRUE (gstpipe != nullptr);
      filter = gst_bin_get_by_name (GST_BIN (gstpipe), "tfilter");
      ASSERT_TRUE (filter != nullptr);

      readback = NULL;
      g_object_get (filter, "accelerator", &readback, NULL);
      EXPECT_STREQ (readback, accelerators[i]) << pipeline;
      g_free (readback);

      gst_object_unref (filter);
      gst_object_unref (gstpipe);
      g_free (pipeline);
    }
  }

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
