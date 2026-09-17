/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_decoder_python3.cc
 * @date    15 Sep 2026
 * @brief   Unit test for the python3 tensor_decoder sub-plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <string.h>
#include <unittest_python3_util.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>

/** @brief Size of each of the two input tensors */
#define TENSOR_SIZE (4U)

/**
 * @brief Build the path of custom_decoder_raw.py
 */
static gchar *
_script_path (void)
{
  return g_build_filename (g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH"), "tests",
      "test_models", "models", "custom_decoder_raw.py", NULL);
}

/**
 * @brief Open the python3 decoder with custom_decoder_raw.py in the given mode.
 */
static const GstTensorDecoderDef *
_open_decoder (void **pdata, const gchar *mode)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("python3");
  gchar *script = _script_path ();
  int ret;

  if (dec == NULL || !dec->init (pdata)
      || !py_test_setenv ("NNS_TEST_PY_DECODER_MODE", mode)) {
    g_free (script);
    return NULL;
  }

  ret = dec->setOption (pdata, 0, script);
  g_free (script);

  return ret ? dec : NULL;
}

/**
 * @brief Two uint8 tensors of TENSOR_SIZE.
 */
static void
_init_config (GstTensorsConfig *config)
{
  gst_tensors_config_init (config);
  config->info.num_tensors = 2;
  for (guint i = 0; i < 2; i++) {
    config->info.info[i].type = _NNS_UINT8;
    gst_tensor_parse_dimension ("4", config->info.info[i].dimension);
  }
  config->rate_n = 0;
  config->rate_d = 1;
}

/**
 * @brief Decode two tensors filled from seed; return the flow and check the concatenated output.
 */
static GstFlowReturn
_decode_and_check (const GstTensorDecoderDef *dec, void **pdata,
    const GstTensorsConfig *config, guint seed, gboolean *matched)
{
  guint8 data[2 * TENSOR_SIZE];
  GstTensorMemory input[2];
  GstBuffer *outbuf = gst_buffer_new ();
  GstFlowReturn ret;

  for (guint i = 0; i < sizeof (data); i++)
    data[i] = (guint8) (seed + i);
  for (guint i = 0; i < 2; i++) {
    input[i].data = data + i * TENSOR_SIZE;
    input[i].size = TENSOR_SIZE;
  }

  ret = dec->decode (pdata, config, input, outbuf);
  *matched = (ret == GST_FLOW_OK && gst_buffer_get_size (outbuf) == sizeof (data)
              && gst_buffer_memcmp (outbuf, 0, data, sizeof (data)) == 0);
  gst_buffer_unref (outbuf);

  return ret;
}

/**
 * @brief The decoder hands both tensors and their shapes to the script.
 */
TEST (nnstreamerDecoderPython3, decodeOutput)
{
  void *pdata = NULL;
  GstTensorsConfig config;
  gboolean matched = FALSE;
  const GstTensorDecoderDef *dec = _open_decoder (&pdata, "concat");

  ASSERT_NE (dec, nullptr);
  _init_config (&config);

  EXPECT_EQ (_decode_and_check (dec, &pdata, &config, 7, &matched), GST_FLOW_OK);
  EXPECT_TRUE (matched);

  dec->exit (&pdata);
  gst_tensors_config_free (&config);
}

/**
 * @brief The caps the script gives are the caps of the decoder.
 */
TEST (nnstreamerDecoderPython3, getOutCaps)
{
  void *pdata = NULL;
  GstTensorsConfig config;
  GstCaps *caps;
  GstStructure *structure;
  gint rate_n = 0, rate_d = 0;
  const GstTensorDecoderDef *dec = _open_decoder (&pdata, "concat");

  ASSERT_NE (dec, nullptr);
  _init_config (&config);
  config.rate_n = 30;

  caps = dec->getOutCaps (&pdata, &config);
  ASSERT_NE (caps, nullptr);
  ASSERT_GT (gst_caps_get_size (caps), 0U);
  structure = gst_caps_get_structure (caps, 0);
  EXPECT_STREQ (gst_structure_get_name (structure), "application/octet-stream");
  EXPECT_TRUE (gst_structure_get_fraction (structure, "framerate", &rate_n, &rate_d));
  EXPECT_EQ (rate_n, 30);
  EXPECT_EQ (rate_d, 1);
  gst_caps_unref (caps);

  dec->exit (&pdata);
  gst_tensors_config_free (&config);
}

/**
 * @brief Decoding releases the lists and the tensor shapes it builds for the script.
 */
TEST (nnstreamerDecoderPython3, decodeRepeatedKeepsObjects)
{
  void *pdata = NULL;
  GstTensorsConfig config;
  gboolean matched = FALSE;
  guint mismatch = 0;
  Py_ssize_t before, after;
  const GstTensorDecoderDef *dec = _open_decoder (&pdata, "concat");

  ASSERT_NE (dec, nullptr);
  _init_config (&config);

  for (guint i = 0; i < 10; i++)
    _decode_and_check (dec, &pdata, &config, i, &matched);

  before = py_test_gc_object_count ();
  ASSERT_GT (before, 0);
  for (guint i = 0; i < 500; i++) {
    if (_decode_and_check (dec, &pdata, &config, i, &matched) != GST_FLOW_OK || !matched)
      mismatch++;
  }
  after = py_test_gc_object_count ();

  EXPECT_EQ (mismatch, 0U);
  EXPECT_LE (after - before, PY_TEST_GC_SLACK);

  dec->exit (&pdata);
  gst_tensors_config_free (&config);
}

/**
 * @brief A script raising in decode fails the decode and still releases what was built for it.
 */
TEST (nnstreamerDecoderPython3, decodeRaiseKeepsObjects_n)
{
  void *pdata = NULL;
  GstTensorsConfig config;
  gboolean matched = FALSE;
  guint succeeded = 0;
  Py_ssize_t before, after;
  const GstTensorDecoderDef *dec = _open_decoder (&pdata, "raise");

  ASSERT_NE (dec, nullptr);
  _init_config (&config);

  for (guint i = 0; i < 10; i++)
    _decode_and_check (dec, &pdata, &config, i, &matched);

  before = py_test_gc_object_count ();
  ASSERT_GT (before, 0);
  for (guint i = 0; i < 50; i++) {
    if (_decode_and_check (dec, &pdata, &config, i, &matched) != GST_FLOW_ERROR)
      succeeded++;
  }
  after = py_test_gc_object_count ();

  EXPECT_EQ (succeeded, 0U);
  EXPECT_LE (after - before, PY_TEST_GC_SLACK);

  dec->exit (&pdata);
  gst_tensors_config_free (&config);
}

/**
 * @brief An output buffer the decoder cannot write fails the decode and releases the script's result.
 */
TEST (nnstreamerDecoderPython3, decodeUnwritableOutput_n)
{
  void *pdata = NULL;
  GstTensorsConfig config;
  guint8 data[2 * TENSOR_SIZE] = { 0 };
  guint8 readonly[TENSOR_SIZE] = { 0 };
  GstTensorMemory input[2];
  guint succeeded = 0;
  Py_ssize_t before, after;
  const GstTensorDecoderDef *dec = _open_decoder (&pdata, "fixed");

  ASSERT_NE (dec, nullptr);
  _init_config (&config);
  for (guint i = 0; i < 2; i++) {
    input[i].data = data + i * TENSOR_SIZE;
    input[i].size = TENSOR_SIZE;
  }

  before = py_test_attr_refcount ("custom_decoder_raw", "FIXED");
  ASSERT_GT (before, 0);
  for (guint i = 0; i < 50; i++) {
    GstBuffer *outbuf = gst_buffer_new_wrapped_full (GST_MEMORY_FLAG_READONLY,
        readonly, sizeof (readonly), 0, sizeof (readonly), NULL, NULL);

    if (dec->decode (&pdata, &config, input, outbuf) != GST_FLOW_ERROR)
      succeeded++;
    gst_buffer_unref (outbuf);
  }
  after = py_test_attr_refcount ("custom_decoder_raw", "FIXED");

  EXPECT_EQ (succeeded, 0U);
  EXPECT_EQ (after, before);

  dec->exit (&pdata);
  gst_tensors_config_free (&config);
}

/**
 * @brief Arguments of a decoding thread
 */
typedef struct {
  const GstTensorDecoderDef *dec;
  void *pdata;
  const GstTensorsConfig *config;
  guint seed;
  guint mismatch;
} DecodeThreadData;

/**
 * @brief Decode in a thread of its own with the instance it was given.
 */
static gpointer
_decode_thread (gpointer user_data)
{
  DecodeThreadData *td = (DecodeThreadData *) user_data;
  gboolean matched = FALSE;

  for (guint i = 0; i < 300; i++) {
    if (_decode_and_check (td->dec, &td->pdata, td->config, td->seed + i, &matched) != GST_FLOW_OK
        || !matched)
      td->mismatch++;
  }

  return NULL;
}

/**
 * @brief Decoders running in threads of their own decode correctly and release what they build.
 */
TEST (nnstreamerDecoderPython3, decodeMultiThread)
{
  DecodeThreadData td[4];
  GThread *threads[4];
  GstTensorsConfig config;
  gboolean matched = FALSE;
  Py_ssize_t before, after;

  _init_config (&config);
  for (guint t = 0; t < 4; t++) {
    td[t].pdata = NULL;
    td[t].dec = _open_decoder (&td[t].pdata, "concat");
    ASSERT_NE (td[t].dec, nullptr);
    td[t].config = &config;
    td[t].seed = t * 50;
    td[t].mismatch = 0;
    for (guint i = 0; i < 10; i++)
      _decode_and_check (td[t].dec, &td[t].pdata, &config, i, &matched);
  }

  before = py_test_gc_object_count ();
  ASSERT_GT (before, 0);
  for (guint t = 0; t < 4; t++)
    threads[t] = g_thread_new ("decode", _decode_thread, &td[t]);
  for (guint t = 0; t < 4; t++)
    g_thread_join (threads[t]);
  after = py_test_gc_object_count ();

  for (guint t = 0; t < 4; t++) {
    EXPECT_EQ (td[t].mismatch, 0U);
    td[t].dec->exit (&td[t].pdata);
  }
  EXPECT_LE (after - before, PY_TEST_GC_SLACK);
  gst_tensors_config_free (&config);
}

/**
 * @brief Remove a directory and everything in it.
 */
static void
_remove_dir (const gchar *path)
{
  GDir *dir = g_dir_open (path, 0, NULL);

  if (dir) {
    const gchar *name;

    while ((name = g_dir_read_name (dir)) != NULL) {
      gchar *child = g_build_filename (path, name, NULL);

      if (g_file_test (child, G_FILE_TEST_IS_DIR))
        _remove_dir (child);
      else
        g_unlink (child);
      g_free (child);
    }
    g_dir_close (dir);
  }
  g_rmdir (path);
}

/**
 * @brief A script of another directory puts that directory on sys.path, once.
 * @note The interpreter writes __pycache__ next to the copy, so the whole directory goes.
 */
TEST (nnstreamerDecoderPython3, openOtherDirectoryGrowsSysPathOnce)
{
  void *pdata = NULL;
  gchar *script = _script_path ();
  gchar *dir = g_dir_make_tmp ("nns_decoder_py3_XXXXXX", NULL);
  gchar *copy = NULL, *contents = NULL;
  gsize length = 0;
  Py_ssize_t before, after, again;
  const GstTensorDecoderDef *dec = _open_decoder (&pdata, "concat");

  ASSERT_NE (dec, nullptr);
  dec->exit (&pdata);
  ASSERT_NE (dir, nullptr);

  copy = g_build_filename (dir, "custom_decoder_copy.py", NULL);
  ASSERT_TRUE (g_file_get_contents (script, &contents, &length, NULL));
  ASSERT_TRUE (g_file_set_contents (copy, contents, length, NULL));

  before = py_test_sys_path_length ();
  ASSERT_GT (before, 0);
  ASSERT_TRUE (dec->init (&pdata));
  ASSERT_TRUE (dec->setOption (&pdata, 0, copy));
  after = py_test_sys_path_length ();
  dec->exit (&pdata);

  ASSERT_TRUE (dec->init (&pdata));
  ASSERT_TRUE (dec->setOption (&pdata, 0, copy));
  again = py_test_sys_path_length ();
  dec->exit (&pdata);

  EXPECT_EQ (after, before + 1);
  EXPECT_EQ (again, after);

  _remove_dir (dir);
  EXPECT_FALSE (g_file_test (dir, G_FILE_TEST_EXISTS));
  g_free (contents);
  g_free (copy);
  g_free (dir);
  g_free (script);
}

/**
 * @brief Opening the decoder again does not grow sys.path.
 */
TEST (nnstreamerDecoderPython3, reopenKeepsSysPath)
{
  void *pdata = NULL;
  Py_ssize_t first, last = -1;
  const GstTensorDecoderDef *dec = _open_decoder (&pdata, "concat");

  ASSERT_NE (dec, nullptr);
  dec->exit (&pdata);
  first = py_test_sys_path_length ();
  ASSERT_GT (first, 0);

  for (guint i = 0; i < 20; i++) {
    dec = _open_decoder (&pdata, "concat");
    ASSERT_NE (dec, nullptr);
    dec->exit (&pdata);
    last = py_test_sys_path_length ();
  }

  EXPECT_EQ (last, first);
}

/**
 * @brief A script that fails to load neither opens the decoder nor grows sys.path.
 */
TEST (nnstreamerDecoderPython3, openInvalidScript_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("python3");
  gchar *script = g_build_filename (g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH"),
      "tests", "test_models", "models", "NOT_EXIST_decoder.py", NULL);
  Py_ssize_t first = -1, last = -1;

  ASSERT_NE (dec, nullptr);
  for (guint i = 0; i < 10; i++) {
    void *pdata = NULL;

    ASSERT_TRUE (dec->init (&pdata));
    EXPECT_FALSE (dec->setOption (&pdata, 0, script));
    EXPECT_EQ (pdata, nullptr);
    if (i == 0)
      first = py_test_sys_path_length ();
    last = py_test_sys_path_length ();
  }

  EXPECT_GT (first, 0);
  EXPECT_EQ (last, first);
  g_free (script);
}

/**
 * @brief Count the failed GLib assertions (g_return_if_fail and its kin) of a log message.
 */
static void
_count_assertion (const gchar *log_domain, GLogLevelFlags log_level,
    const gchar *message, gpointer user_data)
{
  guint *count = (guint *) user_data;

  if (message && strstr (message, "assertion '") != NULL)
    (*count)++;
  g_log_default_handler (log_domain, log_level, message, NULL);
}

/**
 * @brief Log handlers counting failed assertions in the domains the decoder may hit.
 */
typedef struct {
  guint count;
  guint gst_id;
  guint glib_id;
  GLogFunc prev;
} AssertionCounter;

/**
 * @brief Start counting failed assertions.
 */
static void
_assertion_counter_start (AssertionCounter *counter)
{
  GLogLevelFlags levels = (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_LEVEL_WARNING);

  counter->count = 0;
  counter->gst_id
      = g_log_set_handler ("GStreamer", levels, _count_assertion, &counter->count);
  counter->glib_id
      = g_log_set_handler ("GLib", levels, _count_assertion, &counter->count);
  counter->prev = g_log_set_default_handler (_count_assertion, &counter->count);
}

/**
 * @brief Stop counting failed assertions.
 * @return the number of failed assertions logged since the start.
 */
static guint
_assertion_counter_stop (AssertionCounter *counter)
{
  g_log_set_default_handler (counter->prev, NULL);
  g_log_remove_handler ("GLib", counter->glib_id);
  g_log_remove_handler ("GStreamer", counter->gst_id);

  return counter->count;
}

/**
 * @brief Without a script the decoder refuses caps and decoding, and closes quietly.
 */
TEST (nnstreamerDecoderPython3, noScript_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("python3");
  void *pdata = NULL;
  GstTensorsConfig config;
  guint8 data[2 * TENSOR_SIZE] = { 0 };
  GstTensorMemory input[2];
  GstBuffer *outbuf;
  AssertionCounter counter;

  ASSERT_NE (dec, nullptr);
  ASSERT_TRUE (dec->init (&pdata));
  _init_config (&config);
  for (guint i = 0; i < 2; i++) {
    input[i].data = data + i * TENSOR_SIZE;
    input[i].size = TENSOR_SIZE;
  }

  _assertion_counter_start (&counter);
  EXPECT_EQ (dec->getOutCaps (&pdata, &config), nullptr);

  outbuf = gst_buffer_new ();
  EXPECT_EQ (dec->decode (&pdata, &config, input, outbuf), GST_FLOW_ERROR);
  EXPECT_EQ (gst_buffer_get_size (outbuf), 0U);
  gst_buffer_unref (outbuf);

  dec->exit (&pdata);
  EXPECT_EQ (_assertion_counter_stop (&counter), 0U);
  EXPECT_EQ (pdata, nullptr);

  gst_tensors_config_free (&config);
}

/**
 * @brief A script whose decoder class raises in its constructor is not loaded.
 */
TEST (nnstreamerDecoderPython3, openRaisingConstructor_n)
{
  void *pdata = NULL;
  const GstTensorDecoderDef *dec = _open_decoder (&pdata, "init-raise");

  EXPECT_EQ (dec, nullptr);
  EXPECT_EQ (pdata, nullptr);

  /* a wrongly opened decoder is closed so that the next cases start clean */
  if (pdata) {
    dec = nnstreamer_decoder_find ("python3");
    ASSERT_NE (dec, nullptr);
    dec->exit (&pdata);
  }
}

/**
 * @brief Run a pipeline with the python3 decoder until an error, EOS or a timeout.
 * @param option1 The option1 of the decoder, or NULL to leave it unset.
 * @return the type of the message that ended the run, GST_MESSAGE_UNKNOWN on a timeout,
 *         or GST_MESSAGE_ANY if the pipeline was not made.
 */
static GstMessageType
_run_decoder_pipeline (const gchar *option1)
{
  gchar *desc = g_strdup_printf (
      "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=4,height=4 ! "
      "tensor_converter ! tensor_decoder mode=python3 %s%s ! fakesink",
      option1 ? "option1=" : "", option1 ? option1 : "");
  GstElement *pipeline = gst_parse_launch (desc, NULL);
  GstMessageType type = GST_MESSAGE_ANY;

  g_free (desc);
  if (pipeline) {
    GstBus *bus = gst_element_get_bus (pipeline);
    GstMessage *msg;

    gst_element_set_state (pipeline, GST_STATE_PLAYING);
    msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
        (GstMessageType) (GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    type = msg ? GST_MESSAGE_TYPE (msg) : GST_MESSAGE_UNKNOWN;
    if (msg)
      gst_message_unref (msg);

    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (bus);
    gst_object_unref (pipeline);
  }

  return type;
}

/**
 * @brief A pipeline whose python3 decoder has no script fails with an error.
 */
TEST (nnstreamerDecoderPython3, pipelineNoScript_n)
{
  EXPECT_EQ (_run_decoder_pipeline (NULL), GST_MESSAGE_ERROR);
}

/**
 * @brief A pipeline whose python3 decoder failed to load its script fails with an error.
 */
TEST (nnstreamerDecoderPython3, pipelineInvalidScript_n)
{
  gchar *script = g_build_filename (g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH"),
      "tests", "test_models", "models", "NOT_EXIST_decoder.py", NULL);

  EXPECT_EQ (_run_decoder_pipeline (script), GST_MESSAGE_ERROR);
  g_free (script);
}

/**
 * @brief A pipeline with a valid script reaches EOS.
 */
TEST (nnstreamerDecoderPython3, pipelineScript)
{
  gchar *script = _script_path ();

  /* loading the sub-plugin starts the interpreter the environment is set in */
  ASSERT_NE (nnstreamer_decoder_find ("python3"), nullptr);
  ASSERT_TRUE (py_test_setenv ("NNS_TEST_PY_DECODER_MODE", "concat"));
  EXPECT_EQ (_run_decoder_pipeline (script), GST_MESSAGE_EOS);
  g_free (script);
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
