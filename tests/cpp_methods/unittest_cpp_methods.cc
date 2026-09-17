/**
 * @file        unittest_cpp_methods.cc
 * @date        15 Jan 2019
 * @brief       Unit test cases for tensor_filter::cpp
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>

#include <nnstreamer_util.h>
#include <unittest_util.h>
#include "../../gst/nnstreamer/tensor_filter/tensor_filter.h"
#include "cppfilter_test.hh"

static char *path_to_lib = NULL;

/** @brief Test C++ filter allocating its own output, otherwise the same as filter_basic */
class filter_self_alloc : public filter_basic
{
  public:
  /** @brief Construct the test filter with the given filter name */
  filter_self_alloc (const char *str) : filter_basic (str)
  {
  }

  /** @brief Tell the framework the filter allocates the output buffer */
  bool isAllocatedBeforeInvoke ()
  {
    return false;
  }

  /** @brief Allocate the output, then fill it as filter_basic does */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    out[0].data = g_malloc (out[0].size);
    return filter_basic::invoke (in, out);
  }
};

/** @brief Test C++ filter recording the tensor_filter properties its callbacks see */
class filter_prop_probe : public filter_basic
{
  public:
  const GstTensorFilterProperties *hold_for; /**< invoke with these properties waits for release */
  const GstTensorFilterProperties *release_from; /**< invoke with these properties releases the waiting one and waits until it has recorded */
  const GstTensorFilterProperties *seen_by_held; /**< properties the waiting invoke sees after release */
  const GstTensorFilterProperties *seen_by_releaser; /**< properties the releasing invoke sees after the waiting one has recorded */
  std::atomic<const GstTensorFilterProperties *> seen; /**< properties the latest callback saw */
  bool held; /**< the invoke with hold_for is waiting */
  bool released; /**< let the waiting invoke go on */
  bool recorded; /**< the released invoke has set seen_by_held */
  GMutex lock; /**< protects the members above except seen */
  GCond cond; /**< signals held, released and recorded */

  /** @brief Construct the test filter with the given filter name */
  filter_prop_probe (const char *str)
      : filter_basic (str), hold_for (nullptr), release_from (nullptr),
        seen_by_held (nullptr), seen_by_releaser (nullptr), seen (nullptr),
        held (false), released (false), recorded (false)
  {
    g_mutex_init (&lock);
    g_cond_init (&cond);
  }

  /** @brief Destructor of the test filter */
  ~filter_prop_probe ()
  {
    g_cond_clear (&cond);
    g_mutex_clear (&lock);
  }

  /** @brief The properties the filter would see on the calling thread now */
  const GstTensorFilterProperties *current ()
  {
    return prop;
  }

  /** @brief Record the properties, then report the input dimension */
  int getInputDim (GstTensorsInfo *info)
  {
    seen = prop;
    return filter_basic::getInputDim (info);
  }

  /** @brief Record the properties, then report the output dimension */
  int getOutputDim (GstTensorsInfo *info)
  {
    seen = prop;
    return filter_basic::getOutputDim (info);
  }

  /** @brief Record the properties, then reject the dimension */
  int setInputDim (const GstTensorsInfo *in, GstTensorsInfo *out)
  {
    seen = prop;
    return filter_basic::setInputDim (in, out);
  }

  /** @brief Wait for or give the release if asked to, record the properties, then run filter_basic */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    gint64 end = g_get_monotonic_time () + 5 * G_TIME_SPAN_SECOND;

    g_mutex_lock (&lock);
    if (hold_for && prop == hold_for) {
      held = true;
      g_cond_broadcast (&cond);
      while (!released && g_cond_wait_until (&cond, &lock, end))
        ;
      seen_by_held = prop;
      recorded = true;
      g_cond_broadcast (&cond);
    } else if (release_from && prop == release_from) {
      released = true;
      g_cond_broadcast (&cond);
      while (!recorded && g_cond_wait_until (&cond, &lock, end))
        ;
      seen_by_releaser = prop;
    }
    g_mutex_unlock (&lock);

    seen = prop;
    return filter_basic::invoke (in, out);
  }
};

/** @brief Arguments of an invoke run on another thread */
typedef struct {
  const GstTensorFilterFramework *fw; /**< the cpp framework */
  GstTensorFilterProperties *prop; /**< properties of the invoking element */
  void *private_data; /**< opened filter */
  GstTensorMemory in; /**< input tensor */
  GstTensorMemory out; /**< output tensor */
  int ret; /**< invoke result */
} invoke_thread_args;

/** @brief Invoke an opened cpp filter on another thread */
static gpointer
_invoke_thread (gpointer data)
{
  invoke_thread_args *args = (invoke_thread_args *) data;

  args->ret = args->fw->invoke_NN (args->prop, &args->private_data, &args->in, &args->out);
  return NULL;
}

/**
 * @brief Prepare the properties a tensor_filter opens a filter_basic-shaped cpp model with
 */
static void
_init_basic_prop (GstTensorFilterProperties *prop, const gchar **models)
{
  filter_basic shape ("b10b11_shape");

  memset (prop, 0, sizeof (*prop));
  prop->fwname = "cpp";
  prop->model_files = models;
  prop->num_models = 1;
  shape.getInputDim (&prop->input_meta);
  shape.getOutputDim (&prop->output_meta);
}

/**
 * @brief Ask the tensor_filter framework whether an opened cpp filter allocates its output in invoke
 */
static gboolean
_allocates_in_invoke (void *private_data)
{
  GstTensorFilterPrivate priv;

  memset (&priv, 0, sizeof (priv));
  priv.fw = nnstreamer_filter_find ("cpp");
  priv.privateData = private_data;

  return gst_tensor_filter_allocate_in_invoke (&priv);
}

/** @brief Positive case for the simplest execution path */
TEST (cppFilterOnDemand, basic01)
{
  filter_basic basic ("basic_01");
  EXPECT_EQ (basic._register (), 0);
  EXPECT_EQ (basic._unregister (), 0);
}

/** @brief Negative case for the simplest execution path */
TEST (cppFilterOnDemand, basic02_n)
{
  filter_basic basic ("basic_02");
  EXPECT_NE (basic._unregister (), 0);
  EXPECT_EQ (basic._register (), 0);
  EXPECT_NE (basic._register (), 0);
  EXPECT_EQ (basic._unregister (), 0);
  EXPECT_NE (basic._unregister (), 0);
}

/** @brief Negative case for the simplest execution path w/ static calls */
TEST (cppFilterOnDemand, basic03_n)
{
  filter_basic basic ("basic_03");
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_EQ (filter_basic::__register (&basic), 0);
  EXPECT_NE (filter_basic::__register (&basic), 0);
  EXPECT_EQ (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
}

/** @brief Negative case for the simplest execution path w/ static calls */
TEST (cppFilterOnDemand, basic04_n)
{
  filter_basic basic ("basic_04");
  EXPECT_NE (filter_basic::__unregister ("basic_xx"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_04"), 0);
  EXPECT_EQ (filter_basic::__register (&basic), 0);
  EXPECT_NE (filter_basic::__register (&basic), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_xx"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_EQ (filter_basic::__unregister ("basic_04"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_04"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_xx"), 0);
}

/** @brief Wait until the pipeline saving the file */
static void
_wait_save_files (const gchar *file, gsize expected_len)
{
  gchar *content = NULL;
  gsize len;

  _wait_pipeline_save_files (file, content, len, expected_len, 1000U);

  g_free (content);
  content = NULL;
}

/** @brief Actual GST Pipeline with cpp on demand */
TEST (cppFilterOnDemand, pipeline01)
{
  filter_basic basic ("pl01");
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_EQ (basic._register (), 0);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue name=q1 ! tensor_filter framework=cpp model=pl01 ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue name=q2 ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
      tmp1, tmp2);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);

    _wait_save_files (tmp1, 480);
    _wait_save_files (tmp2, 240);

    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
    g_usleep (100000);

    gst_object_unref (pipeline);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);

  EXPECT_EQ (basic._unregister (), 0);
}

/** @brief Negative case for the simplest execution path */
TEST (cppFilterOnDemand, unregistered01_n)
{
  filter_basic basic ("basic_01");
  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! "
      "tensor_filter framework=cpp model=XXbasic_01 ! fakesink");

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  EXPECT_EQ (basic._register (), 0);
  gst_object_unref (pipeline);

  pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);
  if (pipeline)
    gst_object_unref (pipeline);
  EXPECT_EQ (basic._unregister (), 0);

  basic._unregister ();
  g_free (str_pipeline);
  EXPECT_NE (basic._unregister (), 0);
}

/** @brief gtest method */
TEST (cppFilterObj, base01_n)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue name=q1 ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue name=q2 ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_03,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
      path_to_lib, tmp1, tmp2, path_to_lib, tmp3);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

    gst_object_unref (pipeline);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
  removeTempFile (&tmp3);
}

/** @brief gtest method */
TEST (cppFilterObj, base02_n)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue name=q1 ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue name=q2 ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_03,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
      path_to_lib, tmp1, tmp2, path_to_lib, tmp3);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

    gst_object_unref (pipeline);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
  removeTempFile (&tmp3);
}

/** @brief gtest method */
TEST (cppFilterObj, base03)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();
  char *tmp4 = getTempFilename ();
  char *tmp5 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);
  EXPECT_NE (tmp4, nullptr);
  EXPECT_NE (tmp5, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_02,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=16,height=16,format=RGB ! tensor_converter ! tee name=t2 "
      "t2. ! queue ! tensor_filter framework=cpp model=basic_so2,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t2. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false ",
      path_to_lib, tmp1, tmp2, path_to_lib, tmp3, path_to_lib, tmp4, tmp5);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);
    _wait_save_files (tmp1, 480);
    _wait_save_files (tmp2, 240);
    _wait_save_files (tmp3, 480);
    _wait_save_files (tmp4, 7680);
    _wait_save_files (tmp5, 3840);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

    gst_object_unref (pipeline);
    g_usleep (300000);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp3), 0);
    EXPECT_EQ (filter_basic2::resultCompare (tmp5, tmp4), 0);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
  removeTempFile (&tmp3);
  removeTempFile (&tmp4);
  removeTempFile (&tmp5);
}

/** @brief Each opened cpp filter keeps its own output allocation contract */
TEST (cppFilterShared, allocContractPerFilter)
{
  filter_basic pre ("b10_pre");
  filter_self_alloc self ("b10_self");
  const gchar *pre_models[] = { "b10_pre" };
  const gchar *self_models[] = { "b10_self" };
  GstTensorFilterProperties prop_pre, prop_self, prop_pre2;
  void *pd_pre = NULL, *pd_self = NULL, *pd_pre2 = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (pre._register (), 0);
  EXPECT_EQ (self._register (), 0);
  _init_basic_prop (&prop_pre, pre_models);
  _init_basic_prop (&prop_self, self_models);
  _init_basic_prop (&prop_pre2, pre_models);

  EXPECT_EQ (fw->open (&prop_pre, &pd_pre), 0);
  EXPECT_EQ (fw->open (&prop_self, &pd_self), 0);
  EXPECT_FALSE (_allocates_in_invoke (pd_pre));
  EXPECT_TRUE (_allocates_in_invoke (pd_self));

  EXPECT_EQ (fw->open (&prop_pre2, &pd_pre2), 0);
  EXPECT_FALSE (_allocates_in_invoke (pd_pre));
  EXPECT_TRUE (_allocates_in_invoke (pd_self));
  EXPECT_FALSE (_allocates_in_invoke (pd_pre2));

  fw->close (&prop_pre2, &pd_pre2);
  fw->close (&prop_self, &pd_self);
  fw->close (&prop_pre, &pd_pre);
  EXPECT_EQ (self._unregister (), 0);
  EXPECT_EQ (pre._unregister (), 0);
}

/** @brief A failed open does not change the contract of an opened cpp filter */
TEST (cppFilterShared, allocContractFailedOpen_n)
{
  filter_self_alloc self ("b10_self_n");
  const gchar *self_models[] = { "b10_self_n" };
  const gchar *unknown_models[] = { "b10_unknown_n" };
  GstTensorFilterProperties prop_self, prop_unknown;
  void *pd_self = NULL, *pd_unknown = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (self._register (), 0);
  _init_basic_prop (&prop_self, self_models);
  _init_basic_prop (&prop_unknown, unknown_models);

  EXPECT_EQ (fw->open (&prop_self, &pd_self), 0);
  EXPECT_NE (fw->open (&prop_unknown, &pd_unknown), 0);
  EXPECT_EQ (pd_unknown, nullptr);
  EXPECT_TRUE (_allocates_in_invoke (pd_self));

  fw->close (&prop_self, &pd_self);
  EXPECT_EQ (self._unregister (), 0);
}

/** @brief A preallocated and a self-allocating cpp filter both produce their output in one pipeline */
TEST (cppFilterShared, allocContractPipeline)
{
  filter_basic pre ("b10_pl_pre");
  filter_self_alloc self ("b10_pl_self");
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);
  EXPECT_EQ (pre._register (), 0);
  EXPECT_EQ (self._register (), 0);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue ! tensor_filter framework=cpp model=b10_pl_pre ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_filter framework=cpp model=b10_pl_self ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
      tmp1, tmp2, tmp3);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);
    _wait_save_files (tmp1, 480);
    _wait_save_files (tmp2, 240);
    _wait_save_files (tmp3, 480);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

    gst_object_unref (pipeline);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp3), 0);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
  removeTempFile (&tmp3);

  EXPECT_EQ (self._unregister (), 0);
  EXPECT_EQ (pre._unregister (), 0);
}

/** @brief A shared cpp filter sees the properties of the element calling it, not of the last one opened */
TEST (cppFilterShared, propFollowsCaller)
{
  filter_prop_probe probe ("b11_probe");
  const gchar *models[] = { "b11_probe" };
  GstTensorFilterProperties prop1, prop2;
  GstTensorsInfo info;
  void *pd1 = NULL, *pd2 = NULL;
  uint8_t in_data[48] = { 0 };
  uint8_t out_data[96];
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out = { out_data, sizeof (out_data) };
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (probe._register (), 0);
  _init_basic_prop (&prop1, models);
  _init_basic_prop (&prop2, models);

  EXPECT_EQ (fw->open (&prop1, &pd1), 0);
  EXPECT_EQ (fw->open (&prop2, &pd2), 0);
  fw->close (&prop2, &pd2);

  EXPECT_EQ (fw->getInputDimension (&prop1, &pd1, &info), 0);
  EXPECT_EQ (probe.seen.load (), &prop1);
  probe.seen = nullptr;
  EXPECT_EQ (fw->getOutputDimension (&prop1, &pd1, &info), 0);
  EXPECT_EQ (probe.seen.load (), &prop1);
  probe.seen = nullptr;
  EXPECT_NE (fw->setInputDimension (&prop1, &pd1, &prop1.input_meta, &info), 0);
  EXPECT_EQ (probe.seen.load (), &prop1);
  probe.seen = nullptr;
  EXPECT_EQ (fw->invoke_NN (&prop1, &pd1, &in, &out), 0);
  EXPECT_EQ (probe.seen.load (), &prop1);
  EXPECT_EQ (out_data[0], 0);
  EXPECT_EQ (out_data[48], 1);

  fw->close (&prop1, &pd1);
  EXPECT_EQ (probe._unregister (), 0);
}

/** @brief A shared cpp filter has no element properties outside a callback */
TEST (cppFilterShared, propOutsideCallback_n)
{
  filter_prop_probe probe ("b11_outside_n");
  const gchar *models[] = { "b11_outside_n" };
  GstTensorFilterProperties prop1;
  GstTensorsInfo info;
  void *pd1 = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (probe._register (), 0);
  _init_basic_prop (&prop1, models);

  EXPECT_EQ (fw->open (&prop1, &pd1), 0);
  EXPECT_EQ (probe.current (), nullptr);
  EXPECT_EQ (fw->getInputDimension (&prop1, &pd1, &info), 0);
  EXPECT_EQ (probe.seen.load (), &prop1);
  EXPECT_EQ (probe.current (), nullptr);

  fw->close (&prop1, &pd1);
  EXPECT_EQ (probe.current (), nullptr);
  EXPECT_EQ (probe._unregister (), 0);
}

/** @brief Elements invoking a shared cpp filter at the same time each keep their own properties */
TEST (cppFilterShared, propPerThread)
{
  filter_prop_probe probe ("b11_thread");
  const gchar *models[] = { "b11_thread" };
  GstTensorFilterProperties prop1, prop2;
  void *pd2 = NULL;
  uint8_t in1[48] = { 0 }, in2[48] = { 0 };
  uint8_t out1[96], out2[96];
  GstTensorMemory in = { in2, sizeof (in2) };
  GstTensorMemory out = { out2, sizeof (out2) };
  invoke_thread_args args;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");
  gint64 end;
  bool held;

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (probe._register (), 0);
  _init_basic_prop (&prop1, models);
  _init_basic_prop (&prop2, models);

  args.fw = fw;
  args.prop = &prop1;
  args.private_data = NULL;
  args.in.data = in1;
  args.in.size = sizeof (in1);
  args.out.data = out1;
  args.out.size = sizeof (out1);
  args.ret = -1;
  EXPECT_EQ (fw->open (&prop1, &args.private_data), 0);
  probe.hold_for = &prop1;
  probe.release_from = &prop2;

  GThread *thread = g_thread_new ("b11_invoke", _invoke_thread, &args);

  g_mutex_lock (&probe.lock);
  end = g_get_monotonic_time () + 5 * G_TIME_SPAN_SECOND;
  while (!probe.held && g_cond_wait_until (&probe.cond, &probe.lock, end))
    ;
  held = probe.held;
  g_mutex_unlock (&probe.lock);
  EXPECT_TRUE (held);

  /* The waiting invoke records its properties while this one is still running */
  EXPECT_EQ (fw->open (&prop2, &pd2), 0);
  EXPECT_EQ (fw->invoke_NN (&prop2, &pd2, &in, &out), 0);
  fw->close (&prop2, &pd2);

  g_mutex_lock (&probe.lock);
  EXPECT_TRUE (probe.recorded);
  probe.released = true;
  g_cond_broadcast (&probe.cond);
  g_mutex_unlock (&probe.lock);
  g_thread_join (thread);

  EXPECT_EQ (args.ret, 0);
  EXPECT_EQ (probe.seen_by_held, &prop1);
  EXPECT_EQ (probe.seen_by_releaser, &prop2);

  fw->close (&prop1, &args.private_data);
  EXPECT_EQ (probe._unregister (), 0);
}

/** @brief Count the buffers a fakesink receives */
static void
_handoff_count (GstElement *sink, GstBuffer *buffer, GstPad *pad, gpointer user_data)
{
  UNUSED (sink);
  UNUSED (buffer);
  UNUSED (pad);
  g_atomic_int_inc ((gint *) user_data);
}

/** @brief A cpp filter keeps working after another element sharing it is destroyed */
TEST (cppFilterShared, propAfterPeerDestroyed)
{
  filter_prop_probe probe ("b11_pl");
  gint received = 0;
  GstFlowReturn flow = GST_FLOW_ERROR;

  GstElement *pipeline1 = gst_parse_launch (
      "appsrc name=src caps=other/tensors,num_tensors=1,dimensions=(string)3:4:4:1,types=(string)uint8,format=static,framerate=(fraction)0/1 ! "
      "tensor_filter name=filter framework=cpp model=b11_pl ! fakesink name=sink signal-handoffs=true sync=false async=false",
      NULL);
  ASSERT_NE (pipeline1, nullptr);
  EXPECT_EQ (probe._register (), 0);

  GstElement *src = gst_bin_get_by_name (GST_BIN (pipeline1), "src");
  GstElement *filter = gst_bin_get_by_name (GST_BIN (pipeline1), "filter");
  GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline1), "sink");
  EXPECT_NE (src, nullptr);
  EXPECT_NE (filter, nullptr);
  EXPECT_NE (sink, nullptr);
  g_signal_connect (sink, "handoff", G_CALLBACK (_handoff_count), &received);
  EXPECT_EQ (setPipelineStateSync (pipeline1, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  GstElement *pipeline2 = gst_parse_launch (
      "videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,width=4,height=4,format=RGB ! "
      "tensor_converter ! tensor_filter framework=cpp model=b11_pl ! fakesink sync=false async=false",
      NULL);
  EXPECT_NE (pipeline2, nullptr);
  if (pipeline2) {
    EXPECT_EQ (setPipelineStateSync (pipeline2, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);
    GstBus *bus = gst_element_get_bus (pipeline2);
    GstMessage *msg = gst_bus_timed_pop_filtered (bus, 5 * GST_SECOND,
        (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    EXPECT_NE (msg, nullptr);
    if (msg) {
      EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_EOS);
      gst_message_unref (msg);
    }
    gst_object_unref (bus);
    EXPECT_EQ (setPipelineStateSync (pipeline2, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
    gst_object_unref (pipeline2);
  }

  GstBuffer *buf = gst_buffer_new_allocate (NULL, 48, NULL);
  gst_buffer_memset (buf, 0, 0, 48);
  probe.seen = nullptr;
  g_signal_emit_by_name (src, "push-buffer", buf, &flow);
  gst_buffer_unref (buf);
  EXPECT_EQ (flow, GST_FLOW_OK);

  for (int i = 0; i < 500 && g_atomic_int_get (&received) < 1; i++)
    g_usleep (10000);
  EXPECT_EQ (g_atomic_int_get (&received), 1);
  if (filter) {
    EXPECT_EQ (probe.seen.load (), &GST_TENSOR_FILTER_CAST (filter)->priv.prop);
  }

  EXPECT_EQ (setPipelineStateSync (pipeline1, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  if (src)
    gst_object_unref (src);
  if (filter)
    gst_object_unref (filter);
  if (sink)
    gst_object_unref (sink);
  gst_object_unref (pipeline1);
  EXPECT_EQ (probe._unregister (), 0);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = 0;
  int delete_path = 0;

  if (argc > 3 && !g_strcmp0 (argv[1], "-libpath")) {
    path_to_lib = argv[2];
  } else {
    gchar *dir = g_path_get_dirname (argv[0]);
    path_to_lib = g_strdup_printf ("%s/", dir);
    delete_path = 1;
    g_free (dir);
    g_printerr ("LIBPATH = %s\n", path_to_lib);
  }

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

  if (delete_path)
    g_free (path_to_lib);
  return result;
}
