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
#include "../../gst/nnstreamer/tensor_filter/tensor_filter_single.h"
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
  std::atomic<const GstTensorFilterProperties *> seen_get_in; /**< properties the latest getInputDim saw */
  std::atomic<const GstTensorFilterProperties *> seen_get_out; /**< properties the latest getOutputDim saw */
  std::atomic<const GstTensorFilterProperties *> seen_set_in; /**< properties the latest setInputDim saw */
  bool held; /**< the invoke with hold_for is waiting */
  bool released; /**< let the waiting invoke go on */
  bool recorded; /**< the released invoke has set seen_by_held */
  GMutex lock; /**< protects the members above except seen */
  GCond cond; /**< signals held, released and recorded */

  /** @brief Construct the test filter with the given filter name */
  filter_prop_probe (const char *str)
      : filter_basic (str), hold_for (nullptr), release_from (nullptr),
        seen_by_held (nullptr), seen_by_releaser (nullptr), seen (nullptr),
        seen_get_in (nullptr), seen_get_out (nullptr), seen_set_in (nullptr),
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
    seen = seen_get_in = prop;
    return filter_basic::getInputDim (info);
  }

  /** @brief Record the properties, then report the output dimension */
  int getOutputDim (GstTensorsInfo *info)
  {
    seen = seen_get_out = prop;
    return filter_basic::getOutputDim (info);
  }

  /** @brief Record the properties, then reject the dimension */
  int setInputDim (const GstTensorsInfo *in, GstTensorsInfo *out)
  {
    seen = seen_set_in = prop;
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

/** @brief Test C++ filter dropping every odd frame, otherwise the same as filter_basic */
class filter_drop_odd : public filter_basic
{
  public:
  std::atomic_uint calls; /**< number of invoke calls */

  /** @brief Construct the test filter with the given filter name */
  filter_drop_odd (const char *str) : filter_basic (str), calls (0)
  {
  }

  /** @brief Ask tensor_filter to drop the 1st, 3rd, ... frame and fill the others */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    if (++calls % 2)
      return 1;
    return filter_basic::invoke (in, out);
  }
};

/** @brief Test C++ filter failing every invoke with -ENODEV */
class filter_invoke_fail : public filter_basic
{
  public:
  /** @brief Construct the test filter with the given filter name */
  filter_invoke_fail (const char *str) : filter_basic (str)
  {
  }

  /** @brief Fail with an error code other than -EINVAL */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    UNUSED (in);
    UNUSED (out);
    return -ENODEV;
  }
};

/** @brief Fill info with uint8 3:width:width:1 */
static void
_fill_rgb_info (GstTensorsInfo *info, guint width)
{
  gst_tensors_info_init (info);
  info->num_tensors = 1;
  info->info[0].type = _NNS_UINT8;
  info->info[0].dimension[0] = 3;
  info->info[0].dimension[1] = width;
  info->info[0].dimension[2] = width;
  info->info[0].dimension[3] = 1;
}

/**
 * @brief Test C++ filter writing out = in + 1, with a fixed uint8 3:4:4:1 input
 *        and output only when asked, and a setInputDim that copies the input
 *        to the output or fails
 */
class filter_dims : public tensor_filter_cpp
{
  public:
  const bool fixed_in; /**< getInputDim reports uint8 3:4:4:1 */
  const bool fixed_out; /**< getOutputDim reports uint8 3:4:4:1 */
  const bool set_ok; /**< setInputDim gives the output the input dimension */

  /** @brief Construct the test filter with the given filter name and dimension callbacks */
  filter_dims (const char *str, bool in, bool out, bool set)
      : tensor_filter_cpp (str), fixed_in (in), fixed_out (out), set_ok (set)
  {
  }

  /** @brief Report the fixed input dimension, if any */
  int getInputDim (GstTensorsInfo *info)
  {
    if (!fixed_in)
      return -EINVAL;
    _fill_rgb_info (info, 4);
    return 0;
  }

  /** @brief Report the fixed output dimension, if any */
  int getOutputDim (GstTensorsInfo *info)
  {
    if (!fixed_out)
      return -EINVAL;
    _fill_rgb_info (info, 4);
    return 0;
  }

  /** @brief Give the output the dimension and type of the input, or fail */
  int setInputDim (const GstTensorsInfo *in, GstTensorsInfo *out)
  {
    if (!set_ok)
      return -EINVAL;
    gst_tensors_info_copy (out, in);
    return 0;
  }

  /** @brief Tell the framework the output buffer is preallocated */
  bool isAllocatedBeforeInvoke ()
  {
    return true;
  }

  /** @brief Write in + 1 to the output */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    for (size_t i = 0; i < in[0].size && i < out[0].size; i++)
      ((uint8_t *) out[0].data)[i] = ((uint8_t *) in[0].data)[i] + 1;
    return 0;
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

  args->ret = args->fw->invoke (
      args->fw, args->prop, args->private_data, &args->in, &args->out);
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
  if (priv.fw->getFrameworkInfo (priv.fw, &priv.prop, private_data, &priv.info) != 0)
    return FALSE;

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
  GstTensorsInfo info, out_info;
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

  EXPECT_EQ (fw->getModelInfo (fw, &prop1, pd1, GET_IN_OUT_INFO, &info, &out_info), 0);
  EXPECT_EQ (probe.seen_get_in.load (), &prop1);
  EXPECT_EQ (probe.seen_get_out.load (), &prop1);
  EXPECT_NE (fw->getModelInfo (fw, &prop1, pd1, SET_INPUT_INFO, &prop1.input_meta, &info), 0);
  EXPECT_EQ (probe.seen_set_in.load (), &prop1);
  probe.seen = nullptr;
  EXPECT_EQ (fw->invoke (fw, &prop1, pd1, &in, &out), 0);
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
  GstTensorsInfo info, out_info;
  void *pd1 = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (probe._register (), 0);
  _init_basic_prop (&prop1, models);

  EXPECT_EQ (fw->open (&prop1, &pd1), 0);
  EXPECT_EQ (probe.current (), nullptr);
  EXPECT_EQ (fw->getModelInfo (fw, &prop1, pd1, GET_IN_OUT_INFO, &info, &out_info), 0);
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
  EXPECT_EQ (fw->invoke (fw, &prop2, pd2, &in, &out), 0);
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

/** @brief The cpp sub-plugin registers itself as a v1 framework with its fixed properties */
TEST (cppFilterV1, frameworkInfo)
{
  filter_self_alloc self ("v1_info");
  const gchar *models[] = { "v1_info" };
  GstTensorFilterFrameworkInfo info;
  GstTensorFilterProperties prop;
  void *pd = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (fw->version, GST_TENSOR_FILTER_FRAMEWORK_V1);

  memset (&prop, 0, sizeof (prop));
  memset (&info, 0xff, sizeof (info));
  EXPECT_EQ (fw->getFrameworkInfo (fw, &prop, NULL, &info), 0);
  EXPECT_STREQ (info.name, "cpp");
  EXPECT_FALSE (info.allow_in_place);
  EXPECT_FALSE (info.allocate_in_invoke);
  EXPECT_FALSE (info.run_without_model);
  EXPECT_FALSE (info.verify_model_path);
  EXPECT_EQ (info.hw_list, nullptr);
  EXPECT_EQ (info.num_hw, 0);
  EXPECT_EQ (info.accl_auto, ACCL_NONE);
  EXPECT_EQ (info.accl_default, ACCL_NONE);
  EXPECT_EQ (info.statistics, nullptr);

  /* Once opened, the filter tells its own allocation contract */
  EXPECT_EQ (self._register (), 0);
  _init_basic_prop (&prop, models);
  EXPECT_EQ (fw->open (&prop, &pd), 0);
  EXPECT_EQ (fw->getFrameworkInfo (fw, &prop, pd, &info), 0);
  EXPECT_STREQ (info.name, "cpp");
  EXPECT_TRUE (info.allocate_in_invoke);

  fw->close (&prop, &pd);
  EXPECT_EQ (self._unregister (), 0);
}

/** @brief Model info requests without an opened filter or an info to fill are refused */
TEST (cppFilterV1, modelInfoInvalidArgs_n)
{
  filter_basic basic ("v1_args_n");
  const gchar *models[] = { "v1_args_n" };
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;
  void *pd = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (basic._register (), 0);
  _init_basic_prop (&prop, models);

  EXPECT_EQ (fw->getModelInfo (fw, &prop, NULL, GET_IN_OUT_INFO, &in_info, &out_info), -EINVAL);
  EXPECT_EQ (fw->getModelInfo (fw, &prop, NULL, SET_INPUT_INFO, &prop.input_meta, &out_info),
      -EINVAL);

  EXPECT_EQ (fw->open (&prop, &pd), 0);
  EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, GET_IN_OUT_INFO, NULL, &out_info), -EINVAL);
  EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, GET_IN_OUT_INFO, &in_info, NULL), -EINVAL);
  EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, SET_INPUT_INFO, NULL, &out_info), -EINVAL);
  EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, SET_INPUT_INFO, &prop.input_meta, NULL), -EINVAL);

  fw->close (&prop, &pd);
  EXPECT_EQ (basic._unregister (), 0);
}

/** @brief The cpp sub-plugin handles no event, so tensor_filter keeps its defaults */
TEST (cppFilterV1, eventNotHandled_n)
{
  filter_basic basic ("v1_event_n");
  const gchar *models[] = { "v1_event_n" };
  GstTensorFilterProperties prop;
  GstTensorFilterFrameworkEventData data;
  void *pd = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (basic._register (), 0);
  _init_basic_prop (&prop, models);
  memset (&data, 0, sizeof (data));

  EXPECT_EQ (fw->eventHandler (fw, &prop, NULL, RELOAD_MODEL, NULL), -ENOENT);
  EXPECT_EQ (fw->open (&prop, &pd), 0);
  EXPECT_EQ (fw->eventHandler (fw, &prop, pd, RELOAD_MODEL, &data), -ENOENT);
  EXPECT_EQ (fw->eventHandler (fw, &prop, pd, DESTROY_NOTIFY, &data), -ENOENT);
  EXPECT_EQ (fw->eventHandler (fw, &prop, pd, CUSTOM_PROP, &data), -ENOENT);

  fw->close (&prop, &pd);
  EXPECT_EQ (basic._unregister (), 0);
}

/** @brief A model info request other than GET_IN_OUT_INFO or SET_INPUT_INFO is refused */
TEST (cppFilterV1, modelInfoUnknownOps_n)
{
  filter_basic basic ("v1_ops_n");
  const gchar *models[] = { "v1_ops_n" };
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;
  void *pd = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (basic._register (), 0);
  _init_basic_prop (&prop, models);

  EXPECT_EQ (fw->open (&prop, &pd), 0);
  EXPECT_EQ (fw->getModelInfo (fw, &prop, pd,
                 (model_info_ops) (SET_INPUT_INFO + 1), &in_info, &out_info),
      -ENOENT);

  fw->close (&prop, &pd);
  EXPECT_EQ (basic._unregister (), 0);
}

/** @brief GET_IN_OUT_INFO gives a filter with only a fixed input the output of setInputDim */
TEST (cppFilterV1, modelInfoFixedInputOnly)
{
  filter_dims filter ("v1_in_only", true, false, true);
  const gchar *models[] = { "v1_in_only" };
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info, expected;
  void *pd = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (filter._register (), 0);
  _init_basic_prop (&prop, models);
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  _fill_rgb_info (&expected, 4);

  EXPECT_EQ (fw->open (&prop, &pd), 0);
  EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, GET_IN_OUT_INFO, &in_info, &out_info), 0);
  EXPECT_TRUE (gst_tensors_info_is_equal (&in_info, &expected));
  EXPECT_TRUE (gst_tensors_info_is_equal (&out_info, &expected));

  fw->close (&prop, &pd);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  gst_tensors_info_free (&expected);
  EXPECT_EQ (filter._unregister (), 0);
}

/** @brief GET_IN_OUT_INFO fails for a filter with only a fixed input whose setInputDim fails */
TEST (cppFilterV1, modelInfoFixedInputOnly_n)
{
  filter_dims filter ("v1_in_only_n", true, false, false);
  const gchar *models[] = { "v1_in_only_n" };
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;
  void *pd = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (filter._register (), 0);
  _init_basic_prop (&prop, models);
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  EXPECT_EQ (fw->open (&prop, &pd), 0);
  EXPECT_NE (fw->getModelInfo (fw, &prop, pd, GET_IN_OUT_INFO, &in_info, &out_info), 0);

  fw->close (&prop, &pd);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  EXPECT_EQ (filter._unregister (), 0);
}

/** @brief SET_INPUT_INFO gives a filter with only a fixed output that output when setInputDim fails */
TEST (cppFilterV1, modelInfoFixedOutputOnly)
{
  filter_dims filter ("v1_out_only", false, true, false);
  const gchar *models[] = { "v1_out_only" };
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info, expected;
  void *pd = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (filter._register (), 0);
  _init_basic_prop (&prop, models);
  gst_tensors_info_init (&out_info);
  _fill_rgb_info (&in_info, 16);
  _fill_rgb_info (&expected, 4);

  EXPECT_EQ (fw->open (&prop, &pd), 0);
  EXPECT_NE (fw->getModelInfo (fw, &prop, pd, GET_IN_OUT_INFO, &in_info, &out_info), 0);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  _fill_rgb_info (&in_info, 16);
  EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, SET_INPUT_INFO, &in_info, &out_info), 0);
  EXPECT_TRUE (gst_tensors_info_is_equal (&out_info, &expected));

  fw->close (&prop, &pd);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  gst_tensors_info_free (&expected);
  EXPECT_EQ (filter._unregister (), 0);
}

/** @brief The error code of a failing invoke reaches tensor_filter unchanged */
TEST (cppFilterV1, invokeErrorCode_n)
{
  filter_invoke_fail filter ("v1_fail_n");
  const gchar *models[] = { "v1_fail_n" };
  GstTensorFilterProperties prop;
  uint8_t in_data[48] = { 0 };
  uint8_t out_data[96];
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out = { out_data, sizeof (out_data) };
  void *pd = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (filter._register (), 0);
  _init_basic_prop (&prop, models);

  EXPECT_EQ (fw->open (&prop, &pd), 0);
  EXPECT_EQ (fw->invoke (fw, &prop, pd, &in, &out), -ENODEV);

  fw->close (&prop, &pd);
  EXPECT_EQ (filter._unregister (), 0);
}

/** @brief Run a pipeline until the expected EOS or error and count the buffers its fakesink named sink receives */
static gint
_run_and_count (const gchar *str_pipeline, GstMessageType expected = GST_MESSAGE_EOS)
{
  gint received = 0;
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);

  EXPECT_NE (pipeline, nullptr);
  if (!pipeline)
    return -1;

  GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  EXPECT_NE (sink, nullptr);
  if (sink) {
    g_signal_connect (sink, "handoff", G_CALLBACK (_handoff_count), &received);
    gst_object_unref (sink);
  }

  int ret = setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  if (expected == GST_MESSAGE_EOS) {
    EXPECT_EQ (ret, 0);
  }
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *msg = gst_bus_timed_pop_filtered (bus, 5 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  EXPECT_NE (msg, nullptr);
  if (msg) {
    EXPECT_EQ (GST_MESSAGE_TYPE (msg), expected);
    gst_message_unref (msg);
  }
  gst_object_unref (bus);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (pipeline);

  return g_atomic_int_get (&received);
}

/** @brief A positive invoke result drops the frame instead of failing the stream */
TEST (cppFilterV1, invokeDrop)
{
  filter_drop_odd filter ("v1_drop");
  gint received;

  EXPECT_EQ (filter._register (), 0);
  received = _run_and_count (
      "videotestsrc num-buffers=4 ! videoconvert ! videoscale ! video/x-raw,width=4,height=4,format=RGB ! "
      "tensor_converter ! tensor_filter framework=cpp model=v1_drop ! "
      "fakesink name=sink signal-handoffs=true sync=false async=false");

  EXPECT_EQ (filter.calls.load (), 4U);
  EXPECT_EQ (received, 2);
  EXPECT_EQ (filter._unregister (), 0);
}

/** @brief Run a 4x4 RGB stream through the cpp filter model and check out = in + 1 */
static void
_check_plus_one_pipeline (const gchar *model)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  gchar *in_data = NULL, *out_data = NULL;
  gsize in_len = 0, out_len = 0;

  ASSERT_NE (tmp1, nullptr);
  ASSERT_NE (tmp2, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=4,height=4,format=RGB ! "
      "tensor_converter ! tee name=t "
      "t. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_filter framework=cpp model=%s ! tee name=t2 "
      "t2. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t2. ! queue ! fakesink name=sink signal-handoffs=true sync=false async=false",
      tmp1, model, tmp2);
  EXPECT_EQ (_run_and_count (str_pipeline), 3);
  g_free (str_pipeline);

  EXPECT_TRUE (g_file_get_contents (tmp1, &in_data, &in_len, NULL));
  EXPECT_TRUE (g_file_get_contents (tmp2, &out_data, &out_len, NULL));
  EXPECT_EQ (in_len, 144U);
  EXPECT_EQ (out_len, in_len);
  if (in_data && out_data && in_len == out_len) {
    for (gsize i = 0; i < in_len; i++)
      EXPECT_EQ ((uint8_t) out_data[i], (uint8_t) (in_data[i] + 1));
  }

  g_free (in_data);
  g_free (out_data);
  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
}

/** @brief A filter without fixed dimensions gets its output dimension from the negotiated input */
TEST (cppFilterV1, setInputDimPipeline)
{
  filter_dims filter ("v1_follow", false, false, true);

  EXPECT_EQ (filter._register (), 0);
  _check_plus_one_pipeline ("v1_follow");
  EXPECT_EQ (filter._unregister (), 0);
}

/** @brief A filter with only a fixed input gets its output dimension from setInputDim */
TEST (cppFilterV1, fixedInputPipeline)
{
  filter_dims filter ("v1_fixed", true, false, true);

  EXPECT_EQ (filter._register (), 0);
  _check_plus_one_pipeline ("v1_fixed");
  EXPECT_EQ (filter._unregister (), 0);
}

/** @brief A filter with only a fixed output uses it when its setInputDim fails */
TEST (cppFilterV1, fixedOutputPipeline)
{
  filter_dims filter ("v1_fixed_out", false, true, false);

  EXPECT_EQ (filter._register (), 0);
  _check_plus_one_pipeline ("v1_fixed_out");
  EXPECT_EQ (filter._unregister (), 0);
}

/** @brief Outcome of the stream and single-shot cases for one filter_dims */
typedef struct {
  gint static4; /**< buffers out of a static 4x4 stream, -1 if refused */
  gint static16; /**< buffers out of a static 16x16 stream, -1 if refused */
  gint flexible; /**< buffers out of a flexible 4x4 stream, -1 if refused */
  gint flexible_input; /**< the same with the input property set, -1 if refused */
  gint dynamic; /**< the same with invoke-dynamic and no input property, -1 if refused */
  gint dynamic_static; /**< buffers out of a static 4x4 stream with invoke-dynamic, -1 if refused */
  guint single4; /**< output width after a single-shot 4x4 input, 0 if refused */
  guint single16; /**< output width after a single-shot 16x16 input, 0 if refused */
} dims_outcome;

/** @brief Run a pipeline whose fakesink is named sink, then count its buffers or return -1 on an error */
static gint
_run_or_refuse (const gchar *str_pipeline)
{
  gint received = 0, result = -1;
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);

  EXPECT_NE (pipeline, nullptr);
  if (!pipeline)
    return -2;

  GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  EXPECT_NE (sink, nullptr);
  if (sink) {
    g_signal_connect (sink, "handoff", G_CALLBACK (_handoff_count), &received);
    gst_object_unref (sink);
  }

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *msg = gst_bus_timed_pop_filtered (bus, 5 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  EXPECT_NE (msg, nullptr);
  if (msg) {
    if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_EOS)
      result = g_atomic_int_get (&received);
    gst_message_unref (msg);
  }
  gst_object_unref (bus);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (pipeline);

  return result;
}

/** @brief Open model with the single-shot filter, set a 3:width:width:1 input and return the output width, 0 if refused */
static guint
_single_set_input (const gchar *model, guint width)
{
  GTensorFilterSingle *single
      = (GTensorFilterSingle *) g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
  GTensorFilterSingleClass *klass
      = (GTensorFilterSingleClass *) g_type_class_ref (G_TYPE_TENSOR_FILTER_SINGLE);
  GstTensorsInfo in_info, out_info;
  guint out_width = 0;

  g_object_set (G_OBJECT (single), "framework", "cpp", "model", model, NULL);
  EXPECT_TRUE (klass->start (single));
  _fill_rgb_info (&in_info, width);
  gst_tensors_info_init (&out_info);
  if (klass->set_input_info (single, &in_info, &out_info) == 0)
    out_width = out_info.info[0].dimension[1];
  EXPECT_TRUE (klass->stop (single));

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  g_object_unref (single);
  g_type_class_unref (klass);
  return out_width;
}

/** @brief Run the stream and single-shot cases on a filter_dims and compare them with what v0 gave */
static void
_check_dims (const gchar *model, const dims_outcome *expected, gboolean refusals)
{
  const gchar *head = "videotestsrc num-buffers=3 ! video/x-raw,width=%u,height=%u,format=RGB ! tensor_converter ! ";
  const gchar *tail = " ! fakesink name=sink signal-handoffs=true sync=false async=false";
  const struct {
    guint width;
    const gchar *middle;
    gint expected;
  } streams[] = {
    { 4, "tensor_filter framework=cpp model=%s", expected->static4 },
    { 16, "tensor_filter framework=cpp model=%s", expected->static16 },
    { 4, "other/tensors,format=flexible ! tensor_filter framework=cpp model=%s",
        expected->flexible },
    { 4, "other/tensors,format=flexible ! tensor_filter framework=cpp model=%s input=3:4:4:1 inputtype=uint8",
        expected->flexible_input },
    { 4, "other/tensors,format=flexible ! tensor_filter framework=cpp model=%s invoke-dynamic=true ! other/tensors,format=flexible",
        expected->dynamic },
    { 4, "tensor_filter framework=cpp model=%s invoke-dynamic=true ! other/tensors,format=flexible",
        expected->dynamic_static },
  };

  for (const auto &stream : streams) {
    if ((stream.expected < 0) != refusals)
      continue;
    gchar *h = g_strdup_printf (head, stream.width, stream.width);
    gchar *m = g_strdup_printf (stream.middle, model);
    gchar *str_pipeline = g_strconcat (h, m, tail, NULL);
    EXPECT_EQ (_run_or_refuse (str_pipeline), stream.expected) << str_pipeline;
    g_free (h);
    g_free (m);
    g_free (str_pipeline);
  }

  if ((expected->single4 == 0) == refusals) {
    EXPECT_EQ (_single_set_input (model, 4), expected->single4);
  }
  if ((expected->single16 == 0) == refusals) {
    EXPECT_EQ (_single_set_input (model, 16), expected->single16);
  }
}

/** @brief Run the stream and single-shot cases on each kind of filter_dims, either the accepted or the refused ones */
static void
_check_dims_all (const gchar *prefix, gboolean refusals)
{
  const struct {
    const gchar *kind;
    bool fixed_in, fixed_out, set_ok;
    dims_outcome expected;
  } kinds[] = {
    { "both", true, true, true, { 3, -1, 3, 3, 3, 3, 4, 16 } },
    { "in", true, false, true, { 3, -1, 3, 3, 3, 3, 4, 16 } },
    { "in_noset", true, false, false, { -1, -1, -1, -1, 3, 3, 0, 0 } },
    { "out", false, true, false, { 3, 3, -1, 3, -1, -1, 4, 4 } },
    { "none", false, false, true, { 3, 3, -1, 3, -1, -1, 4, 16 } },
  };

  for (const auto &k : kinds) {
    gchar *model = g_strdup_printf ("%s_%s", prefix, k.kind);
    filter_dims filter (model, k.fixed_in, k.fixed_out, k.set_ok);

    EXPECT_EQ (filter._register (), 0);
    _check_dims (model, &k.expected, refusals);
    EXPECT_EQ (filter._unregister (), 0);
    g_free (model);
  }
}

/** @brief The streams and single-shot inputs that v0 accepted are accepted for each kind of filter_dims */
TEST (cppFilterV1, dimsAccepted)
{
  _check_dims_all ("v1_d", FALSE);
}

/** @brief The streams and single-shot inputs that v0 refused are refused for each kind of filter_dims */
TEST (cppFilterV1, dimsRefused_n)
{
  _check_dims_all ("v1_r", TRUE);
}

/** @brief is-updatable stays off: the cpp sub-plugin cannot reload a model */
TEST (cppFilterV1, isUpdatableRefused_n)
{
  gboolean updatable = TRUE;
  GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);

  ASSERT_NE (filter, nullptr);
  g_object_set (filter, "framework", "cpp", NULL);
  g_object_set (filter, "is-updatable", TRUE, NULL);
  g_object_get (filter, "is-updatable", &updatable, NULL);
  EXPECT_FALSE (updatable);

  gst_object_unref (filter);
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
