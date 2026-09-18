/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file	unittest_filter_common.cc
 * @date	18 September 2026
 * @brief	Unit test for the shared model table and the model property of tensor_filter
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <errno.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <unistd.h>

#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>

#include "../gst/nnstreamer/tensor_filter/tensor_filter_common.h"

#define TEST_SHARED_KEY "unittest_filter_common_key"

static guint num_freed;
static guint num_replaced;
static void *last_freed;
static void *last_replaced_instance;
static void *last_replaced_interpreter;

/**
 * @brief Record a destroyed interpreter instead of releasing it.
 */
static void
_count_free (void *interpreter)
{
  num_freed++;
  last_freed = interpreter;
}

/**
 * @brief Record an instance the shared model table asked to take a new interpreter.
 */
static void
_count_replace (void *instance, void *interpreter)
{
  num_replaced++;
  last_replaced_instance = instance;
  last_replaced_interpreter = interpreter;
}

/**
 * @brief Test fixture driving the shared model table through two filter instances.
 */
class testFilterSharedModel : public ::testing::Test
{
  protected:
  GstTensorFilterPrivate priv1;
  GstTensorFilterPrivate priv2;

  /** @brief initialize two filter instances sharing the same key */
  void SetUp () override
  {
    num_freed = num_replaced = 0;
    last_freed = last_replaced_instance = last_replaced_interpreter = nullptr;

    gst_tensor_filter_common_init_property (&priv1);
    gst_tensor_filter_common_init_property (&priv2);
    setSharedKey (&priv1, TEST_SHARED_KEY);
    setSharedKey (&priv2, TEST_SHARED_KEY);
  }

  /** @brief release the instances that the test itself has not released */
  void TearDown () override
  {
    gst_tensor_filter_common_free_property (&priv1);
    gst_tensor_filter_common_free_property (&priv2);
  }

  /**
   * @brief Set the shared model key of a filter instance.
   * @param priv the filter instance
   * @param key the value of the shared-tensor-filter-key property
   */
  void setSharedKey (GstTensorFilterPrivate *priv, const gchar *key)
  {
    GValue value = G_VALUE_INIT;

    g_value_init (&value, G_TYPE_STRING);
    g_value_set_string (&value, key);
    EXPECT_TRUE (gst_tensor_filter_common_set_property (
        priv, PROP_SHARED_TENSOR_FILTER_KEY, &value, NULL));
    g_value_unset (&value);
  }
};

/**
 * @brief A filter that goes away leaves the shared model of the other filter alone.
 * @details The table is process-global. Releasing the properties of one
 *          instance used to destroy it, which left the survivor unable to find,
 *          replace or release its own interpreter.
 */
TEST_F (testFilterSharedModel, keepTableForOtherInstance)
{
  int interpreter = 0;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  /* The first instance is closed and then finalized. */
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 0U);
  gst_tensor_filter_common_free_property (&priv1);
  gst_tensor_filter_common_init_property (&priv1);

  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
  EXPECT_TRUE (last_freed == &interpreter);
}

/**
 * @brief A model reload of the surviving filter still reaches its referred list.
 */
TEST_F (testFilterSharedModel, replaceAfterOtherInstanceFreed)
{
  int interpreter = 0;
  int reloaded = 0;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  gst_tensor_filter_common_free_property (&priv1);
  gst_tensor_filter_common_init_property (&priv1);

  nnstreamer_filter_shared_model_replace (
      &priv2, TEST_SHARED_KEY, &reloaded, _count_replace, _count_free);

  EXPECT_EQ (num_replaced, 1U);
  EXPECT_TRUE (last_replaced_instance == &priv2);
  EXPECT_TRUE (last_replaced_interpreter == &reloaded);
  EXPECT_EQ (num_freed, 1U);
  EXPECT_TRUE (last_freed == &interpreter);
  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &reloaded);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 2U);
}

/**
 * @brief Replacing the key of one filter does not release the table of the others.
 */
TEST_F (testFilterSharedModel, keepTableOnKeyChange)
{
  int interpreter = 0;

  setSharedKey (&priv1, TEST_SHARED_KEY "_other");
  gst_tensor_filter_common_free_property (&priv1);
  gst_tensor_filter_common_init_property (&priv1);

  EXPECT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv2, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
}

/**
 * @brief The shared model table is gone once the last filter holding a key is freed.
 */
TEST_F (testFilterSharedModel, getAfterLastKeyFreed_n)
{
  gst_tensor_filter_common_free_property (&priv1);
  gst_tensor_filter_common_init_property (&priv1);
  gst_tensor_filter_common_free_property (&priv2);
  gst_tensor_filter_common_init_property (&priv2);

  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == NULL);
  EXPECT_FALSE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 0U);
}

/**
 * @brief An entry left in the table keeps it alive, so its owner can still release it.
 */
TEST_F (testFilterSharedModel, removeAfterKeyFreed)
{
  int interpreter = 0;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);

  gst_tensor_filter_common_free_property (&priv2);
  gst_tensor_filter_common_init_property (&priv2);
  gst_tensor_filter_common_free_property (&priv1);
  gst_tensor_filter_common_init_property (&priv1);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
}

/**
 * @brief The same key cannot be inserted twice.
 */
TEST_F (testFilterSharedModel, insertTwice_n)
{
  int interpreter = 0;
  int other = 0;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  EXPECT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv2, (char *) TEST_SHARED_KEY, &other)
               == NULL);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
}

/**
 * @brief Inserting with no instance, no key or no interpreter is refused.
 */
TEST_F (testFilterSharedModel, insertInvalidParam_n)
{
  int interpreter = 0;

  EXPECT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   NULL, (char *) TEST_SHARED_KEY, &interpreter)
               == NULL);
  EXPECT_TRUE (nnstreamer_filter_shared_model_insert_and_get (&priv1, NULL, &interpreter)
               == NULL);
  EXPECT_TRUE (nnstreamer_filter_shared_model_insert_and_get (&priv1, (char *) TEST_SHARED_KEY, NULL)
               == NULL);
}

/**
 * @brief Looking up, removing or replacing an unknown key changes nothing.
 */
TEST_F (testFilterSharedModel, unknownKey_n)
{
  int reloaded = 0;

  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv1, TEST_SHARED_KEY "_unknown") == NULL);
  EXPECT_FALSE (nnstreamer_filter_shared_model_remove (
      &priv1, TEST_SHARED_KEY "_unknown", _count_free));
  nnstreamer_filter_shared_model_replace (&priv1, TEST_SHARED_KEY "_unknown",
      &reloaded, _count_replace, _count_free);

  EXPECT_EQ (num_replaced, 0U);
  EXPECT_EQ (num_freed, 0U);
}

/**
 * @brief Replacing with no key changes nothing.
 */
TEST_F (testFilterSharedModel, replaceNullKey_n)
{
  int reloaded = 0;

  nnstreamer_filter_shared_model_replace (&priv1, NULL, &reloaded, _count_replace, _count_free);

  EXPECT_EQ (num_replaced, 0U);
  EXPECT_EQ (num_freed, 0U);
}

/**
 * @brief C++ sub-plugin that holds open() until the test releases it.
 * @details It keeps the model path pointer it was given on entry and reads it
 *          back after the test has replaced the model property, so that a
 *          released model file list shows up as a changed string.
 */
class model_mock_subplugin : public nnstreamer::tensor_filter_subplugin
{
  public:
  static const char *mock_name;
  static model_mock_subplugin *registered;
  static GMutex lock;
  static GCond cond;
  static gboolean hold_open;
  static gboolean inside_open;
  static gboolean released;
  static gboolean readback_matched;
  static gchar *seen_model;
  static const gchar *seen_pointer;
  static int event_ret;
  static guint num_reloads;

  /** @brief mandatory method */
  tensor_filter_subplugin &getEmptyInstance () override
  {
    return *(new model_mock_subplugin ());
  }

  /** @brief hold the call while the test replaces the model property */
  void configure_instance (const GstTensorFilterProperties *prop) override
  {
    g_mutex_lock (&lock);
    g_free (seen_model);
    seen_pointer = prop->model_files[0];
    seen_model = g_strdup (seen_pointer);
    inside_open = TRUE;
    g_cond_broadcast (&cond);

    while (hold_open && !released)
      g_cond_wait (&cond, &lock);

    readback_matched = (g_strcmp0 (seen_pointer, seen_model) == 0);
    g_mutex_unlock (&lock);
  }

  /** @brief mandatory method */
  void invoke (const GstTensorMemory *input, GstTensorMemory *output) override
  {
    UNUSED (input);
    UNUSED (output);
  }

  /** @brief mandatory method */
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info) override
  {
    info.name = mock_name;
    info.allow_in_place = FALSE;
    info.allocate_in_invoke = FALSE;
    info.run_without_model = FALSE;
    info.verify_model_path = TRUE;
    info.hw_list = nullptr;
    info.num_hw = 0;
  }

  /** @brief mandatory method */
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info) override
  {
    UNUSED (ops);
    UNUSED (in_info);
    UNUSED (out_info);
    return -ENOENT;
  }

  /** @brief count the model reloads and report what the test asks for */
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data) override
  {
    UNUSED (data);
    if (ops == RELOAD_MODEL)
      num_reloads++;

    return event_ret;
  }

  /** @brief register this mock subplugin */
  static void init ()
  {
    registered = register_subplugin<model_mock_subplugin> ();
  }

  /** @brief unregister this mock subplugin */
  static void fini ()
  {
    unregister_subplugin<model_mock_subplugin> (registered);
    registered = nullptr;
  }
};

const char *model_mock_subplugin::mock_name = "model_mock_subplugin";
model_mock_subplugin *model_mock_subplugin::registered = nullptr;
GMutex model_mock_subplugin::lock;
GCond model_mock_subplugin::cond;
gboolean model_mock_subplugin::hold_open = FALSE;
gboolean model_mock_subplugin::inside_open = FALSE;
gboolean model_mock_subplugin::released = FALSE;
gboolean model_mock_subplugin::readback_matched = FALSE;
gchar *model_mock_subplugin::seen_model = nullptr;
const gchar *model_mock_subplugin::seen_pointer = nullptr;
int model_mock_subplugin::event_ret = 0;
guint model_mock_subplugin::num_reloads = 0;

/**
 * @brief Test fixture replacing the model property of a filter under a sub-plugin call.
 */
class testFilterModelProperty : public ::testing::Test
{
  protected:
  GstTensorFilterPrivate priv;
  gchar *model_a;
  gchar *model_b;

  /** @brief register the mock and give the filter a model file */
  void SetUp () override
  {
    gint fd;

    model_a = model_b = nullptr;
    fd = g_file_open_tmp ("nnsb14aXXXXXX", &model_a, NULL);
    ASSERT_GE (fd, 0);
    close (fd);
    fd = g_file_open_tmp ("nnsb14bXXXXXX", &model_b, NULL);
    ASSERT_GE (fd, 0);
    close (fd);

    model_mock_subplugin::hold_open = FALSE;
    model_mock_subplugin::inside_open = FALSE;
    model_mock_subplugin::released = FALSE;
    model_mock_subplugin::readback_matched = FALSE;
    model_mock_subplugin::seen_pointer = nullptr;
    model_mock_subplugin::event_ret = 0;
    model_mock_subplugin::num_reloads = 0;
    model_mock_subplugin::init ();

    gst_tensor_filter_common_init_property (&priv);
    g_free ((gpointer) priv.prop.fwname);
    priv.prop.fwname = g_strdup (model_mock_subplugin::mock_name);
    priv.fw = nnstreamer_filter_find (model_mock_subplugin::mock_name);
    ASSERT_TRUE (priv.fw != NULL);
  }

  /** @brief close the filter and drop the temporary model files */
  void TearDown () override
  {
    gst_tensor_filter_common_close_fw (&priv);
    gst_tensor_filter_common_free_property (&priv);
    model_mock_subplugin::fini ();

    g_free (model_mock_subplugin::seen_model);
    model_mock_subplugin::seen_model = nullptr;

    if (model_a) {
      g_remove (model_a);
      g_free (model_a);
    }
    if (model_b) {
      g_remove (model_b);
      g_free (model_b);
    }
  }

  /**
   * @brief Set the model property of the filter.
   * @param models the value of the model property
   */
  void setModel (const gchar *models)
  {
    GValue value = G_VALUE_INIT;

    g_value_init (&value, G_TYPE_STRING);
    g_value_set_string (&value, models);
    EXPECT_TRUE (gst_tensor_filter_common_set_property (&priv, PROP_MODEL, &value, NULL));
    g_value_unset (&value);
  }

  /**
   * @brief Read the model property of the filter back.
   * @return the value of the model property. Free it with g_free().
   */
  gchar *getModel ()
  {
    GValue value = G_VALUE_INIT;
    gchar *models;

    g_value_init (&value, G_TYPE_STRING);
    EXPECT_TRUE (gst_tensor_filter_common_get_property (&priv, PROP_MODEL, &value, NULL));
    models = g_value_dup_string (&value);
    g_value_unset (&value);

    return models;
  }

  /**
   * @brief Open the filter and let it reload models afterwards.
   */
  void openUpdatable ()
  {
    GValue value = G_VALUE_INIT;

    ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));

    g_value_init (&value, G_TYPE_BOOLEAN);
    g_value_set_boolean (&value, TRUE);
    EXPECT_TRUE (gst_tensor_filter_common_set_property (
        &priv, PROP_IS_UPDATABLE, &value, NULL));
    g_value_unset (&value);
    ASSERT_TRUE (priv.is_updatable);
  }
};

/**
 * @brief Open the filter of the given fixture. Runs on a worker thread.
 */
static gpointer
_open_fw_thread (gpointer data)
{
  GstTensorFilterPrivate *priv = (GstTensorFilterPrivate *) data;

  return GINT_TO_POINTER (gst_tensor_filter_common_open_fw (priv));
}

#define B14_CHURN_ROUNDS (100)

/**
 * @brief Replace the model of the given fixture over and over. Runs on a worker thread.
 */
static gpointer
_set_model_thread (gpointer data)
{
  gchar **models = (gchar **) data;
  GstTensorFilterPrivate *priv = (GstTensorFilterPrivate *) models[0];
  GValue value = G_VALUE_INIT;
  guint i;

  g_value_init (&value, G_TYPE_STRING);
  for (i = 0; i < B14_CHURN_ROUNDS; i++) {
    g_value_set_string (&value, models[1 + (i & 1)]);
    gst_tensor_filter_common_set_property (priv, PROP_MODEL, &value, NULL);
  }
  g_value_unset (&value);

  return NULL;
}

/**
 * @brief The model property can be replaced while a sub-plugin reads it.
 * @details The sub-plugin is given prop->model_files and the model property is
 *          not serialized with the streaming thread, so a model file list
 *          released on replacement was read back by the call still holding it.
 */
TEST_F (testFilterModelProperty, replaceModelWhileOpening)
{
  GThread *opener;
  gint64 deadline;
  gboolean in_time = TRUE;

  setModel (model_a);
  model_mock_subplugin::hold_open = TRUE;

  opener = g_thread_new ("b14-open", _open_fw_thread, &priv);
  deadline = g_get_monotonic_time () + 10 * G_TIME_SPAN_SECOND;

  g_mutex_lock (&model_mock_subplugin::lock);
  while (!model_mock_subplugin::inside_open && in_time)
    in_time = g_cond_wait_until (
        &model_mock_subplugin::cond, &model_mock_subplugin::lock, deadline);
  g_mutex_unlock (&model_mock_subplugin::lock);
  EXPECT_TRUE (in_time) << "the sub-plugin did not reach the hold point";

  setModel (model_b);

  g_mutex_lock (&model_mock_subplugin::lock);
  model_mock_subplugin::released = TRUE;
  g_cond_broadcast (&model_mock_subplugin::cond);
  g_mutex_unlock (&model_mock_subplugin::lock);

  EXPECT_TRUE (GPOINTER_TO_INT (g_thread_join (opener)));
  EXPECT_TRUE (model_mock_subplugin::readback_matched);
  EXPECT_STREQ (model_mock_subplugin::seen_model, model_a);
}

/**
 * @brief The model file list is not read while it is being replaced.
 * @details Opening the filter walks the list to verify every path, and reading
 *          the model property joins it into a string. Both index the list by
 *          prop->num_models, so a list replaced under them was read past its
 *          end or after it was released.
 */
TEST_F (testFilterModelProperty, replaceModelWhileReading)
{
  g_autofree gchar *both = g_strdup_printf ("%s,%s", model_a, model_b);
  gpointer models[3] = { &priv, both, model_a };
  GThread *setter;
  guint i;

  setModel (model_a);
  setter = g_thread_new ("b14-set", _set_model_thread, models);

  for (i = 0; i < B14_CHURN_ROUNDS; i++) {
    g_autofree gchar *read = getModel ();

    ASSERT_TRUE (g_strcmp0 (read, both) == 0 || g_strcmp0 (read, model_a) == 0)
        << "read '" << read << "'";
    gst_tensor_filter_common_open_fw (&priv);
    gst_tensor_filter_common_unload_fw (&priv, FALSE);
  }

  g_thread_join (setter);
}

/**
 * @brief A filter with no model file is not opened.
 */
TEST_F (testFilterModelProperty, openWithoutModel_n)
{
  EXPECT_FALSE (gst_tensor_filter_common_open_fw (&priv));
  EXPECT_FALSE (priv.prop.fw_opened);
}

/**
 * @brief An empty model property leaves the filter with no model file.
 */
TEST_F (testFilterModelProperty, openWithEmptyModel_n)
{
  g_autofree gchar *read = NULL;

  setModel (model_a);
  setModel ("");

  read = getModel ();
  EXPECT_STREQ (read, "");
  EXPECT_EQ (priv.prop.num_models, 0);
  EXPECT_FALSE (gst_tensor_filter_common_open_fw (&priv));
}

/**
 * @brief A model file that does not exist is refused.
 */
TEST_F (testFilterModelProperty, openMissingModel_n)
{
  g_autofree gchar *missing = g_strdup_printf ("%s.missing", model_a);

  setModel (missing);
  EXPECT_FALSE (gst_tensor_filter_common_open_fw (&priv));
  EXPECT_FALSE (priv.prop.fw_opened);
}

/**
 * @brief Reloading the model of an opened filter installs the new model file.
 */
TEST_F (testFilterModelProperty, reloadModelWhileOpened)
{
  g_autofree gchar *read = NULL;

  setModel (model_a);
  openUpdatable ();

  setModel (model_b);

  EXPECT_EQ (model_mock_subplugin::num_reloads, 2U);
  read = getModel ();
  EXPECT_STREQ (read, model_b);
}

/**
 * @brief A reload the sub-plugin refuses restores the model file it was using.
 */
TEST_F (testFilterModelProperty, reloadModelRefused_n)
{
  g_autofree gchar *read = NULL;

  setModel (model_a);
  openUpdatable ();

  model_mock_subplugin::event_ret = -EINVAL;
  setModel (model_b);

  read = getModel ();
  EXPECT_STREQ (read, model_a);
  EXPECT_EQ (priv.prop.num_models, 1);
}

/**
 * @brief Every property of a tensor_filter element can be read back.
 * @details The model property is joined from the model file list, so the getter
 *          walks it the same way the sub-plugins do. Reading the neighbours as
 *          well pins that the change did not disturb them.
 */
TEST (testFilterProperties, getEveryProperty)
{
  GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);
  GParamSpec **pspecs;
  guint i, num_props;
  g_autofree gchar *models = NULL;

  ASSERT_TRUE (filter != NULL);

  g_object_set (filter, "framework", "custom-easy", "model", "a.so,b.so",
      "input", "3:4:4:1", "inputtype", "uint8", "output", "3:4:4:1",
      "outputtype", "uint8", "custom", "key=value", "accelerator", "true:cpu",
      "input-combination", "i0", "output-combination", "i0,o0", "silent", FALSE,
      "latency", 1, "throughput", 1, "suspend", 1, NULL);

  pspecs = g_object_class_list_properties (G_OBJECT_GET_CLASS (filter), &num_props);
  for (i = 0; i < num_props; i++) {
    GValue value = G_VALUE_INIT;

    g_value_init (&value, pspecs[i]->value_type);
    g_object_get_property (G_OBJECT (filter), pspecs[i]->name, &value);
    g_value_unset (&value);
  }
  g_free (pspecs);

  g_object_get (filter, "model", &models, NULL);
  EXPECT_STREQ (models, "a.so,b.so");

  gst_object_unref (filter);
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
