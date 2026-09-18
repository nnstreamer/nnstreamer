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
#include <glib.h>
#include <gst/gst.h>

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
  EXPECT_TRUE (nnstreamer_filter_shared_model_insert_and_get (&priv1, NULL, &interpreter) == NULL);
  EXPECT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, NULL)
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
  nnstreamer_filter_shared_model_replace (
      &priv1, TEST_SHARED_KEY "_unknown", &reloaded, _count_replace, _count_free);

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
