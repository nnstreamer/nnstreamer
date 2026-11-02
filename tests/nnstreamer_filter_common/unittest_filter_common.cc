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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <errno.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <mutex>
#include <string.h>
#include <thread>
#include <unistd.h>

#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>

#include "../gst/nnstreamer/tensor_filter/tensor_filter_common.h"
#include "../gst/nnstreamer/tensor_filter/tensor_filter_single.h"

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

#define MAX_REPLACE_RECORDS (8)

/**
 * @brief One call the shared model table made to a replace callback.
 */
typedef struct {
  void *instance;
  void *interpreter;
} replace_record_s;

static replace_record_s records[MAX_REPLACE_RECORDS];
static guint num_records;
static gint refuse_nth;
static gint refuse_rollback_nth;
static guint num_printerr;
static gchar *last_printerr;

/**
 * @brief Record a replace request and refuse the calls the test asked to refuse.
 */
static int
_checked_replace (void *instance, void *interpreter)
{
  int ret = 0;

  if (num_records < MAX_REPLACE_RECORDS) {
    records[num_records].instance = instance;
    records[num_records].interpreter = interpreter;
  }
  if ((gint) num_records == refuse_nth || (gint) num_records == refuse_rollback_nth)
    ret = -EINVAL;
  num_records++;

  return ret;
}

/**
 * @brief Keep what the deprecated helper wrote to stderr instead of printing it.
 */
static void
_record_printerr (const gchar *message)
{
  num_printerr++;
  g_free (last_printerr);
  last_printerr = g_strdup (message);
}

/**
 * @brief Test fixture driving the shared model table through two filter instances.
 */
class testFilterSharedModel : public ::testing::Test
{
  protected:
  GstTensorFilterPrivate priv1;
  GstTensorFilterPrivate priv2;
  GstTensorFilterPrivate priv3;
  gboolean priv3_used;

  /** @brief initialize two filter instances sharing the same key */
  void SetUp () override
  {
    num_freed = num_replaced = num_records = num_printerr = 0;
    last_freed = last_replaced_instance = last_replaced_interpreter = nullptr;
    refuse_nth = refuse_rollback_nth = -1;
    memset (records, 0, sizeof (records));
    g_free (last_printerr);
    last_printerr = nullptr;

    gst_tensor_filter_common_init_property (&priv1);
    gst_tensor_filter_common_init_property (&priv2);
    setSharedKey (&priv1, TEST_SHARED_KEY);
    setSharedKey (&priv2, TEST_SHARED_KEY);
    priv3_used = FALSE;
  }

  /**
   * @brief Release the instances that the test itself has not released.
   * @details A case that ends early, on a failed assertion, leaves its
   *          instances registered under the key; dropping them here keeps the
   *          process-wide table from carrying them into the cases that follow.
   *          A case that passed has already dropped its own, and asking again
   *          would only log that the key is gone.
   */
  void TearDown () override
  {
    if (HasFailure ()) {
      nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free);
      nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free);
      if (priv3_used)
        nnstreamer_filter_shared_model_remove (&priv3, TEST_SHARED_KEY, _count_free);
    }

    gst_tensor_filter_common_free_property (&priv1);
    gst_tensor_filter_common_free_property (&priv2);
    if (priv3_used)
      gst_tensor_filter_common_free_property (&priv3);
  }

  /**
   * @brief Add a third instance sharing the key.
   * @details TearDown() releases its properties, and drops its registration as
   *          well when the case ended on a failure, so that an assertion
   *          failing mid-case cannot leave it in the process-wide table.
   */
  void useThirdInstance ()
  {
    gst_tensor_filter_common_init_property (&priv3);
    priv3_used = TRUE;
    setSharedKey (&priv3, TEST_SHARED_KEY);
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

  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv2,
                 TEST_SHARED_KEY, &reloaded, _checked_replace, _count_free),
      0);

  EXPECT_EQ (num_records, 1U);
  EXPECT_TRUE (records[0].instance == &priv2);
  EXPECT_TRUE (records[0].interpreter == &reloaded);
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
  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv1,
                 TEST_SHARED_KEY "_unknown", &reloaded, _checked_replace, _count_free),
      -ENOENT);

  EXPECT_EQ (num_records, 0U);
  EXPECT_EQ (num_freed, 0U);
}

/**
 * @brief Replacing with no key, no interpreter or no callback changes nothing.
 */
TEST_F (testFilterSharedModel, replaceInvalidParam_n)
{
  int reloaded = 0;

  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (
                 &priv1, NULL, &reloaded, _checked_replace, _count_free),
      -EINVAL);
  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (
                 &priv1, TEST_SHARED_KEY, NULL, _checked_replace, _count_free),
      -EINVAL);
  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (
                 &priv1, TEST_SHARED_KEY, &reloaded, NULL, _count_free),
      -EINVAL);
  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (
                 &priv1, TEST_SHARED_KEY, &reloaded, _checked_replace, NULL),
      -EINVAL);

  EXPECT_EQ (num_records, 0U);
  EXPECT_EQ (num_freed, 0U);
}

/**
 * @brief Replacing without a shared model table changes nothing.
 */
TEST_F (testFilterSharedModel, replaceWithoutTable_n)
{
  int reloaded = 0;

  gst_tensor_filter_common_free_property (&priv1);
  gst_tensor_filter_common_init_property (&priv1);
  gst_tensor_filter_common_free_property (&priv2);
  gst_tensor_filter_common_init_property (&priv2);

  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv1,
                 TEST_SHARED_KEY, &reloaded, _checked_replace, _count_free),
      -EPERM);

  EXPECT_EQ (num_records, 0U);
  EXPECT_EQ (num_freed, 0U);
}

/**
 * @brief Every instance takes the new interpreter, so the old one is released.
 */
TEST_F (testFilterSharedModel, replaceEveryInstance)
{
  int interpreter = 0;
  int reloaded = 0;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv1,
                 TEST_SHARED_KEY, &reloaded, _checked_replace, _count_free),
      0);

  EXPECT_EQ (num_records, 2U);
  EXPECT_TRUE (records[0].instance == &priv1);
  EXPECT_TRUE (records[0].interpreter == &reloaded);
  EXPECT_TRUE (records[1].instance == &priv2);
  EXPECT_TRUE (records[1].interpreter == &reloaded);
  EXPECT_EQ (num_freed, 1U);
  EXPECT_TRUE (last_freed == &interpreter);
  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &reloaded);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 2U);
}

/**
 * @brief The first instance refuses, so nothing is replaced and nothing is released.
 */
TEST_F (testFilterSharedModel, replaceRefusedByFirst_n)
{
  int interpreter = 0;
  int reloaded = 0;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  refuse_nth = 0;
  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv1,
                 TEST_SHARED_KEY, &reloaded, _checked_replace, _count_free),
      -EINVAL);

  EXPECT_EQ (num_records, 1U);
  EXPECT_EQ (num_freed, 0U);
  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
  EXPECT_TRUE (last_freed == &interpreter);
}

/**
 * @brief A later instance refuses, so the instances that took the new interpreter get the old one back.
 * @details This is the use-after-free of item B16 of issue 4920: the interpreter
 *          the refusing instance still points at must stay alive.
 */
TEST_F (testFilterSharedModel, replaceRefusedByLast_n)
{
  int interpreter = 0;
  int reloaded = 0;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  refuse_nth = 1;
  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv1,
                 TEST_SHARED_KEY, &reloaded, _checked_replace, _count_free),
      -EINVAL);

  EXPECT_EQ (num_records, 3U);
  EXPECT_TRUE (records[2].instance == &priv1);
  EXPECT_TRUE (records[2].interpreter == &interpreter);
  EXPECT_EQ (num_freed, 0U);
  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
  EXPECT_TRUE (last_freed == &interpreter);
}

/**
 * @brief An instance refuses the old interpreter back, so both interpreters stay in use.
 * @details Neither interpreter may be released while an instance points at it,
 *          so the caller is told to keep both.
 */
TEST_F (testFilterSharedModel, replaceRollbackRefused_n)
{
  int interpreter = 0;
  int reloaded = 0;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  refuse_nth = 1;
  refuse_rollback_nth = 2;
  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv1,
                 TEST_SHARED_KEY, &reloaded, _checked_replace, _count_free),
      -EBUSY);

  EXPECT_EQ (num_records, 3U);
  EXPECT_EQ (num_freed, 0U);
  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
}

/**
 * @brief Handing over the interpreter that is already shared is refused.
 * @details Taking it would release the interpreter and then publish the
 *          released pointer as the shared one. The code differs from the one a
 *          refusal gives, because this interpreter belongs to the table and the
 *          caller must not destroy it.
 */
TEST_F (testFilterSharedModel, replaceSameInterpreter_n)
{
  int interpreter = 0;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);

  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv1,
                 TEST_SHARED_KEY, &interpreter, _checked_replace, _count_free),
      -EEXIST);

  EXPECT_EQ (num_records, 0U);
  EXPECT_EQ (num_freed, 0U);
  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv1, TEST_SHARED_KEY) == &interpreter);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
}

/**
 * @brief Every instance that had taken the new interpreter is given the old one back.
 * @details Three instances make the rollback more than one step, so that a
 *          rollback which stops at the first instance is distinguishable.
 */
TEST_F (testFilterSharedModel, replaceRollsBackEveryInstance_n)
{
  int interpreter = 0;
  int reloaded = 0;

  useThirdInstance ();

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv3, TEST_SHARED_KEY) == &interpreter);

  refuse_nth = 2;
  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv1,
                 TEST_SHARED_KEY, &reloaded, _checked_replace, _count_free),
      -EINVAL);

  EXPECT_EQ (num_records, 5U);
  EXPECT_TRUE (records[3].instance == &priv1);
  EXPECT_TRUE (records[3].interpreter == &interpreter);
  EXPECT_TRUE (records[4].instance == &priv2);
  EXPECT_TRUE (records[4].interpreter == &interpreter);
  EXPECT_EQ (num_freed, 0U);
  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv3, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
}

/**
 * @brief An instance refusing the old interpreter back does not stop the rollback.
 * @details The instances behind the refusing one must still be given the old
 *          interpreter, so that as few as possible are left on the new one.
 */
TEST_F (testFilterSharedModel, replaceRollbackContinuesAfterRefusal_n)
{
  int interpreter = 0;
  int reloaded = 0;

  useThirdInstance ();

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv3, TEST_SHARED_KEY) == &interpreter);

  refuse_nth = 2;
  refuse_rollback_nth = 3;
  EXPECT_EQ (nnstreamer_filter_shared_model_replace_checked (&priv1,
                 TEST_SHARED_KEY, &reloaded, _checked_replace, _count_free),
      -EBUSY);

  EXPECT_EQ (num_records, 5U);
  EXPECT_TRUE (records[4].instance == &priv2);
  EXPECT_TRUE (records[4].interpreter == &interpreter);
  EXPECT_EQ (num_freed, 0U);
  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv3, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 1U);
}

/**
 * @brief The deprecated helper still replaces, and warns once per process.
 * @details The warning has to reach stderr, because the log channel is dlog on
 *          Tizen. This is the only case that calls the deprecated helper, so the
 *          process-wide once-only warning can be counted here whatever order the
 *          cases run in; every branch of that helper is driven from here.
 */
TEST_F (testFilterSharedModel, deprecatedReplaceWarnsOnce)
{
  int interpreter = 0;
  int reloaded = 0;
  GPrintFunc previous;

  ASSERT_TRUE (nnstreamer_filter_shared_model_insert_and_get (
                   &priv1, (char *) TEST_SHARED_KEY, &interpreter)
               == &interpreter);
  ASSERT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &interpreter);

  previous = g_set_printerr_handler (_record_printerr);

  G_GNUC_BEGIN_IGNORE_DEPRECATIONS
  nnstreamer_filter_shared_model_replace (&priv1, NULL, &reloaded, _count_replace, _count_free);
  EXPECT_EQ (num_printerr, 1U);
  nnstreamer_filter_shared_model_replace (&priv1, TEST_SHARED_KEY "_unknown",
      &reloaded, _count_replace, _count_free);
  EXPECT_EQ (num_printerr, 1U);
  nnstreamer_filter_shared_model_replace (
      &priv1, TEST_SHARED_KEY, &reloaded, _count_replace, _count_free);
  EXPECT_EQ (num_printerr, 1U);
  G_GNUC_END_IGNORE_DEPRECATIONS

  g_set_printerr_handler (previous);

  ASSERT_TRUE (last_printerr != nullptr);
  EXPECT_TRUE (g_strstr_len (last_printerr, -1, "nnstreamer_filter_shared_model_replace_checked")
               != NULL);
  EXPECT_EQ (num_replaced, 2U);
  EXPECT_TRUE (last_replaced_instance == &priv2);
  EXPECT_TRUE (last_replaced_interpreter == &reloaded);
  EXPECT_EQ (num_freed, 1U);
  EXPECT_TRUE (last_freed == &interpreter);
  EXPECT_TRUE (nnstreamer_filter_shared_model_get (&priv2, TEST_SHARED_KEY) == &reloaded);

  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv1, TEST_SHARED_KEY, _count_free));
  EXPECT_TRUE (nnstreamer_filter_shared_model_remove (&priv2, TEST_SHARED_KEY, _count_free));
  EXPECT_EQ (num_freed, 2U);
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
      "input-combination", "0", "output-combination", "i0,o0", "silent", FALSE,
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
 * @brief The combination properties of a tensor_filter element keep a valid value and ignore blanks around its entries.
 */
TEST (testFilterProperties, combination)
{
  GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);
  g_autofree gchar *in_combi = NULL;
  g_autofree gchar *out_combi = NULL;

  ASSERT_TRUE (filter != NULL);

  g_object_set (filter, "input-combination", "0, 2", "output-combination", "i1, o0", NULL);
  g_object_get (filter, "input-combination", &in_combi, "output-combination",
      &out_combi, NULL);
  EXPECT_STREQ (in_combi, "0,2");
  EXPECT_STREQ (out_combi, "i1,o0");

  gst_object_unref (filter);
}

/**
 * @brief The combination properties of a tensor_filter element drop a value with an entry that is not a tensor index.
 */
TEST (testFilterProperties, combinationInvalidIndex_n)
{
  GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);
  g_autofree gchar *in_combi = NULL;
  g_autofree gchar *out_combi = NULL;

  ASSERT_TRUE (filter != NULL);

  g_object_set (filter, "input-combination", "0,2", "output-combination", "i1,o0", NULL);
  g_object_set (filter, "input-combination", "0,1x", "output-combination", "i1,o", NULL);
  g_object_get (filter, "input-combination", &in_combi, "output-combination",
      &out_combi, NULL);
  EXPECT_STREQ (in_combi, "");
  EXPECT_STREQ (out_combi, "");

  gst_object_unref (filter);
}

/**
 * @brief V1 C++ sub-plugin recording the accelerator it is opened with.
 */
class accl_mock_subplugin : public nnstreamer::tensor_filter_subplugin
{
  public:
  static const char *mock_name;
  static accl_mock_subplugin *registered;
  static const accl_hw hw_list[];
  static gchar *seen_accl_str;
  static gint seen_num_hw;
  static accl_hw seen_hw;
  static int event_ret;
  static accl_hw basic_hw;

  /** @brief mandatory method */
  tensor_filter_subplugin &getEmptyInstance () override
  {
    return *(new accl_mock_subplugin ());
  }

  /** @brief record the accelerator given on open */
  void configure_instance (const GstTensorFilterProperties *prop) override
  {
    g_free (seen_accl_str);
    seen_accl_str = g_strdup (prop->accl_str);
    seen_num_hw = prop->num_hw;
    seen_hw = (prop->num_hw > 0) ? prop->hw_list[0] : ACCL_NONE;
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
    info.run_without_model = TRUE;
    info.verify_model_path = FALSE;
    info.hw_list = hw_list;
    info.num_hw = 3;
    info.accl_auto = basic_hw;
    info.accl_default = basic_hw;
  }

  /** @brief mandatory method */
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info) override
  {
    UNUSED (ops);
    UNUSED (in_info);
    UNUSED (out_info);
    return -ENOENT;
  }

  /** @brief report what the test asks for */
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data) override
  {
    UNUSED (ops);
    UNUSED (data);
    return event_ret;
  }

  /** @brief register this mock subplugin */
  static void init ()
  {
    registered = register_subplugin<accl_mock_subplugin> ();
  }

  /** @brief unregister this mock subplugin */
  static void fini ()
  {
    unregister_subplugin<accl_mock_subplugin> (registered);
    registered = nullptr;
  }
};

const char *accl_mock_subplugin::mock_name = "accl_mock_subplugin";
accl_mock_subplugin *accl_mock_subplugin::registered = nullptr;
const accl_hw accl_mock_subplugin::hw_list[] = { ACCL_CPU, ACCL_GPU, ACCL_NPU_EDGE_TPU };
gchar *accl_mock_subplugin::seen_accl_str = nullptr;
gint accl_mock_subplugin::seen_num_hw = 0;
accl_hw accl_mock_subplugin::seen_hw = ACCL_NONE;
int accl_mock_subplugin::event_ret = 0;
accl_hw accl_mock_subplugin::basic_hw = ACCL_CPU;

/**
 * @brief V1 C++ sub-plugin supporting the CPU only.
 */
class accl_cpu_mock_subplugin : public accl_mock_subplugin
{
  public:
  static const char *cpu_mock_name;
  static accl_cpu_mock_subplugin *cpu_registered;
  static const accl_hw cpu_hw_list[];

  /** @brief mandatory method */
  tensor_filter_subplugin &getEmptyInstance () override
  {
    return *(new accl_cpu_mock_subplugin ());
  }

  /** @brief list the CPU only */
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info) override
  {
    accl_mock_subplugin::getFrameworkInfo (info);
    info.name = cpu_mock_name;
    info.hw_list = cpu_hw_list;
    info.num_hw = 1;
  }

  /** @brief register this mock subplugin */
  static void init ()
  {
    cpu_registered = register_subplugin<accl_cpu_mock_subplugin> ();
  }

  /** @brief unregister this mock subplugin */
  static void fini ()
  {
    unregister_subplugin<accl_cpu_mock_subplugin> (cpu_registered);
    cpu_registered = nullptr;
  }
};

const char *accl_cpu_mock_subplugin::cpu_mock_name = "accl_cpu_mock_subplugin";
accl_cpu_mock_subplugin *accl_cpu_mock_subplugin::cpu_registered = nullptr;
const accl_hw accl_cpu_mock_subplugin::cpu_hw_list[] = { ACCL_CPU };

/**
 * @brief V1 C++ sub-plugin preferring NEON, then the GPU.
 */
class accl_neon_mock_subplugin : public accl_mock_subplugin
{
  public:
  static const char *neon_mock_name;
  static accl_neon_mock_subplugin *neon_registered;
  static const accl_hw neon_hw_list[];
  static int neon_num_hw;

  /** @brief mandatory method */
  tensor_filter_subplugin &getEmptyInstance () override
  {
    return *(new accl_neon_mock_subplugin ());
  }

  /** @brief list NEON first, and use it for auto and default */
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info) override
  {
    accl_mock_subplugin::getFrameworkInfo (info);
    info.name = neon_mock_name;
    info.hw_list = neon_hw_list;
    info.num_hw = neon_num_hw;
    info.accl_auto = ACCL_CPU_NEON;
    info.accl_default = ACCL_CPU_NEON;
  }

  /** @brief register this mock subplugin */
  static void init ()
  {
    neon_registered = register_subplugin<accl_neon_mock_subplugin> ();
  }

  /** @brief unregister this mock subplugin */
  static void fini ()
  {
    unregister_subplugin<accl_neon_mock_subplugin> (neon_registered);
    neon_registered = nullptr;
  }
};

const char *accl_neon_mock_subplugin::neon_mock_name = "accl_neon_mock_subplugin";
accl_neon_mock_subplugin *accl_neon_mock_subplugin::neon_registered = nullptr;
const accl_hw accl_neon_mock_subplugin::neon_hw_list[] = { ACCL_CPU_NEON, ACCL_GPU };
int accl_neon_mock_subplugin::neon_num_hw = 2;

static gint fake_neon_ret = 0;
static guint fake_neon_calls = 0;

/**
 * @brief Replace the NEON check of libnnstreamer to test a device without NEON.
 * @return fake_neon_ret, 0 (NEON available) unless a test changes it.
 */
extern "C" gint
cpu_neon_accel_available (void)
{
  fake_neon_calls++;
  return fake_neon_ret;
}

#if !defined(__TIZEN__) && !defined(__ANDROID__)
/**
 * @brief Count the critical messages of the default log domain.
 */
static void
_count_accl_critical (const gchar *log_domain, GLogLevelFlags log_level,
    const gchar *message, gpointer user_data)
{
  UNUSED (log_domain);
  UNUSED (log_level);
  UNUSED (message);
  (*(guint *) user_data)++;
}

/**
 * @brief Start counting the critical messages of the default log domain, after checking the handler gets them.
 * @param count the counter, reset to 0 once the handler is checked
 * @return The handler id, to be removed with g_log_remove_handler ().
 */
static guint
_start_accl_critical_count (guint *count)
{
  guint handler = g_log_set_handler (NULL,
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL),
      _count_accl_critical, count);
  GLogLevelFlags fatal_mask;

  *count = 0;
  fatal_mask = g_log_set_always_fatal ((GLogLevelFlags) G_LOG_FATAL_MASK);
  g_log (NULL, G_LOG_LEVEL_CRITICAL, "[probe] accelerator log handler self-check");
  g_log_set_always_fatal (fatal_mask);
  EXPECT_EQ (*count, 1U) << "no live handler for critical messages";
  *count = 0;

  return handler;
}
#endif

/**
 * @brief Test fixture setting the accelerator property of a filter the way a pipeline description does.
 */
class testFilterAccelerator : public ::testing::Test
{
  protected:
  GstTensorFilterPrivate priv;

  /** @brief register the V1 mocks */
  void SetUp () override
  {
    accl_mock_subplugin::seen_accl_str = nullptr;
    accl_mock_subplugin::seen_num_hw = 0;
    accl_mock_subplugin::seen_hw = ACCL_NONE;
    accl_mock_subplugin::event_ret = 0;
    accl_mock_subplugin::basic_hw = ACCL_CPU;
    accl_mock_subplugin::init ();
    accl_cpu_mock_subplugin::init ();
    accl_neon_mock_subplugin::neon_num_hw = 2;
    accl_neon_mock_subplugin::init ();
    fake_neon_ret = 0;
    fake_neon_calls = 0;

    gst_tensor_filter_common_init_property (&priv);
  }

  /** @brief close the filter and unregister the mocks */
  void TearDown () override
  {
    gst_tensor_filter_common_close_fw (&priv);
    gst_tensor_filter_common_free_property (&priv);
    accl_neon_mock_subplugin::fini ();
    accl_cpu_mock_subplugin::fini ();
    accl_mock_subplugin::fini ();
    fake_neon_ret = 0;

    g_free (accl_mock_subplugin::seen_accl_str);
    accl_mock_subplugin::seen_accl_str = nullptr;
  }

  /**
   * @brief Set a string property of the filter.
   * @param prop_id the property to set
   * @param str the value of the property
   */
  void setString (guint prop_id, const gchar *str)
  {
    GValue value = G_VALUE_INIT;

    g_value_init (&value, G_TYPE_STRING);
    g_value_set_string (&value, str);
    EXPECT_TRUE (gst_tensor_filter_common_set_property (&priv, prop_id, &value, NULL));
    g_value_unset (&value);
  }

  /**
   * @brief Check that the NEON check of libnnstreamer is replaced in this build.
   * @return true if cpu_neon_accel_available () of this test is called.
   */
  bool neonReplaced ()
  {
    const gchar *probe_list[] = { ACCL_CPU_NEON_STR, NULL };

    fake_neon_calls = 0;
    parse_accl_hw ("true", probe_list, NULL, NULL);
    return fake_neon_calls > 0;
  }

  /**
   * @brief Read the accelerator property of the filter back.
   * @return the value of the property. Free it with g_free().
   */
  gchar *getAccelerator ()
  {
    GValue value = G_VALUE_INIT;
    gchar *str;

    g_value_init (&value, G_TYPE_STRING);
    EXPECT_TRUE (gst_tensor_filter_common_get_property (&priv, PROP_ACCELERATOR, &value, NULL));
    str = g_value_dup_string (&value);
    g_value_unset (&value);

    return str;
  }
};

/**
 * @brief A V1 sub-plugin opened after "framework=... accelerator=..." gets the
 *        string the pipeline set, as a V0 sub-plugin does, and the parsed list.
 *        The string is read back as it was set.
 */
TEST_F (testFilterAccelerator, afterFramework)
{
  g_autofree gchar *read = NULL;

  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  setString (PROP_ACCELERATOR, "true:npu.edgetpu,gpu");
  ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));

  EXPECT_STREQ (accl_mock_subplugin::seen_accl_str, "true:npu.edgetpu,gpu");
  EXPECT_EQ (accl_mock_subplugin::seen_num_hw, 2);
  EXPECT_EQ (accl_mock_subplugin::seen_hw, ACCL_NPU_EDGE_TPU);
  ASSERT_EQ (priv.prop.num_hw, 2);
  EXPECT_EQ (priv.prop.hw_list[1], ACCL_GPU);

  read = getAccelerator ();
  EXPECT_STREQ (read, "true:npu.edgetpu,gpu");
}

/**
 * @brief A V1 sub-plugin gets the same accelerator when the pipeline sets it before the framework.
 */
TEST_F (testFilterAccelerator, beforeFramework)
{
  g_autofree gchar *read = NULL;

  setString (PROP_ACCELERATOR, "true:npu.edgetpu,gpu");
  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));

  EXPECT_STREQ (accl_mock_subplugin::seen_accl_str, "true:npu.edgetpu,gpu");
  EXPECT_EQ (accl_mock_subplugin::seen_num_hw, 2);
  EXPECT_EQ (accl_mock_subplugin::seen_hw, ACCL_NPU_EDGE_TPU);

  read = getAccelerator ();
  EXPECT_STREQ (read, "true:npu.edgetpu,gpu");
}

/**
 * @brief The string set after a V0 framework is kept and read back as it was set.
 */
TEST_F (testFilterAccelerator, afterFrameworkV0)
{
  g_autofree gchar *read = NULL;

  setString (PROP_FRAMEWORK, "custom");
  ASSERT_TRUE (priv.fw != NULL && GST_TF_FW_V0 (priv.fw));
  setString (PROP_ACCELERATOR, "true:npu.edgetpu,gpu");

  EXPECT_STREQ (priv.prop.accl_str, "true:npu.edgetpu,gpu");
  EXPECT_EQ (priv.prop.num_hw, 0);

  read = getAccelerator ();
  EXPECT_STREQ (read, "true:npu.edgetpu,gpu");
}

/**
 * @brief Replacing a V0 framework with a V1 one keeps the accelerator set after the V0 one.
 */
TEST_F (testFilterAccelerator, frameworkV0ToV1)
{
  setString (PROP_FRAMEWORK, "custom");
  setString (PROP_ACCELERATOR, "true:gpu");
  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));

  EXPECT_STREQ (accl_mock_subplugin::seen_accl_str, "true:gpu");
  EXPECT_EQ (accl_mock_subplugin::seen_num_hw, 1);
  EXPECT_EQ (accl_mock_subplugin::seen_hw, ACCL_GPU);
}

/**
 * @brief Replacing a V1 framework with a V0 one keeps the accelerator set after the V1 one.
 */
TEST_F (testFilterAccelerator, frameworkV1ToV0)
{
  g_autofree gchar *read = NULL;

  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  setString (PROP_ACCELERATOR, "true:gpu");
  setString (PROP_FRAMEWORK, "custom");
  ASSERT_TRUE (priv.fw != NULL && GST_TF_FW_V0 (priv.fw));

  EXPECT_STREQ (priv.prop.accl_str, "true:gpu");
  read = getAccelerator ();
  EXPECT_STREQ (read, "true:gpu");
}

/**
 * @brief "false" set after a V1 framework disables the acceleration and is read back as it was set.
 */
TEST_F (testFilterAccelerator, disabledAfterFramework)
{
  g_autofree gchar *read = NULL;

  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  setString (PROP_ACCELERATOR, "false");
  ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));

  EXPECT_STREQ (accl_mock_subplugin::seen_accl_str, "false");
  EXPECT_EQ (accl_mock_subplugin::seen_num_hw, 1);
  EXPECT_EQ (accl_mock_subplugin::seen_hw, ACCL_NONE);

  read = getAccelerator ();
  EXPECT_STREQ (read, "false");
}

/**
 * @brief An accelerator the V1 framework does not support falls back on its auto accelerator.
 */
TEST_F (testFilterAccelerator, unsupportedAfterFramework_n)
{
  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  setString (PROP_ACCELERATOR, "true:npu");
  ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));

  EXPECT_STREQ (accl_mock_subplugin::seen_accl_str, "true:npu");
  EXPECT_EQ (accl_mock_subplugin::seen_num_hw, 1);
  EXPECT_EQ (accl_mock_subplugin::seen_hw, ACCL_CPU);
}

/**
 * @brief An opened V1 framework refusing a new accelerator keeps the one it
 *        was opened with, and the refusal is logged as V0 logs it.
 */
TEST_F (testFilterAccelerator, changeRefusedWhileOpened_n)
{
  g_autofree gchar *read = NULL;

  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  setString (PROP_ACCELERATOR, "true:gpu");
  ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));

  accl_mock_subplugin::event_ret = -ENOENT;
#if !defined(__TIZEN__) && !defined(__ANDROID__)
  guint critical_count;
  guint handler = _start_accl_critical_count (&critical_count);
#endif
  setString (PROP_ACCELERATOR, "true:npu.edgetpu");
#if !defined(__TIZEN__) && !defined(__ANDROID__)
  g_log_remove_handler (NULL, handler);
  EXPECT_EQ (critical_count, 1U);
#endif

  EXPECT_STREQ (priv.prop.accl_str, "true:gpu");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_GPU);
  read = getAccelerator ();
  EXPECT_STREQ (read, "true:gpu");
}

/**
 * @brief An opened V1 framework taking a new accelerator keeps the string it was parsed from.
 */
TEST_F (testFilterAccelerator, changeWhileOpened)
{
  g_autofree gchar *read = NULL;

  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  setString (PROP_ACCELERATOR, "true:gpu");
  ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));

#if !defined(__TIZEN__) && !defined(__ANDROID__)
  guint critical_count;
  guint handler = _start_accl_critical_count (&critical_count);
#endif
  setString (PROP_ACCELERATOR, "true:npu.edgetpu");
#if !defined(__TIZEN__) && !defined(__ANDROID__)
  g_log_remove_handler (NULL, handler);
  EXPECT_EQ (critical_count, 0U);
#endif

  EXPECT_STREQ (priv.prop.accl_str, "true:npu.edgetpu");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_NPU_EDGE_TPU);
  read = getAccelerator ();
  EXPECT_STREQ (read, "true:npu.edgetpu");
}

/**
 * @brief Replacing a V1 framework with another V1 one parses the accelerator again for the new one.
 */
TEST_F (testFilterAccelerator, frameworkV1ToV1)
{
  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  setString (PROP_ACCELERATOR, "true:gpu");
  ASSERT_EQ (priv.prop.num_hw, 1);

  setString (PROP_FRAMEWORK, accl_cpu_mock_subplugin::cpu_mock_name);
  EXPECT_STREQ (priv.prop.accl_str, "true:gpu");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_CPU);

  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));
  EXPECT_STREQ (accl_mock_subplugin::seen_accl_str, "true:gpu");
  EXPECT_EQ (accl_mock_subplugin::seen_num_hw, 1);
  EXPECT_EQ (accl_mock_subplugin::seen_hw, ACCL_GPU);
}

/**
 * @brief A V1 framework leaving auto and default at -1 gets the first accelerator it lists for them.
 */
TEST_F (testFilterAccelerator, basicUnsetFirstHw)
{
  accl_mock_subplugin::basic_hw = (accl_hw) -1;
  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);

  setString (PROP_ACCELERATOR, "true:auto");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_CPU);

  setString (PROP_ACCELERATOR, "true:default");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_CPU);
}

/**
 * @brief The value read back from a V1 framework can be set again and gives the same accelerators.
 */
TEST_F (testFilterAccelerator, readBackSetAgain)
{
  g_autofree gchar *read = NULL;
  g_autofree gchar *read_again = NULL;

  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  setString (PROP_ACCELERATOR, "true:!npu.edgetpu,gpu,cpu");
  ASSERT_EQ (priv.prop.num_hw, 2);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_GPU);
  EXPECT_EQ (priv.prop.hw_list[1], ACCL_CPU);

  read = getAccelerator ();
  EXPECT_STREQ (read, "true:!npu.edgetpu,gpu,cpu");
  setString (PROP_ACCELERATOR, read);
  ASSERT_EQ (priv.prop.num_hw, 2);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_GPU);
  EXPECT_EQ (priv.prop.hw_list[1], ACCL_CPU);

  read_again = getAccelerator ();
  EXPECT_STREQ (read_again, read);
}

/**
 * @brief A V1 framework with no accelerator set reads back an empty string, as a V0 one does.
 */
TEST_F (testFilterAccelerator, notSetV1)
{
  g_autofree gchar *read = NULL;

  setString (PROP_FRAMEWORK, accl_mock_subplugin::mock_name);
  ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));
  EXPECT_EQ (priv.prop.num_hw, 0);

  read = getAccelerator ();
  EXPECT_STREQ (read, "");
}

/**
 * @brief A V1 framework keeps NEON for auto and default on a device with NEON.
 */
TEST_F (testFilterAccelerator, neonAvailable)
{
  g_autofree gchar *neon_gpu
      = g_strdup_printf ("true:%s,gpu", get_accl_hw_str (ACCL_CPU_NEON));

  if (!neonReplaced ())
    GTEST_SKIP () << "cpu_neon_accel_available () cannot be replaced in this build.";

  setString (PROP_FRAMEWORK, accl_neon_mock_subplugin::neon_mock_name);
  setString (PROP_ACCELERATOR, "true:auto");

  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_CPU_NEON);

  setString (PROP_ACCELERATOR, "true:default");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_CPU_NEON);

  setString (PROP_ACCELERATOR, neon_gpu);
  ASSERT_EQ (priv.prop.num_hw, 2);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_CPU_NEON);
  EXPECT_EQ (priv.prop.hw_list[1], ACCL_GPU);
}

/**
 * @brief A V1 framework drops NEON on a device without NEON, so auto and
 *        default fall back on its next accelerator.
 */
TEST_F (testFilterAccelerator, neonUnavailable_n)
{
  g_autofree gchar *neon_gpu
      = g_strdup_printf ("true:%s,gpu", get_accl_hw_str (ACCL_CPU_NEON));

  if (!neonReplaced ())
    GTEST_SKIP () << "cpu_neon_accel_available () cannot be replaced in this build.";

  fake_neon_ret = -EINVAL;
  setString (PROP_FRAMEWORK, accl_neon_mock_subplugin::neon_mock_name);
  setString (PROP_ACCELERATOR, "true:auto");

  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_GPU);

  setString (PROP_ACCELERATOR, "true:default");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_GPU);

  setString (PROP_ACCELERATOR, "true:gpu");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_GPU);

  setString (PROP_ACCELERATOR, "true:cpu.neon,gpu");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_GPU);

  setString (PROP_ACCELERATOR, "true:cpu.neon");
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_GPU);

  /* NEON by the name of this architecture (cpu.simd on non-ARM) */
  setString (PROP_ACCELERATOR, neon_gpu);
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_GPU);
}

/**
 * @brief A V1 framework supporting NEON only gets no acceleration on a device without NEON.
 */
TEST_F (testFilterAccelerator, neonOnlyUnavailable_n)
{
  if (!neonReplaced ())
    GTEST_SKIP () << "cpu_neon_accel_available () cannot be replaced in this build.";

  fake_neon_ret = -EINVAL;
  accl_neon_mock_subplugin::neon_num_hw = 1;
  setString (PROP_FRAMEWORK, accl_neon_mock_subplugin::neon_mock_name);
  setString (PROP_ACCELERATOR, "true:auto");

  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], ACCL_NONE);
}

/**
 * @brief V0 and V1 frameworks listing the same accelerators resolve "auto" to
 *        the same one on a device without NEON.
 */
TEST_F (testFilterAccelerator, neonUnavailableSameAsV0_n)
{
  const gchar *v0_hw_list[] = { ACCL_CPU_NEON_STR, ACCL_GPU_STR, NULL };
  accl_hw v0_hw;

  if (!neonReplaced ())
    GTEST_SKIP () << "cpu_neon_accel_available () cannot be replaced in this build.";

  fake_neon_ret = -EINVAL;
  v0_hw = parse_accl_hw ("true:auto", v0_hw_list, ACCL_CPU_NEON_STR, ACCL_CPU_NEON_STR);
  setString (PROP_FRAMEWORK, accl_neon_mock_subplugin::neon_mock_name);
  setString (PROP_ACCELERATOR, "true:auto");

  EXPECT_EQ (v0_hw, ACCL_GPU);
  ASSERT_EQ (priv.prop.num_hw, 1);
  EXPECT_EQ (priv.prop.hw_list[0], v0_hw);
}

/**
 * @brief C++ sub-plugin whose asynchronous output is released by the test.
 */
class async_output_subplugin : public nnstreamer::tensor_filter_subplugin
{
  public:
  static async_output_subplugin *registered; /**< registered factory */
  static std::mutex lock; /**< protects the invocation handoff */
  static std::condition_variable cond; /**< signals an input or model close */
  static bool closed; /**< records suspend closing the active instance */
  static guint invocations; /**< number of inputs accepted by the mock */
  bool configured = false; /**< distinguishes an instance from the factory */
  GstTensorFilterProperties *properties = nullptr; /**< configured instance properties */
  static GstTensorFilterProperties *pending; /**< live properties until teardown */

  /** @brief record when a configured instance is suspended or stopped */
  ~async_output_subplugin () override
  {
    if (configured) {
      std::lock_guard<std::mutex> guard (lock);
      closed = true;
      cond.notify_all ();
    }
  }

  /** @brief create an independent instance */
  tensor_filter_subplugin &getEmptyInstance () override
  {
    return *(new async_output_subplugin ());
  }

  /** @brief accept the fixture's temporary model file */
  void configure_instance (const GstTensorFilterProperties *prop) override
  {
    properties = const_cast<GstTensorFilterProperties *> (prop);
    configured = true;
  }

  /** @brief queue an input through the non-dynamic entry point */
  void invoke (const GstTensorMemory *input, GstTensorMemory *output) override
  {
    invoke_dynamic (properties, input, output);
  }

  /** @brief queue async inputs, or return an allocated synchronous result */
  void invoke_dynamic (GstTensorFilterProperties *prop,
      const GstTensorMemory *input, GstTensorMemory *output) override
  {
    if (!prop->invoke_async)
      gst_tensors_info_copy (&prop->output_meta, &prop->input_meta);
    {
      std::lock_guard<std::mutex> guard (lock);
      pending = prop;
      invocations++;
      cond.notify_all ();
    }

    if (g_strcmp0 (prop->custom_properties, "fail") == 0)
      throw std::invalid_argument ("Requested invocation failure");

    if (!prop->invoke_async) {
      output[0].data = _g_memdup (input[0].data, input[0].size);
      output[0].size = input[0].size;
    }
  }

  /** @brief describe an allocating, model-backed V1 framework */
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info) override
  {
    info.name = "async_output_mock";
    info.allow_in_place = FALSE;
    info.allocate_in_invoke = TRUE;
    info.run_without_model = FALSE;
    info.verify_model_path = TRUE;
    info.hw_list = nullptr;
    info.num_hw = 0;
  }

  /** @brief report one uint8 tensor in each direction */
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info) override
  {
    if (ops != GET_IN_OUT_INFO)
      return -ENOENT;

    gst_tensors_info_init (&in_info);
    in_info.num_tensors = 1;
    in_info.info[0].type = _NNS_UINT8;
    for (guint i = 0; i < 4; i++)
      in_info.info[0].dimension[i] = 1;
    gst_tensors_info_copy (&out_info, &in_info);
    return 0;
  }

  /** @brief provide the framework's optional event hook */
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data) override
  {
    UNUSED (ops);
    UNUSED (data);
    return -ENOENT;
  }

  /** @brief register the mock factory */
  static void init ()
  {
    registered = register_subplugin<async_output_subplugin> ();
  }

  /** @brief unregister the mock factory */
  static void fini ()
  {
    if (registered)
      unregister_subplugin<async_output_subplugin> (registered);
    registered = nullptr;
  }

  /** @brief wait for the streaming thread to queue its input */
  static bool waitForInput (guint count = 1)
  {
    std::unique_lock<std::mutex> guard (lock);
    return cond.wait_for (guard, std::chrono::seconds (5),
        [count] { return invocations >= count; });
  }

  /** @brief wait for the suspend watchdog to close the model */
  static bool waitForClose ()
  {
    std::unique_lock<std::mutex> guard (lock);
    return cond.wait_for (guard, std::chrono::seconds (2), [] { return closed; });
  }

  /** @brief dispatch one result independently of the input's streaming thread */
  static void dispatch (guint8 value)
  {
    GstTensorMemory output = { _g_memdup (&value, sizeof (value)), sizeof (value) };
    GstTensorFilterProperties *prop;

    {
      std::lock_guard<std::mutex> guard (lock);
      prop = pending;
    }
    gst_tensors_info_copy (&prop->output_meta, &prop->input_meta);
    nnstreamer_filter_dispatch_output_async (prop, &output);
  }
};

async_output_subplugin *async_output_subplugin::registered = nullptr;
std::mutex async_output_subplugin::lock;
std::condition_variable async_output_subplugin::cond;
bool async_output_subplugin::closed = false;
guint async_output_subplugin::invocations = 0;
GstTensorFilterProperties *async_output_subplugin::pending = nullptr;

/** @brief Exercise queued inputs and callback outputs without an ML runtime. */
class testFilterAsyncOutput : public ::testing::Test
{
  protected:
  GstElement *pipeline; /**< pipeline owned by the fixture */
  GstElement *sink; /**< retained appsink reference */
  gchar *model; /**< dummy file used to open the mock */

  /** @brief register the mock and create its model placeholder */
  void SetUp () override
  {
    pipeline = sink = nullptr;
    model = nullptr;
    async_output_subplugin::pending = nullptr;
    async_output_subplugin::closed = false;
    async_output_subplugin::invocations = 0;
    async_output_subplugin::init ();
    ASSERT_NE (async_output_subplugin::registered, nullptr);
    int fd = g_file_open_tmp ("nnstreamer-async-XXXXXX", &model, nullptr);
    ASSERT_GE (fd, 0);
    close (fd);
  }

  /** @brief stop streaming before unregistering the mock */
  void TearDown () override
  {
    if (pipeline) {
      gst_element_set_state (pipeline, GST_STATE_NULL);
      gst_object_unref (pipeline);
    }
    if (sink)
      gst_object_unref (sink);
    async_output_subplugin::fini ();
    async_output_subplugin::pending = nullptr;
    if (model)
      g_unlink (model);
    g_free (model);
  }

  /** @brief submit one byte to a dynamic or non-dynamic C++ filter */
  void start (bool asynchronous, bool fail = false, guint suspend = 0, bool dynamic = true)
  {
    g_autofree gchar *description = g_strdup_printf (
        "appsrc name=source caps=\"other/tensors,num_tensors=1,types=uint8,"
        "dimensions=1:1:1:1,format=static,framerate=0/1\" ! "
        "tensor_filter framework=async_output_mock model=%s invoke-dynamic=%s "
        "invoke-async=%s custom=%s suspend=%u ! other/tensors,format=flexible ! "
        "appsink name=sink sync=false async=false",
        model, dynamic ? "true" : "false", asynchronous ? "true" : "false",
        fail ? "fail" : "ok", suspend);
    GError *error = nullptr;
    pipeline = gst_parse_launch (description, &error);
    ASSERT_EQ (error, nullptr);
    ASSERT_NE (pipeline, nullptr);
    sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
    GstElement *source = gst_bin_get_by_name (GST_BIN (pipeline), "source");
    ASSERT_NE (sink, nullptr);
    ASSERT_NE (source, nullptr);
    ASSERT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    guint8 value = 0x2a;
    GstBuffer *input = gst_buffer_new_allocate (nullptr, sizeof (value), nullptr);
    gst_buffer_fill (input, 0, &value, sizeof (value));
    EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (source), input), GST_FLOW_OK);
    gst_object_unref (source);
    ASSERT_TRUE (async_output_subplugin::waitForInput ());
  }

  /** @brief check the scalar payload after its flexible tensor header */
  void expectValue (GstSample *sample, guint8 value)
  {
    ASSERT_NE (sample, nullptr);
    GstBuffer *buffer = gst_sample_get_buffer (sample);
    ASSERT_EQ (gst_buffer_n_memory (buffer), 1U);
    GstMapInfo map;
    ASSERT_TRUE (gst_buffer_map (buffer, &map, GST_MAP_READ));
    EXPECT_GT (map.size, 1U);
    if (map.size > 0) {
      EXPECT_EQ (map.data[map.size - 1], value);
    }
    gst_buffer_unmap (buffer, &map);
  }
};

/** @brief Queued inputs produce nothing until a callback supplies real data. */
TEST_F (testFilterAsyncOutput, noPlaceholderBuffer)
{
  ASSERT_NO_FATAL_FAILURE (start (true));
  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 100 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);

  async_output_subplugin::dispatch (0x5a);
  sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
  EXPECT_NO_FATAL_FAILURE (expectValue (sample, 0x5a));
  if (sample)
    gst_sample_unref (sample);
  sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 100 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);
}

/** @brief One queued input may generate several ordered callback outputs. */
TEST_F (testFilterAsyncOutput, multipleCallbacksFromOneInput)
{
  ASSERT_NO_FATAL_FAILURE (start (true));
  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 50 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);

  for (guint8 value = 1; value <= 8; value++) {
    async_output_subplugin::dispatch (value);
    sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
    EXPECT_NO_FATAL_FAILURE (expectValue (sample, value));
    if (sample)
      gst_sample_unref (sample);
  }
  sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 50 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);
}

/** @brief Dropping several inputs must not add a sample for any of them. */
TEST_F (testFilterAsyncOutput, inputBurstWithoutPlaceholders)
{
  ASSERT_NO_FATAL_FAILURE (start (true));
  GstElement *source = gst_bin_get_by_name (GST_BIN (pipeline), "source");
  ASSERT_NE (source, nullptr);
  for (guint8 value = 1; value < 16; value++) {
    GstBuffer *input = gst_buffer_new_allocate (nullptr, sizeof (value), nullptr);
    gst_buffer_fill (input, 0, &value, sizeof (value));
    EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (source), input), GST_FLOW_OK);
  }
  gst_object_unref (source);
  ASSERT_TRUE (async_output_subplugin::waitForInput (16));
  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 50 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);

  async_output_subplugin::dispatch (0x7f);
  sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
  EXPECT_NO_FATAL_FAILURE (expectValue (sample, 0x7f));
  if (sample)
    gst_sample_unref (sample);
  sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 50 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);
}

/** @brief An accepted input with no generated output can finish with EOS. */
TEST_F (testFilterAsyncOutput, eosWithoutGeneratedOutput)
{
  ASSERT_NO_FATAL_FAILURE (start (true));
  GstElement *source = gst_bin_get_by_name (GST_BIN (pipeline), "source");
  ASSERT_NE (source, nullptr);
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (source)), GST_FLOW_OK);
  gst_object_unref (source);

  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *message = gst_bus_timed_pop_filtered (
      bus, GST_SECOND, (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  EXPECT_NE (message, nullptr);
  if (message) {
    EXPECT_EQ (GST_MESSAGE_TYPE (message), GST_MESSAGE_EOS);
    gst_message_unref (message);
  }
  EXPECT_TRUE (gst_app_sink_is_eos (GST_APP_SINK (sink)));
  gst_object_unref (bus);
}

/** @brief A synchronous invoke still forwards its allocated result. */
TEST_F (testFilterAsyncOutput, synchronousOutput)
{
  ASSERT_NO_FATAL_FAILURE (start (false));
  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
  EXPECT_NO_FATAL_FAILURE (expectValue (sample, 0x2a));
  if (sample)
    gst_sample_unref (sample);
}

/** @brief Async mode does not turn a failed invocation into a dropped buffer. */
TEST_F (testFilterAsyncOutput, invokeError_n)
{
  ASSERT_NO_FATAL_FAILURE (start (true, true));
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *error = gst_bus_timed_pop_filtered (bus, GST_SECOND, GST_MESSAGE_ERROR);
  EXPECT_NE (error, nullptr);
  if (error)
    gst_message_unref (error);
  gst_object_unref (bus);
}

/** @brief Count a single-shot callback while checking its scalar result. */
static int
_record_single_async_output (GstTensorMemory *data, GstTensorsInfo *info, void *user_data)
{
  guint *count = (guint *) user_data;
  EXPECT_EQ (info->num_tensors, 1U);
  EXPECT_EQ (data[0].size, 1U);
  EXPECT_EQ (*(guint8 *) data[0].data, 0x5a);
  (*count)++;
  return 0;
}

/** @brief The single-shot API still accepts a successfully queued async input. */
TEST_F (testFilterAsyncOutput, singleShotAsyncSuccess)
{
  g_autoptr (GObject) object = (GObject *) g_object_new (G_TYPE_TENSOR_FILTER_SINGLE,
      "framework", "async_output_mock", "model", model, "invoke-dynamic", TRUE,
      "invoke-async", TRUE, nullptr);
  auto *single = G_TENSOR_FILTER_SINGLE (object);
  auto *klass = G_TENSOR_FILTER_SINGLE_CLASS (G_OBJECT_GET_CLASS (single));
  guint8 value = 0x2a;
  guint count = 0;
  GstTensorMemory input = { &value, sizeof (value) };
  GstTensorMemory output = {};

  ASSERT_TRUE (klass->set_invoke_async_callback (single, _record_single_async_output, &count));
  EXPECT_TRUE (klass->invoke (single, &input, &output, TRUE));
  EXPECT_EQ (output.data, nullptr);
  EXPECT_EQ (count, 0U);
  async_output_subplugin::dispatch (0x5a);
  EXPECT_EQ (count, 1U);
}

/** @brief Accepting an async input still arms the idle suspend watchdog. */
TEST_F (testFilterAsyncOutput, asyncSuspend)
{
  ASSERT_NO_FATAL_FAILURE (start (true, false, 50));
  EXPECT_TRUE (async_output_subplugin::waitForClose ());
}

/** @brief Build metadata for independently owned variable-size results. */
static void
_async_result_info (GstTensorsInfo *info, guint count, guint size)
{
  gst_tensors_info_init (info);
  info->num_tensors = count;
  info->format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  for (guint i = 0; i < count; i++) {
    GstTensorInfo *tensor = gst_tensors_info_get_nth_info (info, i);
    tensor->type = _NNS_UINT8;
    tensor->dimension[0] = size;
    for (guint j = 1; j < 4; j++)
      tensor->dimension[j] = 1;
  }
}

/** @brief Per-result count and dimensions must override negotiated metadata. */
TEST_F (testFilterAsyncOutput, independentOutputMetadata)
{
  ASSERT_NO_FATAL_FAILURE (start (true));
  for (guint count = 2; count <= 3; count++) {
    GstTensorsInfo info;
    GstTensorMemory output[3] = {};
    _async_result_info (&info, count, count + 5);
    for (guint i = 0; i < count; i++) {
      output[i].size = count + 5;
      output[i].data = g_malloc0 (output[i].size);
    }
    nnstreamer_filter_dispatch_output_async_with_info (
        async_output_subplugin::pending, output, &info);
    // The callback consumes the description synchronously, not its ownership.
    gst_tensors_info_free (&info);
    GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
    ASSERT_NE (sample, nullptr);
    GstBuffer *buffer = gst_sample_get_buffer (sample);
    EXPECT_EQ (gst_tensor_buffer_get_count (buffer), count);
    for (guint i = 0; i < gst_tensor_buffer_get_count (buffer); i++) {
      GstMemory *memory = gst_tensor_buffer_get_nth_memory (buffer, i);
      GstMapInfo map;
      if (gst_memory_map (memory, &map, GST_MAP_READ)) {
        GstTensorMetaInfo meta;
        gst_tensor_meta_info_init (&meta);
        EXPECT_TRUE (gst_tensor_meta_info_parse_header (&meta, map.data));
        EXPECT_EQ (meta.dimension[0], count + 5);
        EXPECT_EQ (meta.type, _NNS_UINT8);
        EXPECT_EQ (map.size, gst_tensor_meta_info_get_header_size (&meta) + count + 5);
        gst_memory_unmap (memory, &map);
      } else {
        ADD_FAILURE () << "Cannot map the async result";
      }
      gst_memory_unref (memory);
    }
    gst_sample_unref (sample);
    EXPECT_EQ (async_output_subplugin::pending->output_meta.num_tensors, 0U);
  }
}

/** @brief A legacy worker may resize metadata while more inputs are queued. */
TEST_F (testFilterAsyncOutput, concurrentLegacyMetadata)
{
  ASSERT_NO_FATAL_FAILURE (start (true));
  GstElement *source = gst_bin_get_by_name (GST_BIN (pipeline), "source");
  ASSERT_NE (source, nullptr);
  std::atomic<bool> go{ false };
  GstTensorFilterProperties *prop;
  {
    std::lock_guard<std::mutex> guard (async_output_subplugin::lock);
    prop = async_output_subplugin::pending;
  }
  std::thread worker ([prop, &go] {
    while (!go.load ())
      std::this_thread::yield ();
    gst_tensors_info_copy (&prop->output_meta, &prop->input_meta);
    for (guint i = 0; i < 128; i++) {
      guint size = 1 + i % 8;
      prop->output_meta.info[0].dimension[0] = size;
      GstTensorMemory output = { g_malloc0 (size), size };
      nnstreamer_filter_dispatch_output_async (prop, &output);
    }
  });
  go.store (true);
  for (guint i = 0; i < 256; i++) {
    GstBuffer *input = gst_buffer_new_allocate (nullptr, 1, nullptr);
    EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (source), input), GST_FLOW_OK);
  }
  worker.join ();
  gst_object_unref (source);
  EXPECT_TRUE (async_output_subplugin::waitForInput (257));
  for (guint i = 0; i < 128; i++) {
    GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
    ASSERT_NE (sample, nullptr);
    GstBuffer *buffer = gst_sample_get_buffer (sample);
    GstMapInfo map;
    ASSERT_TRUE (gst_buffer_map (buffer, &map, GST_MAP_READ));
    GstTensorMetaInfo meta;
    gst_tensor_meta_info_init (&meta);
    EXPECT_TRUE (gst_tensor_meta_info_parse_header (&meta, map.data));
    EXPECT_EQ (meta.dimension[0], 1 + i % 8);
    EXPECT_EQ (map.size, gst_tensor_meta_info_get_header_size (&meta) + 1 + i % 8);
    gst_buffer_unmap (buffer, &map);
    gst_sample_unref (sample);
  }
}

/** @brief Count the tensors delivered by a single-shot per-result callback. */
static int
_count_single_result (GstTensorMemory *data, GstTensorsInfo *info, void *user_data)
{
  UNUSED (data);
  *(guint *) user_data += info->num_tensors;
  return 0;
}

/** @brief Single-shot cleanup follows the callback count, not the model count. */
TEST_F (testFilterAsyncOutput, singleShotIndependentMetadata)
{
  g_autoptr (GObject) object = (GObject *) g_object_new (G_TYPE_TENSOR_FILTER_SINGLE,
      "framework", "async_output_mock", "model", model, "invoke-dynamic", TRUE,
      "invoke-async", TRUE, nullptr);
  auto *single = G_TENSOR_FILTER_SINGLE (object);
  auto *klass = G_TENSOR_FILTER_SINGLE_CLASS (G_OBJECT_GET_CLASS (single));
  guint8 value = 0x2a;
  guint count = 0;
  GstTensorMemory input = { &value, sizeof (value) };
  GstTensorMemory unused = {};
  ASSERT_TRUE (klass->set_invoke_async_callback (single, _count_single_result, &count));
  ASSERT_TRUE (klass->invoke (single, &input, &unused, FALSE));
  GstTensorsInfo info;
  _async_result_info (&info, 2, 4);
  GstTensorMemory output[2] = { { g_malloc0 (4), 4 }, { g_malloc0 (4), 4 } };
  nnstreamer_filter_dispatch_output_async_with_info (
      async_output_subplugin::pending, output, &info);
  EXPECT_EQ (count, 2U);
  EXPECT_EQ (output[0].data, nullptr);
  EXPECT_EQ (output[1].data, nullptr);
  EXPECT_EQ (unused.data, nullptr);
  gst_tensors_info_free (&info);
}

/** @brief Bad tensor descriptions produce no sample and do not poison recovery. */
TEST_F (testFilterAsyncOutput, invalidResultMetadata_n)
{
  ASSERT_NO_FATAL_FAILURE (start (true));
  GstTensorsInfo info;
  _async_result_info (&info, 2, 1);
  info.info[0].type = _NNS_END;
  GstTensorMemory output[2] = { { g_malloc0 (1), 1 }, { g_malloc0 (1), 1 } };
  nnstreamer_filter_dispatch_output_async_with_info (
      async_output_subplugin::pending, output, &info);
  gst_tensors_info_free (&info);
  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 50 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);
  async_output_subplugin::dispatch (0x5a);
  sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
  EXPECT_NO_FATAL_FAILURE (expectValue (sample, 0x5a));
  if (sample)
    gst_sample_unref (sample);
}

/** @brief Missing result memory frees every other entry using the supplied count. */
TEST_F (testFilterAsyncOutput, missingResultData_n)
{
  ASSERT_NO_FATAL_FAILURE (start (true));
  GstTensorsInfo info;
  _async_result_info (&info, 2, 1);
  GstTensorMemory output[2] = { {}, { g_malloc0 (1), 1 } };
  nnstreamer_filter_dispatch_output_async_with_info (
      async_output_subplugin::pending, output, &info);
  EXPECT_EQ (output[0].data, nullptr);
  EXPECT_EQ (output[1].data, nullptr);
  gst_tensors_info_free (&info);
  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 50 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);
}

/** @brief With no consumer, the supplied count controls ownership release. */
TEST_F (testFilterAsyncOutput, noCallbackResult_n)
{
  GstTensorFilterProperties prop = {};
  prop.output_meta.num_tensors = 1;
  GstTensorsInfo info;
  _async_result_info (&info, 2, 1);
  GstTensorMemory output[2] = { { g_malloc0 (1), 1 }, { g_malloc0 (1), 1 } };
  nnstreamer_filter_dispatch_output_async_with_info (&prop, output, &info);
  EXPECT_EQ (output[0].data, nullptr);
  EXPECT_EQ (output[1].data, nullptr);
  gst_tensors_info_free (&info);
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

/** @brief Flexible non-dynamic output uses its own tensor description. */
TEST_F (testFilterAsyncOutput, nonDynamicFlexibleOutput)
{
  ASSERT_NO_FATAL_FAILURE (start (true, false, 0, false));
  ASSERT_FALSE (async_output_subplugin::pending->invoke_dynamic);
  GstTensorsInfo info;
  _async_result_info (&info, 1, 1);
  guint8 value = 0x68;
  GstTensorMemory output = { _g_memdup (&value, 1), 1 };
  nnstreamer_filter_dispatch_output_async_with_info (
      async_output_subplugin::pending, &output, &info);
  gst_tensors_info_free (&info);
  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
  EXPECT_NO_FATAL_FAILURE (expectValue (sample, value));
  if (sample)
    gst_sample_unref (sample);
}

/** @brief Reject invalid first metadata, free every result and recover. */
TEST_F (testFilterAsyncOutput, nonDynamicFlexibleInvalidMetadata_n)
{
  ASSERT_NO_FATAL_FAILURE (start (true, false, 0, false));
  ASSERT_FALSE (async_output_subplugin::pending->invoke_dynamic);
  GstTensorsInfo info;
  _async_result_info (&info, 2, 1);
  info.info[0].type = _NNS_END;
  GstTensorMemory output[2] = { { g_malloc0 (1), 1 }, { g_malloc0 (1), 1 } };
  nnstreamer_filter_dispatch_output_async_with_info (
      async_output_subplugin::pending, output, &info);
  gst_tensors_info_free (&info);
  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 50 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);
  async_output_subplugin::dispatch (0x5a);
  sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
  EXPECT_NO_FATAL_FAILURE (expectValue (sample, 0x5a));
  if (sample)
    gst_sample_unref (sample);
}

/** @brief Reject later invalid metadata after an earlier tensor was prepared. */
TEST_F (testFilterAsyncOutput, nonDynamicFlexibleInvalidLaterMetadata_n)
{
  ASSERT_NO_FATAL_FAILURE (start (true, false, 0, false));
  ASSERT_FALSE (async_output_subplugin::pending->invoke_dynamic);
  GstTensorsInfo info;
  _async_result_info (&info, 2, 1);
  info.info[1].type = _NNS_END;
  GstTensorMemory output[2] = { { g_malloc0 (1), 1 }, { g_malloc0 (1), 1 } };
  nnstreamer_filter_dispatch_output_async_with_info (
      async_output_subplugin::pending, output, &info);
  gst_tensors_info_free (&info);
  GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 50 * GST_MSECOND);
  EXPECT_EQ (sample, nullptr);
  if (sample)
    gst_sample_unref (sample);
  async_output_subplugin::dispatch (0x5a);
  sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), GST_SECOND);
  EXPECT_NO_FATAL_FAILURE (expectValue (sample, 0x5a));
  if (sample)
    gst_sample_unref (sample);
}
