/**
 * @file        unittest_watchdog.cc
 * @date        31 Oct 2024
 * @brief       Unit test for watchdog commonm uitil.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include "../gst/nnstreamer/nnstreamer_watchdog.h"

/**
 * @brief Test for watchdog creation.
 */
TEST (NnstWatchdog, create)
{
  nns_watchdog_h watchdog_h = NULL;
  gboolean ret = FALSE;

  ret = nnstreamer_watchdog_create (&watchdog_h);
  EXPECT_EQ (TRUE, ret);

  nnstreamer_watchdog_destroy (watchdog_h);
}

/**
 * @brief Called when watchdog is triggered.
 */
static gboolean
_watchdog_trigger (gpointer ptr)
{
  guint *received = (guint *) ptr;

  if (received)
    (*received)++;

  /** Trigger 10 times */
  return (*received) != 10;
}

/**
 * @brief Test for feeding watchdog.
 */
TEST (NnstWatchdog, feed)
{
  nns_watchdog_h watchdog_h = nullptr;
  gboolean ret = FALSE;
  guint *received = (guint *) g_malloc0 (sizeof (guint));
  const guint interval_ms = 50;

  ASSERT_NE (nullptr, received);

  ret = nnstreamer_watchdog_create (&watchdog_h);
  EXPECT_EQ (TRUE, ret);

  ret = nnstreamer_watchdog_feed (watchdog_h, _watchdog_trigger, interval_ms, received);
  EXPECT_EQ (TRUE, ret);

  g_usleep (1000000);
  EXPECT_EQ (10U, *received);

  nnstreamer_watchdog_destroy (watchdog_h);
  g_free (received);
}

/**
 * @brief Test for watchdog creation with invalid param.
 */
TEST (NnstWatchdog, create_n)
{
  gboolean ret = FALSE;

  ret = nnstreamer_watchdog_create (NULL);
  EXPECT_EQ (FALSE, ret);
}

/**
 * @brief Test for deeding watchdog with invalid param.
 */
TEST (NnstWatchdog, feed_1_n)
{
  nns_watchdog_h watchdog_h = nullptr;
  gboolean ret = FALSE;
  const guint interval_ms = 50;

  ret = nnstreamer_watchdog_create (&watchdog_h);
  EXPECT_EQ (TRUE, ret);

  ret = nnstreamer_watchdog_feed (NULL, _watchdog_trigger, interval_ms, NULL);
  EXPECT_EQ (FALSE, ret);

  nnstreamer_watchdog_destroy (watchdog_h);
}

/**
 * @brief Test for feeding watchdog with invalid param.
 */
TEST (NnstWatchdog, feed_2_n)
{
  nns_watchdog_h watchdog_h = nullptr;
  gboolean ret = FALSE;
  const guint interval_ms = 50;

  ret = nnstreamer_watchdog_create (&watchdog_h);
  EXPECT_EQ (TRUE, ret);

  ret = nnstreamer_watchdog_feed (watchdog_h, NULL, interval_ms, NULL);
  EXPECT_EQ (FALSE, ret);

  nnstreamer_watchdog_destroy (watchdog_h);
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

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
