/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_edge.cc
 * @date        21 Jul 2022
 * @brief       Unit test for NNStreamer edge element
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Yechan Choi <yechan9.choi@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>

/** @todo Add edgesrc and edgesink unit test */
/**
 * @brief Test for edgesrc get and set properties
 */
TEST (Edge, sourceProperties0)
{
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
  } catch ( ...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch ( ...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
