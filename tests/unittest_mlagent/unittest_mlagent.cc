/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_mlagent.cc
 * @date    30 Nov 2023
 * @brief   Unit test for MLAgent URI parsing
 * @author  Wook Song <wook.song16@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>

#include <iostream>

#ifdef __cplusplus
extern "C" {
#include "ml_agent.h"
}
#endif //__cplusplus

#include "mock_mlagent.h"

static const std::vector<MockModel> default_models{
  MockModel{ "MobileNet_v1", "/tmp/mobilenet_v1_0.tflite", "", false, "", 0U },
  MockModel{ "MobileNet_v1", "/tmp/mobilenet_v1_1.tflite", "", false, "", 1U },
  MockModel{ "ResNet50_v1", "/tmp/resnet50_v1_0.tflite", "", false, "", 0U },
  MockModel{ "ResNet50_v1", "/tmp/resnet50_v1_1.tflite", "", false, "", 1U },
};

/**
 * @brief Initialize the MockMLAgent using given MockModels
 * @param[in] models A vector containg MockModels
 */
void
_init (const std::vector<MockModel> &models = default_models)
{
  ml_agent_mock_init ();

  for (auto iter = models.begin (); iter != models.end (); ++iter) {
    ml_agent_mock_add_model (iter->name ().c_str (), iter->path ().c_str (),
        iter->app_info ().c_str (), iter->is_activated (),
        iter->desc ().c_str (), iter->version ());
  }
}

constexpr gchar valid_uri_format_literal[] = "mlagent://model/%s/%u";
constexpr gchar invalid_uri_format_literal[] = "ml-agent://model/%s/%u";

/**
 * @brief tests of getting model paths with valid URIs
 */
TEST (testMLAgent, GetModelValidURIs_p)
{
  _init ();

  // Test the valid URI cases
  GValue val = G_VALUE_INIT;
  g_value_init (&val, G_TYPE_STRING);
  const std::vector<MockModel> &models = default_models;

  for (auto iter = models.begin (); iter != models.end (); ++iter) {
    g_autofree gchar *uri = g_strdup_printf (
        valid_uri_format_literal, iter->name ().c_str (), iter->version ());
    g_autofree gchar *path = NULL;

    g_value_set_string (&val, uri);

    path = mlagent_get_model_path_from (&val);

    EXPECT_STREQ (path, iter->path ().c_str ());

    g_value_reset (&val);
  }
}

/**
 * @brief tests of getting model paths using URIs with invalid format
 */
TEST (testMLAgent, GetModelInvalidURIFormats_n)
{
  GValue val = G_VALUE_INIT;
  g_value_init (&val, G_TYPE_STRING);
  const std::vector<MockModel> &models = default_models;

  for (auto iter = models.begin (); iter != models.end (); ++iter) {
    g_autofree gchar *uri = g_strdup_printf (
        invalid_uri_format_literal, iter->name ().c_str (), iter->version ());
    g_autofree gchar *path = NULL;

    g_value_set_string (&val, uri);

    path = mlagent_get_model_path_from (&val);

    /**
     * In the case that invalid URIs are given, mlagent_get_model_path_from () returns
     * the given URI as it is so that it is handled by the fallback procedure (i.e., regarding it as a file path).
     */
    EXPECT_STREQ (uri, path);

    g_value_reset (&val);
  }
}

/**
 * @brief tests of getting model paths with invalid URIs
 */
TEST (testMLAgent, GetModelInvalidModel_n)
{
  // Clear the MLAgentMock instance
  _init ();

  // Test the valid URIs
  GValue val = G_VALUE_INIT;
  g_value_init (&val, G_TYPE_STRING);

  g_autofree gchar *uri
      = g_strdup_printf (valid_uri_format_literal, "InvalidModelName", UINT32_MAX);
  g_autofree gchar *path = NULL;

  g_value_set_string (&val, uri);

  path = mlagent_get_model_path_from (&val);

  /**
   * In the case that invalid URIs are given, mlagent_get_model_path_from () returns
   * the given URI as it is so that it is handled by the fallback procedure (i.e., regarding it as a file path).
   */
  EXPECT_STREQ (uri, path);

  g_value_reset (&val);
}

/**
 * @brief Main function for this unit test
 */
int
main (int argc, char *argv[])
{
  int ret = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    ret = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return ret;
}
