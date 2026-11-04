/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file	unittest_filter_tizen_hal.cc
 * @date	05 November 2026
 * @brief	Unit test for the Tizen HAL tensor_filter sub-plugins on a mock HAL
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <errno.h>
#include <glib.h>
#include <gst/gst.h>
#include <string.h>

#include <hal-ml.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>
#include <unittest_util.h>

/**
 * @brief The state of the mock HAL.
 */
static struct {
  int event_ret; /**< the result of an event_handler request */
  guint param_creates; /**< number of hal_ml_param_create calls */
  guint event_requests; /**< number of event_handler requests */
  int last_ops; /**< the ops of the last event_handler request */
} hal_mock;

/**
 * @brief The parameters of a mock request.
 */
typedef struct {
  int ops; /**< the "ops" parameter, -1 if not set */
} hal_mock_param_s;

static int hal_mock_backend; /**< what every mock backend handle points to */

/** @brief Mock: create a backend handle. */
int
hal_ml_create (const char *backend_name, hal_ml_h *handle)
{
  if (!backend_name || !handle)
    return HAL_ML_ERROR_INVALID_PARAMETER;

  *handle = &hal_mock_backend;
  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: destroy a backend handle. */
int
hal_ml_destroy (hal_ml_h handle)
{
  return (handle == &hal_mock_backend) ? HAL_ML_ERROR_NONE : HAL_ML_ERROR_INVALID_PARAMETER;
}

/** @brief Mock: create the parameters of a request. */
int
hal_ml_param_create (hal_ml_param_h *param)
{
  hal_mock_param_s *p = g_new0 (hal_mock_param_s, 1);

  p->ops = -1;
  hal_mock.param_creates++;
  *param = p;
  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: destroy the parameters of a request. */
int
hal_ml_param_destroy (hal_ml_param_h param)
{
  g_free (param);
  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: set a parameter of a request; only "ops" is kept. */
int
hal_ml_param_set (hal_ml_param_h param, const char *key, void *value)
{
  hal_mock_param_s *p = (hal_mock_param_s *) param;

  if (!p || !key || !value)
    return HAL_ML_ERROR_INVALID_PARAMETER;

  if (g_str_equal (key, "ops"))
    p->ops = *(event_ops *) value;
  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: send a request; event_handler is recorded and answered as told. */
int
hal_ml_request (hal_ml_h handle, const char *request_name, hal_ml_param_h param)
{
  hal_mock_param_s *p = (hal_mock_param_s *) param;

  if (handle != &hal_mock_backend || !request_name || !p)
    return HAL_ML_ERROR_INVALID_PARAMETER;

  if (g_str_equal (request_name, "event_handler")) {
    hal_mock.event_requests++;
    hal_mock.last_ops = p->ops;
    return hal_mock.event_ret;
  }

  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: invoke the model. */
int
hal_ml_request_invoke (hal_ml_h handle, const void *input, void *output)
{
  UNUSED (input);
  UNUSED (output);
  return (handle == &hal_mock_backend) ? HAL_ML_ERROR_NONE : HAL_ML_ERROR_INVALID_PARAMETER;
}

/** @brief The Tizen HAL sub-plugins built into this test. */
static const char *hal_subplugins[] = { "tizen-hal", "snpe", "vivante" };

/**
 * @brief Test fixture resetting the mock HAL.
 */
class tizenHalMock : public ::testing::Test
{
  protected:
  GstTensorFilterProperties prop;
  const gchar *models[1] = { "mock.model" };

  /** @brief reset the mock and the properties */
  void SetUp () override
  {
    memset (&hal_mock, 0, sizeof (hal_mock));
    hal_mock.last_ops = -1;
    memset (&prop, 0, sizeof (prop));
    prop.model_files = models;
    prop.num_models = 1;
    prop.custom_properties = "backend:mock";
  }
};

/**
 * @brief Before open, the is-updatable probe is answered by the empty object
 *        without a request to the HAL backend.
 */
TEST_F (tizenHalMock, probeBeforeOpen)
{
  for (const char *name : hal_subplugins) {
    const GstTensorFilterFramework *fw = nnstreamer_filter_find (name);

    SCOPED_TRACE (name);
    ASSERT_TRUE (fw != NULL);
    prop.fwname = name;
    hal_mock.param_creates = hal_mock.event_requests = 0;

    EXPECT_EQ (fw->eventHandler (fw, &prop, NULL, RELOAD_MODEL, NULL), -EINVAL);
    EXPECT_EQ (hal_mock.param_creates, 0U);
    EXPECT_EQ (hal_mock.event_requests, 0U);
  }
}

/**
 * @brief is-updatable set before start stays as set, without a request to
 *        the HAL backend.
 */
TEST_F (tizenHalMock, isUpdatableBeforeOpen)
{
  for (const char *name : hal_subplugins) {
    GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);
    gboolean updatable = FALSE;

    SCOPED_TRACE (name);
    ASSERT_TRUE (filter != nullptr);
    g_object_set (filter, "framework", name, NULL);
    hal_mock.param_creates = hal_mock.event_requests = 0;

    g_object_set (filter, "is-updatable", TRUE, NULL);
    g_object_get (filter, "is-updatable", &updatable, NULL);
    EXPECT_TRUE (updatable);
    EXPECT_EQ (hal_mock.param_creates, 0U);
    EXPECT_EQ (hal_mock.event_requests, 0U);

    gst_object_unref (filter);
  }
}

/**
 * @brief An opened instance forwards an event to the HAL backend.
 */
TEST_F (tizenHalMock, eventAfterOpen)
{
  for (const char *name : hal_subplugins) {
    const GstTensorFilterFramework *fw = nnstreamer_filter_find (name);
    GstTensorFilterFrameworkEventData data;
    void *pd = NULL;

    SCOPED_TRACE (name);
    ASSERT_TRUE (fw != NULL);
    prop.fwname = name;
    ASSERT_EQ (fw->open (&prop, &pd), 0);
    hal_mock.event_requests = 0;

    memset (&data, 0, sizeof (data));
    data.model_files = models;
    data.num_models = 1;
    EXPECT_EQ (fw->eventHandler (fw, &prop, pd, RELOAD_MODEL, &data), 0);
    EXPECT_EQ (hal_mock.event_requests, 1U);
    EXPECT_EQ (hal_mock.last_ops, (int) RELOAD_MODEL);

    fw->close (&prop, &pd);
  }
}

/**
 * @brief An opened instance answers -ENOENT for an event its HAL backend
 *        does not support.
 */
TEST_F (tizenHalMock, eventAfterOpenUnsupported_n)
{
  hal_mock.event_ret = HAL_ML_ERROR_NOT_SUPPORTED;

  for (const char *name : hal_subplugins) {
    const GstTensorFilterFramework *fw = nnstreamer_filter_find (name);
    void *pd = NULL;

    SCOPED_TRACE (name);
    ASSERT_TRUE (fw != NULL);
    prop.fwname = name;
    ASSERT_EQ (fw->open (&prop, &pd), 0);
    hal_mock.event_requests = 0;

    EXPECT_EQ (fw->eventHandler (fw, &prop, pd, RELOAD_MODEL, NULL), -ENOENT);
    EXPECT_EQ (hal_mock.event_requests, 1U);

    fw->close (&prop, &pd);
  }
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

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
