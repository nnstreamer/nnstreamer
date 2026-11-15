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
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <string.h>

#include <hal-ml.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <unittest_util.h>

#define HAL_MOCK_MAGIC (0x4841u) /**< marks a live mock backend handle */

/**
 * @brief A backend handle of the mock HAL.
 */
typedef struct {
  guint magic; /**< HAL_MOCK_MAGIC while the handle is alive */
  gchar name[32]; /**< the backend name given to hal_ml_create */
} hal_mock_backend_s;

/**
 * @brief The parameters of a mock request.
 */
typedef struct {
  int ops; /**< the "ops" parameter, -1 if not set */
  const GstTensorFilterProperties *properties; /**< the "properties" parameter */
  GstTensorFilterFrameworkInfo *framework_info; /**< the "framework_info" parameter */
  GstTensorsInfo *in_info; /**< the "in_info" parameter */
  GstTensorsInfo *out_info; /**< the "out_info" parameter */
  GstTensorFilterFrameworkEventData *data; /**< the "data" parameter */
} hal_mock_param_s;

/**
 * @brief The state of the mock HAL: knobs set by a case, and records of the calls.
 */
static struct {
  int create_ret; /**< result of hal_ml_create */
  int param_create_ret; /**< result of hal_ml_param_create */
  const char *fail_key; /**< hal_ml_param_set of this key returns param_set_ret */
  int param_set_ret; /**< see fail_key */
  const char *fail_request; /**< hal_ml_request of this name returns request_ret */
  int request_ret; /**< see fail_request */
  int invoke_ret; /**< result of hal_ml_request_invoke */

  guint create_calls; /**< number of hal_ml_create calls */
  guint creates; /**< number of handles created */
  guint destroys; /**< number of handles destroyed */
  guint param_creates; /**< number of hal_ml_param_create calls that succeeded */
  guint param_destroys; /**< number of parameters destroyed */
  guint configure_requests; /**< number of configure_instance requests */
  guint info_requests; /**< number of get_framework_info requests */
  guint model_requests; /**< number of get_model_info requests */
  guint event_requests; /**< number of event_handler requests */
  guint invokes; /**< number of hal_ml_request_invoke calls */
  int last_ops; /**< the ops of the last model or event request */
  gchar last_backend[32]; /**< the backend name of the last created handle */
  hal_ml_h last_created; /**< the last created handle */
  hal_ml_h last_handle; /**< the handle of the last request or invoke */
  const void *last_properties; /**< the properties of the last configure_instance */
  const void *last_input; /**< the input of the last invoke */
  const void *last_output; /**< the output of the last invoke */
} hal_mock;

/**
 * @brief Reset the knobs and the records of the mock HAL.
 */
static void
hal_mock_reset (void)
{
  memset (&hal_mock, 0, sizeof (hal_mock));
  hal_mock.last_ops = -1;
}

/**
 * @brief Return the backend of a live mock handle, NULL if it is not one.
 */
static hal_mock_backend_s *
hal_mock_backend (hal_ml_h handle)
{
  hal_mock_backend_s *b = (hal_mock_backend_s *) handle;

  return (b && b->magic == HAL_MOCK_MAGIC) ? b : NULL;
}

/** @brief Mock: create a backend handle. */
int
hal_ml_create (const char *backend_name, hal_ml_h *handle)
{
  hal_mock_backend_s *b;

  hal_mock.create_calls++;
  if (!backend_name || !handle)
    return HAL_ML_ERROR_INVALID_PARAMETER;
  if (hal_mock.create_ret != HAL_ML_ERROR_NONE)
    return hal_mock.create_ret;

  b = g_new0 (hal_mock_backend_s, 1);
  b->magic = HAL_MOCK_MAGIC;
  g_strlcpy (b->name, backend_name, sizeof (b->name));
  g_strlcpy (hal_mock.last_backend, backend_name, sizeof (hal_mock.last_backend));
  hal_mock.creates++;
  hal_mock.last_created = b;
  *handle = b;
  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: destroy a backend handle. */
int
hal_ml_destroy (hal_ml_h handle)
{
  hal_mock_backend_s *b = hal_mock_backend (handle);

  if (!b)
    return HAL_ML_ERROR_INVALID_PARAMETER;

  b->magic = 0;
  g_free (b);
  hal_mock.destroys++;
  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: create the parameters of a request. */
int
hal_ml_param_create (hal_ml_param_h *param)
{
  hal_mock_param_s *p;

  if (!param)
    return HAL_ML_ERROR_INVALID_PARAMETER;
  if (hal_mock.param_create_ret != HAL_ML_ERROR_NONE)
    return hal_mock.param_create_ret;

  p = g_new0 (hal_mock_param_s, 1);
  p->ops = -1;
  hal_mock.param_creates++;
  *param = p;
  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: destroy the parameters of a request. */
int
hal_ml_param_destroy (hal_ml_param_h param)
{
  if (!param)
    return HAL_ML_ERROR_INVALID_PARAMETER;

  g_free (param);
  hal_mock.param_destroys++;
  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: set a parameter of a request. */
int
hal_ml_param_set (hal_ml_param_h param, const char *key, void *value)
{
  hal_mock_param_s *p = (hal_mock_param_s *) param;

  if (!p || !key || !value)
    return HAL_ML_ERROR_INVALID_PARAMETER;
  if (hal_mock.fail_key && g_str_equal (key, hal_mock.fail_key))
    return hal_mock.param_set_ret;

  if (g_str_equal (key, "ops"))
    p->ops = *(int *) value;
  else if (g_str_equal (key, "properties"))
    p->properties = (const GstTensorFilterProperties *) value;
  else if (g_str_equal (key, "framework_info"))
    p->framework_info = (GstTensorFilterFrameworkInfo *) value;
  else if (g_str_equal (key, "in_info"))
    p->in_info = (GstTensorsInfo *) value;
  else if (g_str_equal (key, "out_info"))
    p->out_info = (GstTensorsInfo *) value;
  else if (g_str_equal (key, "data"))
    p->data = (GstTensorFilterFrameworkEventData *) value;
  else
    return HAL_ML_ERROR_INVALID_PARAMETER;

  return HAL_ML_ERROR_NONE;
}

/**
 * @brief Fill a tensors info with one uint8 tensor of 3:4.
 */
static void
hal_mock_fill_info (GstTensorsInfo *info)
{
  gst_tensors_info_init (info);
  info->num_tensors = 1;
  info->info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:4", info->info[0].dimension);
}

/**
 * @brief Mock: send a request to the backend.
 *        get_framework_info sets allow_in_place as a mark of the backend;
 *        get_model_info answers GET_IN_OUT_INFO only.
 */
int
hal_ml_request (hal_ml_h handle, const char *request_name, hal_ml_param_h param)
{
  hal_mock_param_s *p = (hal_mock_param_s *) param;

  if (!hal_mock_backend (handle) || !request_name || !p)
    return HAL_ML_ERROR_INVALID_PARAMETER;

  hal_mock.last_handle = handle;

  if (g_str_equal (request_name, "configure_instance")) {
    hal_mock.configure_requests++;
    hal_mock.last_properties = p->properties;
  } else if (g_str_equal (request_name, "get_framework_info")) {
    hal_mock.info_requests++;
  } else if (g_str_equal (request_name, "get_model_info")) {
    hal_mock.model_requests++;
    hal_mock.last_ops = p->ops;
  } else if (g_str_equal (request_name, "event_handler")) {
    hal_mock.event_requests++;
    hal_mock.last_ops = p->ops;
  } else {
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  if (hal_mock.fail_request && g_str_equal (request_name, hal_mock.fail_request))
    return hal_mock.request_ret;

  if (g_str_equal (request_name, "configure_instance")) {
    if (!p->properties)
      return HAL_ML_ERROR_INVALID_PARAMETER;
  } else if (g_str_equal (request_name, "get_framework_info")) {
    if (!p->framework_info)
      return HAL_ML_ERROR_INVALID_PARAMETER;
    p->framework_info->allow_in_place = 1;
  } else if (g_str_equal (request_name, "get_model_info")) {
    if (!p->in_info || !p->out_info)
      return HAL_ML_ERROR_INVALID_PARAMETER;
    if (p->ops != GET_IN_OUT_INFO)
      return HAL_ML_ERROR_NOT_SUPPORTED;
    hal_mock_fill_info (p->in_info);
    hal_mock_fill_info (p->out_info);
  } else if (!p->data) {
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  return HAL_ML_ERROR_NONE;
}

/** @brief Mock: invoke the model; the output is the bitwise NOT of the input. */
int
hal_ml_request_invoke (hal_ml_h handle, const void *input, void *output)
{
  const GstTensorMemory *in = (const GstTensorMemory *) input;
  GstTensorMemory *out = (GstTensorMemory *) output;

  if (!hal_mock_backend (handle) || !in || !out)
    return HAL_ML_ERROR_INVALID_PARAMETER;

  hal_mock.invokes++;
  hal_mock.last_handle = handle;
  hal_mock.last_input = input;
  hal_mock.last_output = output;
  if (hal_mock.invoke_ret != HAL_ML_ERROR_NONE)
    return hal_mock.invoke_ret;

  for (gsize i = 0; i < MIN (in[0].size, out[0].size); i++)
    ((guint8 *) out[0].data)[i] = (guint8) ~((const guint8 *) in[0].data)[i];

  return HAL_ML_ERROR_NONE;
}

/**
 * @brief A Tizen HAL sub-plugin built into this test, with the backend it creates.
 */
typedef struct {
  const char *name; /**< the framework name */
  const char *backend; /**< the backend name given to hal_ml_create */
} hal_subplugin_s;

/** @brief The Tizen HAL sub-plugins built into this test. */
static const hal_subplugin_s hal_subplugins[] = {
  { "tizen-hal", "mock" },
  { "snpe", "snpe" },
  { "vivante", "vivante" },
};

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
    hal_mock_reset ();
    memset (&prop, 0, sizeof (prop));
    prop.model_files = models;
    prop.num_models = 1;
    prop.custom_properties = "backend:mock";
  }

  /** @brief clear the knobs, so the sub-plugins unregister cleanly at exit */
  void TearDown () override
  {
    hal_mock_reset ();
  }

  /** @brief find the framework of a sub-plugin and set it to prop */
  const GstTensorFilterFramework *findFw (const char *name)
  {
    const GstTensorFilterFramework *fw = nnstreamer_filter_find (name);

    prop.fwname = name;
    return fw;
  }

  /** @brief the number of request parameters not destroyed */
  static guint liveParams ()
  {
    return hal_mock.param_creates - hal_mock.param_destroys;
  }
};

/**
 * @brief Before open, the is-updatable probe is answered by the empty object
 *        without a request to the HAL backend.
 */
TEST_F (tizenHalMock, probeBeforeOpen)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    ASSERT_TRUE (fw != NULL);
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
  for (const hal_subplugin_s &sp : hal_subplugins) {
    GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);
    gboolean updatable = FALSE;

    SCOPED_TRACE (sp.name);
    ASSERT_TRUE (filter != nullptr);
    g_object_set (filter, "framework", sp.name, NULL);
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
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    GstTensorFilterFrameworkEventData data;
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();
    ASSERT_EQ (fw->open (&prop, &pd), 0);

    memset (&data, 0, sizeof (data));
    data.model_files = models;
    data.num_models = 1;
    EXPECT_EQ (fw->eventHandler (fw, &prop, pd, RELOAD_MODEL, &data), 0);
    EXPECT_EQ (hal_mock.event_requests, 1U);
    EXPECT_EQ (hal_mock.last_ops, (int) RELOAD_MODEL);
    EXPECT_EQ (liveParams (), 0U);

    fw->close (&prop, &pd);
  }
}

/**
 * @brief An opened instance answers -ENOENT for an event its HAL backend
 *        does not support.
 */
TEST_F (tizenHalMock, eventAfterOpenUnsupported_n)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();
    ASSERT_EQ (fw->open (&prop, &pd), 0);
    hal_mock.fail_request = "event_handler";
    hal_mock.request_ret = HAL_ML_ERROR_NOT_SUPPORTED;

    EXPECT_EQ (fw->eventHandler (fw, &prop, pd, RELOAD_MODEL, NULL), -ENOENT);
    EXPECT_EQ (hal_mock.event_requests, 1U);

    hal_mock.fail_request = NULL;
    fw->close (&prop, &pd);
  }
}

/**
 * @brief An event the HAL backend fails, or whose request cannot be built,
 *        returns -1.
 */
TEST_F (tizenHalMock, eventFailures_n)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();
    ASSERT_EQ (fw->open (&prop, &pd), 0);

    hal_mock.fail_request = "event_handler";
    hal_mock.request_ret = HAL_ML_ERROR_RUNTIME_ERROR;
    EXPECT_EQ (fw->eventHandler (fw, &prop, pd, RELOAD_MODEL, NULL), -1);
    EXPECT_EQ (hal_mock.event_requests, 1U);
    hal_mock.fail_request = NULL;

    for (const char *key : { "ops", "data" }) {
      SCOPED_TRACE (key);
      hal_mock.fail_key = key;
      hal_mock.param_set_ret = HAL_ML_ERROR_INVALID_PARAMETER;
      EXPECT_EQ (fw->eventHandler (fw, &prop, pd, RELOAD_MODEL, NULL), -1);
    }
    hal_mock.fail_key = NULL;

    hal_mock.param_create_ret = HAL_ML_ERROR_RUNTIME_ERROR;
    EXPECT_EQ (fw->eventHandler (fw, &prop, pd, RELOAD_MODEL, NULL), -1);
    hal_mock.param_create_ret = HAL_ML_ERROR_NONE;

    EXPECT_EQ (hal_mock.event_requests, 1U);
    EXPECT_EQ (liveParams (), 0U);
    fw->close (&prop, &pd);
  }
}

/**
 * @brief Open creates a handle of the backend and configures it with the
 *        properties; close destroys it.
 */
TEST_F (tizenHalMock, openClose)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();

    ASSERT_EQ (fw->open (&prop, &pd), 0);
    EXPECT_TRUE (pd != NULL);
    EXPECT_EQ (hal_mock.creates, 1U);
    EXPECT_STREQ (hal_mock.last_backend, sp.backend);
    EXPECT_EQ (hal_mock.configure_requests, 1U);
    EXPECT_EQ (hal_mock.last_handle, hal_mock.last_created);
    EXPECT_EQ (hal_mock.last_properties, (const void *) &prop);
    EXPECT_EQ (liveParams (), 0U);
    EXPECT_EQ (hal_mock.destroys, 0U);

    fw->close (&prop, &pd);
    EXPECT_TRUE (pd == NULL);
    EXPECT_EQ (hal_mock.destroys, 1U);
  }
}

/**
 * @brief tizen-hal takes the backend from the custom properties, ignoring
 *        case and spaces of the key; the last one given wins.
 */
TEST_F (tizenHalMock, backendFromCustomProperties)
{
  const struct {
    const char *custom;
    const char *backend;
  } cases[] = {
    { " foo : bar , BACKEND : npu-x ,x:y", "npu-x" },
    { "backend:first,backend:second", "second" },
  };
  const GstTensorFilterFramework *fw = findFw ("tizen-hal");

  ASSERT_TRUE (fw != NULL);
  for (const auto &c : cases) {
    void *pd = NULL;

    SCOPED_TRACE (c.custom);
    hal_mock_reset ();
    prop.custom_properties = c.custom;

    ASSERT_EQ (fw->open (&prop, &pd), 0);
    EXPECT_EQ (hal_mock.creates, 1U);
    EXPECT_STREQ (hal_mock.last_backend, c.backend);
    fw->close (&prop, &pd);
    EXPECT_EQ (hal_mock.destroys, 1U);
  }
}

/**
 * @brief tizen-hal refuses to open without a backend in the custom properties,
 *        before creating a handle.
 */
TEST_F (tizenHalMock, openWithoutBackend_n)
{
  const char *customs[] = { NULL, "", "foo:bar", "backend", "back:mock" };
  const GstTensorFilterFramework *fw = findFw ("tizen-hal");

  ASSERT_TRUE (fw != NULL);
  for (const char *custom : customs) {
    void *pd = NULL;

    SCOPED_TRACE (custom ? custom : "(null)");
    hal_mock_reset ();
    prop.custom_properties = custom;

    EXPECT_EQ (fw->open (&prop, &pd), -EINVAL);
    EXPECT_TRUE (pd == NULL);
    EXPECT_EQ (hal_mock.create_calls, 0U);
  }
}

/**
 * @brief Open fails when the backend handle cannot be created, whatever the
 *        error of the HAL.
 */
TEST_F (tizenHalMock, openCreateFails_n)
{
  const int errors[] = { HAL_ML_ERROR_NOT_SUPPORTED, HAL_ML_ERROR_RUNTIME_ERROR,
    HAL_ML_ERROR_INVALID_PARAMETER, -EPERM };

  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);

    ASSERT_TRUE (fw != NULL);
    for (int error : errors) {
      void *pd = NULL;

      SCOPED_TRACE (error);
      hal_mock_reset ();
      hal_mock.create_ret = error;

      EXPECT_EQ (fw->open (&prop, &pd), -EINVAL);
      EXPECT_TRUE (pd == NULL);
      EXPECT_EQ (hal_mock.create_calls, 1U);
      EXPECT_EQ (hal_mock.configure_requests, 0U);
      EXPECT_EQ (hal_mock.param_creates, 0U);
    }
  }
}

/**
 * @brief Open fails when the configure request cannot be built or fails; the
 *        handle and the request parameters are released.
 */
TEST_F (tizenHalMock, openConfigureFails_n)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);

    hal_mock_reset ();
    hal_mock.param_create_ret = HAL_ML_ERROR_RUNTIME_ERROR;
    EXPECT_EQ (fw->open (&prop, &pd), -EINVAL);
    EXPECT_EQ (hal_mock.configure_requests, 0U);
    EXPECT_EQ (hal_mock.creates, 1U);
    EXPECT_EQ (hal_mock.destroys, 1U);

    hal_mock_reset ();
    hal_mock.fail_key = "properties";
    hal_mock.param_set_ret = HAL_ML_ERROR_INVALID_PARAMETER;
    EXPECT_EQ (fw->open (&prop, &pd), -EINVAL);
    EXPECT_EQ (hal_mock.param_creates, 1U);
    EXPECT_EQ (liveParams (), 0U);
    EXPECT_EQ (hal_mock.configure_requests, 0U);
    EXPECT_EQ (hal_mock.destroys, 1U);

    hal_mock_reset ();
    hal_mock.fail_request = "configure_instance";
    hal_mock.request_ret = HAL_ML_ERROR_RUNTIME_ERROR;
    EXPECT_EQ (fw->open (&prop, &pd), -EINVAL);
    EXPECT_EQ (hal_mock.configure_requests, 1U);
    EXPECT_EQ (liveParams (), 0U);
    EXPECT_EQ (hal_mock.destroys, 1U);

    EXPECT_TRUE (pd == NULL);
  }
}

/**
 * @brief Invoke hands the tensors to the backend handle of the instance.
 */
TEST_F (tizenHalMock, invoke)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    guint8 in_data[12], out_data[12];
    GstTensorMemory in = { in_data, sizeof (in_data) };
    GstTensorMemory out = { out_data, sizeof (out_data) };
    hal_ml_h handle;
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();
    ASSERT_EQ (fw->open (&prop, &pd), 0);
    handle = hal_mock.last_created;

    for (guint i = 0; i < sizeof (in_data); i++)
      in_data[i] = (guint8) (i * 7);
    memset (out_data, 0, sizeof (out_data));

    EXPECT_EQ (fw->invoke (fw, &prop, pd, &in, &out), 0);
    EXPECT_EQ (hal_mock.invokes, 1U);
    EXPECT_EQ (hal_mock.last_handle, handle);
    EXPECT_EQ (hal_mock.last_input, (const void *) &in);
    EXPECT_EQ (hal_mock.last_output, (const void *) &out);
    for (guint i = 0; i < sizeof (out_data); i++)
      EXPECT_EQ (out_data[i], (guint8) ~(i * 7));

    fw->close (&prop, &pd);
  }
}

/**
 * @brief Invoke fails without buffers, before reaching the backend, and when
 *        the backend fails.
 */
TEST_F (tizenHalMock, invokeFailures_n)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    guint8 data[4] = { 0 };
    GstTensorMemory mem = { data, sizeof (data) };
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();
    ASSERT_EQ (fw->open (&prop, &pd), 0);

    EXPECT_EQ (fw->invoke (fw, &prop, pd, NULL, &mem), -EINVAL);
    EXPECT_EQ (fw->invoke (fw, &prop, pd, &mem, NULL), -EINVAL);
    EXPECT_EQ (hal_mock.invokes, 0U);

    hal_mock.invoke_ret = HAL_ML_ERROR_RUNTIME_ERROR;
    EXPECT_EQ (fw->invoke (fw, &prop, pd, &mem, &mem), -EINVAL);
    EXPECT_EQ (hal_mock.invokes, 1U);
    hal_mock.invoke_ret = HAL_ML_ERROR_NONE;

    fw->close (&prop, &pd);
  }
}

/**
 * @brief An opened instance asks its backend for the framework info and
 *        reports the name of the sub-plugin.
 */
TEST_F (tizenHalMock, frameworkInfoAfterOpen)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    GstTensorFilterFrameworkInfo info;
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();
    ASSERT_EQ (fw->open (&prop, &pd), 0);

    memset (&info, 0, sizeof (info));
    EXPECT_EQ (fw->getFrameworkInfo (fw, &prop, pd, &info), 0);
    EXPECT_STREQ (info.name, sp.name);
    EXPECT_EQ (info.allow_in_place, 1);
    EXPECT_EQ (hal_mock.info_requests, 1U);
    EXPECT_EQ (hal_mock.last_handle, hal_mock.last_created);
    EXPECT_EQ (liveParams (), 0U);

    fw->close (&prop, &pd);
  }
}

/**
 * @brief Before open, tizen-hal has no backend to ask and reports its name
 *        only; snpe and vivante ask the backend of their empty object.
 */
TEST_F (tizenHalMock, frameworkInfoBeforeOpen)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    const gboolean asks = !g_str_equal (sp.name, "tizen-hal");
    GstTensorFilterFrameworkInfo info;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();

    memset (&info, 0, sizeof (info));
    EXPECT_EQ (fw->getFrameworkInfo (fw, &prop, NULL, &info), 0);
    EXPECT_STREQ (info.name, sp.name);
    EXPECT_EQ (info.allow_in_place, asks ? 1 : 0);
    EXPECT_EQ (hal_mock.info_requests, asks ? 1U : 0U);
    EXPECT_EQ (liveParams (), 0U);
  }
}

/**
 * @brief The framework info fails when its request cannot be built or fails.
 */
TEST_F (tizenHalMock, frameworkInfoFailures_n)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    GstTensorFilterFrameworkInfo info;
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();
    ASSERT_EQ (fw->open (&prop, &pd), 0);
    memset (&info, 0, sizeof (info));

    hal_mock.param_create_ret = HAL_ML_ERROR_RUNTIME_ERROR;
    EXPECT_EQ (fw->getFrameworkInfo (fw, &prop, pd, &info), -EINVAL);
    hal_mock.param_create_ret = HAL_ML_ERROR_NONE;

    hal_mock.fail_key = "framework_info";
    hal_mock.param_set_ret = HAL_ML_ERROR_INVALID_PARAMETER;
    EXPECT_EQ (fw->getFrameworkInfo (fw, &prop, pd, &info), -EINVAL);
    hal_mock.fail_key = NULL;
    EXPECT_EQ (hal_mock.info_requests, 0U);

    hal_mock.fail_request = "get_framework_info";
    hal_mock.request_ret = HAL_ML_ERROR_RUNTIME_ERROR;
    EXPECT_EQ (fw->getFrameworkInfo (fw, &prop, pd, &info), -EINVAL);
    EXPECT_EQ (hal_mock.info_requests, 1U);
    hal_mock.fail_request = NULL;

    EXPECT_EQ (info.allow_in_place, 0);
    EXPECT_EQ (liveParams (), 0U);
    fw->close (&prop, &pd);
  }
}

/**
 * @brief The model info comes from the backend of the instance.
 */
TEST_F (tizenHalMock, modelInfo)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    GstTensorsInfo in_info, out_info;
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();
    ASSERT_EQ (fw->open (&prop, &pd), 0);
    gst_tensors_info_init (&in_info);
    gst_tensors_info_init (&out_info);

    EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, GET_IN_OUT_INFO, &in_info, &out_info), 0);
    EXPECT_EQ (hal_mock.model_requests, 1U);
    EXPECT_EQ (hal_mock.last_ops, (int) GET_IN_OUT_INFO);
    EXPECT_EQ (hal_mock.last_handle, hal_mock.last_created);
    EXPECT_EQ (liveParams (), 0U);
    for (GstTensorsInfo *info : { &in_info, &out_info }) {
      EXPECT_EQ (info->num_tensors, 1U);
      EXPECT_EQ (info->info[0].type, _NNS_UINT8);
      EXPECT_EQ (info->info[0].dimension[0], 3U);
      EXPECT_EQ (info->info[0].dimension[1], 4U);
      EXPECT_EQ (info->info[0].dimension[2], 0U);
    }

    gst_tensors_info_free (&in_info);
    gst_tensors_info_free (&out_info);
    fw->close (&prop, &pd);
  }
}

/**
 * @brief The model info returns -ENOENT for what the backend does not
 *        support, and -1 for any other failure.
 */
TEST_F (tizenHalMock, modelInfoFailures_n)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    SCOPED_TRACE (sp.name);
    const GstTensorFilterFramework *fw = findFw (sp.name);
    GstTensorsInfo in_info, out_info;
    void *pd = NULL;

    ASSERT_TRUE (fw != NULL);
    hal_mock_reset ();
    ASSERT_EQ (fw->open (&prop, &pd), 0);
    gst_tensors_info_init (&in_info);
    gst_tensors_info_init (&out_info);

    EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, SET_INPUT_INFO, &in_info, &out_info), -ENOENT);
    EXPECT_EQ (hal_mock.last_ops, (int) SET_INPUT_INFO);
    EXPECT_EQ (hal_mock.model_requests, 1U);

    hal_mock.fail_request = "get_model_info";
    hal_mock.request_ret = HAL_ML_ERROR_RUNTIME_ERROR;
    EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, GET_IN_OUT_INFO, &in_info, &out_info), -1);
    EXPECT_EQ (hal_mock.model_requests, 2U);
    hal_mock.fail_request = NULL;

    for (const char *key : { "ops", "in_info", "out_info" }) {
      SCOPED_TRACE (key);
      hal_mock.fail_key = key;
      hal_mock.param_set_ret = HAL_ML_ERROR_INVALID_PARAMETER;
      EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, GET_IN_OUT_INFO, &in_info, &out_info), -1);
    }
    hal_mock.fail_key = NULL;

    hal_mock.param_create_ret = HAL_ML_ERROR_RUNTIME_ERROR;
    EXPECT_EQ (fw->getModelInfo (fw, &prop, pd, GET_IN_OUT_INFO, &in_info, &out_info), -1);
    hal_mock.param_create_ret = HAL_ML_ERROR_NONE;

    EXPECT_EQ (hal_mock.model_requests, 2U);
    EXPECT_EQ (in_info.num_tensors, 0U);
    EXPECT_EQ (liveParams (), 0U);
    fw->close (&prop, &pd);
  }
}

/**
 * @brief tensor_filter runs each sub-plugin in a stream: the caps come from
 *        the model info of the backend, the output from its invoke.
 */
TEST_F (tizenHalMock, pipeline)
{
  for (const hal_subplugin_s &sp : hal_subplugins) {
    GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);
    GstHarness *h;
    GstBuffer *in_buf, *out_buf;
    GstMapInfo map;
    guint8 values[12];

    SCOPED_TRACE (sp.name);
    ASSERT_TRUE (filter != nullptr);
    hal_mock_reset ();
    g_object_set (filter, "framework", sp.name, "model", "mock.model", "custom",
        "backend:mock", NULL);
    h = gst_harness_new_with_element (filter, "sink", "src");
    gst_object_unref (filter);
    gst_harness_set_src_caps_str (h, "other/tensors,num_tensors=1,format=static,"
                                     "dimensions=3:4:1:1,types=uint8,framerate=0/1");

    for (guint i = 0; i < sizeof (values); i++)
      values[i] = (guint8) (i + 100);
    in_buf = gst_harness_create_buffer (h, sizeof (values));
    gst_buffer_fill (in_buf, 0, values, sizeof (values));

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
    out_buf = gst_harness_try_pull (h);
    EXPECT_TRUE (out_buf != nullptr);
    if (out_buf) {
      EXPECT_TRUE (gst_buffer_map (out_buf, &map, GST_MAP_READ));
      EXPECT_EQ (map.size, sizeof (values));
      for (gsize i = 0; i < MIN (map.size, sizeof (values)); i++)
        EXPECT_EQ (map.data[i], (guint8) ~values[i]);
      gst_buffer_unmap (out_buf, &map);
      gst_buffer_unref (out_buf);
    }
    EXPECT_EQ (hal_mock.creates, 1U);
    EXPECT_EQ (hal_mock.invokes, 1U);

    gst_harness_teardown (h);
    EXPECT_EQ (hal_mock.destroys, 1U);
    EXPECT_EQ (liveParams (), 0U);
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
