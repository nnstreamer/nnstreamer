/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file	unittest_filter_event.cc
 * @date	17 September 2026
 * @brief	Unit test for the event data tensor_filter hands to a sub-plugin
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <errno.h>
#include <glib.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <string.h>

#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>
#include <unittest_util.h>

#include "../gst/nnstreamer/tensor_filter/tensor_filter_common.h"

/**
 * @brief C++ sub-plugin recording the event data it receives.
 */
class event_mock_subplugin : public nnstreamer::tensor_filter_subplugin
{
  public:
  static const char *mock_name;
  static event_mock_subplugin *registered;
  static int event_ret;
  static guint num_events[RESUME + 1];
  static tensors_layout last_layout;
  static void *last_data;
  static GstTensorFilterFrameworkEventData last_event;

  /** @brief mandatory method */
  tensor_filter_subplugin &getEmptyInstance () override
  {
    return *(new event_mock_subplugin ());
  }

  /** @brief mandatory method */
  void configure_instance (const GstTensorFilterProperties *prop) override
  {
    UNUSED (prop);
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

  /** @brief record the event and read the data it was given */
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data) override
  {
    EXPECT_LT ((guint) ops, G_N_ELEMENTS (num_events));
    if ((guint) ops < G_N_ELEMENTS (num_events))
      g_atomic_int_inc ((gint *) &num_events[ops]);

    if (ops == SET_INPUT_PROP || ops == SET_OUTPUT_PROP) {
      memcpy (last_layout, data.layout, sizeof (last_layout));
    } else {
      last_data = data.data;
      memcpy (&last_event, &data, sizeof (last_event));
    }

    return event_ret;
  }

  /** @brief register this mock subplugin */
  static void init ()
  {
    registered = register_subplugin<event_mock_subplugin> ();
  }

  /** @brief unregister this mock subplugin */
  static void fini ()
  {
    unregister_subplugin<event_mock_subplugin> (registered);
    registered = nullptr;
  }
};

const char *event_mock_subplugin::mock_name = "event_mock_subplugin";
event_mock_subplugin *event_mock_subplugin::registered = nullptr;
int event_mock_subplugin::event_ret = 0;
guint event_mock_subplugin::num_events[RESUME + 1];
tensors_layout event_mock_subplugin::last_layout;
void *event_mock_subplugin::last_data = nullptr;
GstTensorFilterFrameworkEventData event_mock_subplugin::last_event;

/**
 * @brief Fill the recorded event data with non-zero bytes.
 * @details A handler that does not overwrite it, or one that overwrites only
 *          the first member of the union, leaves this pattern behind.
 */
static void
_dirty_event_record (void)
{
  memset (&event_mock_subplugin::last_event, 0xA5, sizeof (event_mock_subplugin::last_event));
}

/**
 * @brief Tell whether every byte of the recorded event data is zero.
 */
static gboolean
_event_record_is_zeroed (void)
{
  GstTensorFilterFrameworkEventData zeroed;

  memset (&zeroed, 0, sizeof (zeroed));
  return memcmp (&event_mock_subplugin::last_event, &zeroed, sizeof (zeroed)) == 0;
}

/**
 * @brief Fill a large part of the stack with non-zero bytes.
 * @details Leaves the frames a following call reuses non-zero, so a field the
 *          caller does not initialize shows up as garbage instead of a lucky zero.
 */
static G_GNUC_NO_INLINE void
_dirty_stack (void)
{
  static void *(*volatile fill) (void *, int, size_t) = memset;
  guint8 junk[1 << 16];

  fill (junk, 0xA5, sizeof (junk));
}

/**
 * @brief Test fixture holding a tensor_filter private data opened with the mock.
 */
class testFilterEvent : public ::testing::Test
{
  protected:
  GstTensorFilterPrivate priv;

  /** @brief register the mock and open it the way tensor_filter does */
  void SetUp () override
  {
    event_mock_subplugin::event_ret = 0;
    memset (event_mock_subplugin::num_events, 0, sizeof (event_mock_subplugin::num_events));
    memset (event_mock_subplugin::last_layout, 0, sizeof (event_mock_subplugin::last_layout));
    event_mock_subplugin::last_data = &priv;
    _dirty_event_record ();
    event_mock_subplugin::init ();

    gst_tensor_filter_common_init_property (&priv);
    g_free ((gpointer) priv.prop.fwname);
    priv.prop.fwname = g_strdup (event_mock_subplugin::mock_name);
    priv.fw = nnstreamer_filter_find (event_mock_subplugin::mock_name);
    ASSERT_TRUE (priv.fw != NULL);
    ASSERT_TRUE (gst_tensor_filter_common_open_fw (&priv));
  }

  /** @brief close the mock and unregister it */
  void TearDown () override
  {
    event_mock_subplugin::event_ret = 0;
    gst_tensor_filter_common_close_fw (&priv);
    gst_tensor_filter_common_free_property (&priv);
    event_mock_subplugin::fini ();
  }

  /**
   * @brief Set a layout property of the configured filter from a dirty stack.
   * @param is_input TRUE for inputlayout, FALSE for outputlayout
   * @param layouts the property value
   */
  void setLayout (gboolean is_input, const gchar *layouts)
  {
    GValue value = G_VALUE_INIT;

    if (is_input)
      priv.prop.input_configured = TRUE;
    else
      priv.prop.output_configured = TRUE;

    g_value_init (&value, G_TYPE_STRING);
    g_value_set_string (&value, layouts);
    _dirty_stack ();
    EXPECT_TRUE (gst_tensor_filter_common_set_property (
        &priv, is_input ? PROP_INPUTLAYOUT : PROP_OUTPUTLAYOUT, &value, NULL));
    g_value_unset (&value);
  }
};

/**
 * @brief Changing the input layout of a configured filter hands the sub-plugin
 *        and stores only the given layouts; every other entry is ANY.
 */
TEST_F (testFilterEvent, setInputLayout)
{
  guint i;

  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++)
    priv.prop.input_layout[i] = _NNS_LAYOUT_NHWC;

  setLayout (TRUE, "NCHW");

  EXPECT_EQ (event_mock_subplugin::num_events[SET_INPUT_PROP], 1U);
  EXPECT_EQ (event_mock_subplugin::last_layout[0], _NNS_LAYOUT_NCHW);
  EXPECT_EQ (priv.prop.input_layout[0], _NNS_LAYOUT_NCHW);
  for (i = 1; i < NNS_TENSOR_SIZE_LIMIT; i++) {
    EXPECT_EQ (event_mock_subplugin::last_layout[i], _NNS_LAYOUT_ANY) << "entry " << i;
    EXPECT_EQ (priv.prop.input_layout[i], _NNS_LAYOUT_ANY) << "entry " << i;
  }
}

/**
 * @brief Changing the output layout of a configured filter hands the sub-plugin
 *        and stores only the given layouts; every other entry is ANY.
 */
TEST_F (testFilterEvent, setOutputLayout)
{
  guint i;

  setLayout (FALSE, "NHWC,NCHW");

  EXPECT_EQ (event_mock_subplugin::num_events[SET_OUTPUT_PROP], 1U);
  EXPECT_EQ (event_mock_subplugin::last_layout[0], _NNS_LAYOUT_NHWC);
  EXPECT_EQ (event_mock_subplugin::last_layout[1], _NNS_LAYOUT_NCHW);
  EXPECT_EQ (priv.prop.output_layout[0], _NNS_LAYOUT_NHWC);
  EXPECT_EQ (priv.prop.output_layout[1], _NNS_LAYOUT_NCHW);
  for (i = 2; i < NNS_TENSOR_SIZE_LIMIT; i++) {
    EXPECT_EQ (event_mock_subplugin::last_layout[i], _NNS_LAYOUT_ANY) << "entry " << i;
    EXPECT_EQ (priv.prop.output_layout[i], _NNS_LAYOUT_ANY) << "entry " << i;
  }
}

/**
 * @brief A layout the sub-plugin refuses leaves the stored layouts unchanged.
 */
TEST_F (testFilterEvent, setLayoutRefused_n)
{
  guint i;

  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++)
    priv.prop.input_layout[i] = _NNS_LAYOUT_NHWC;

  event_mock_subplugin::event_ret = -EINVAL;
  setLayout (TRUE, "NCHW");

  EXPECT_EQ (event_mock_subplugin::num_events[SET_INPUT_PROP], 1U);
  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++)
    EXPECT_EQ (priv.prop.input_layout[i], _NNS_LAYOUT_NHWC) << "entry " << i;
}

/**
 * @brief A C++ sub-plugin can read the event data of SUSPEND and RESUME.
 */
TEST_F (testFilterEvent, suspendResume)
{
  gst_tensor_filter_common_unload_fw (&priv, TRUE);

  EXPECT_EQ (event_mock_subplugin::num_events[SUSPEND], 1U);
  EXPECT_TRUE (event_mock_subplugin::last_data == nullptr);
  EXPECT_TRUE (_event_record_is_zeroed ());
  EXPECT_TRUE (priv.prop.fw_opened);
  EXPECT_TRUE (priv.is_suspended);

  event_mock_subplugin::last_data = &priv;
  _dirty_event_record ();
  EXPECT_TRUE (gst_tensor_filter_common_open_fw (&priv));

  EXPECT_EQ (event_mock_subplugin::num_events[RESUME], 1U);
  EXPECT_TRUE (event_mock_subplugin::last_data == nullptr);
  EXPECT_TRUE (_event_record_is_zeroed ());
  EXPECT_FALSE (priv.is_suspended);
}

/**
 * @brief A C++ sub-plugin not supporting SUSPEND is closed instead.
 */
TEST_F (testFilterEvent, suspendUnsupported_n)
{
  event_mock_subplugin::event_ret = -ENOENT;
  gst_tensor_filter_common_unload_fw (&priv, TRUE);

  EXPECT_EQ (event_mock_subplugin::num_events[SUSPEND], 1U);
  EXPECT_FALSE (priv.prop.fw_opened);
  EXPECT_FALSE (priv.is_suspended);
}

/**
 * @brief A C++ sub-plugin failing RESUME stays suspended.
 */
TEST_F (testFilterEvent, resumeFail_n)
{
  gst_tensor_filter_common_unload_fw (&priv, TRUE);
  ASSERT_TRUE (priv.is_suspended);

  event_mock_subplugin::event_ret = -EINVAL;
  EXPECT_FALSE (gst_tensor_filter_common_open_fw (&priv));

  EXPECT_EQ (event_mock_subplugin::num_events[RESUME], 1U);
  EXPECT_TRUE (priv.prop.fw_opened);
  EXPECT_TRUE (priv.is_suspended);
}

/**
 * @brief Count the critical logs of the default domain.
 */
static void
_count_critical_logs (const gchar *log_domain, GLogLevelFlags log_level,
    const gchar *message, gpointer user_data)
{
  UNUSED (log_domain);
  UNUSED (log_level);
  UNUSED (message);
  (*(guint *) user_data)++;
}

/**
 * @brief Unloading a suspended C++ sub-plugin with suspend keeps it suspended,
 *        so it is resumed, not reopened, afterwards.
 */
TEST_F (testFilterEvent, unloadSuspended)
{
  guint critical_count = 0;
  guint handler = g_log_set_handler (
      NULL, G_LOG_LEVEL_CRITICAL, _count_critical_logs, &critical_count);

  gst_tensor_filter_common_unload_fw (&priv, TRUE);
  gst_tensor_filter_common_unload_fw (&priv, TRUE);
  g_log_remove_handler (NULL, handler);

  EXPECT_EQ (critical_count, 0U);
  EXPECT_EQ (event_mock_subplugin::num_events[SUSPEND], 1U);
  EXPECT_TRUE (priv.prop.fw_opened);
  EXPECT_TRUE (priv.is_suspended);

  EXPECT_TRUE (gst_tensor_filter_common_open_fw (&priv));
  EXPECT_EQ (event_mock_subplugin::num_events[RESUME], 1U);
  EXPECT_FALSE (priv.is_suspended);
}

/**
 * @brief A suspended C++ sub-plugin failing RESUME stays suspended, and
 *        unloading it again with suspend still sends no SUSPEND.
 */
TEST_F (testFilterEvent, unloadSuspendedResumeFail_n)
{
  guint critical_count = 0;
  guint handler;

  gst_tensor_filter_common_unload_fw (&priv, TRUE);
  event_mock_subplugin::event_ret = -EINVAL;
  EXPECT_FALSE (gst_tensor_filter_common_open_fw (&priv));
  EXPECT_TRUE (priv.is_suspended);

  handler = g_log_set_handler (NULL, G_LOG_LEVEL_CRITICAL, _count_critical_logs, &critical_count);
  gst_tensor_filter_common_unload_fw (&priv, TRUE);
  g_log_remove_handler (NULL, handler);

  EXPECT_EQ (critical_count, 0U);
  EXPECT_EQ (event_mock_subplugin::num_events[SUSPEND], 1U);
  EXPECT_EQ (event_mock_subplugin::num_events[RESUME], 1U);
  EXPECT_TRUE (priv.prop.fw_opened);
  EXPECT_TRUE (priv.is_suspended);
}

/**
 * @brief Closing a suspended C++ sub-plugin clears the suspended state, so it
 *        can be opened again.
 */
TEST_F (testFilterEvent, closeSuspended)
{
  gst_tensor_filter_common_unload_fw (&priv, TRUE);
  ASSERT_TRUE (priv.is_suspended);

  gst_tensor_filter_common_unload_fw (&priv, FALSE);
  EXPECT_FALSE (priv.prop.fw_opened);
  EXPECT_FALSE (priv.is_suspended);

  EXPECT_TRUE (gst_tensor_filter_common_open_fw (&priv));
  EXPECT_TRUE (priv.prop.fw_opened);
  EXPECT_EQ (event_mock_subplugin::num_events[RESUME], 0U);
}

/**
 * @brief tensor_filter stopped after the suspend watchdog suspended its model
 *        starts again by resuming the model.
 */
TEST_F (testFilterEvent, restartAfterSuspend)
{
  GstElement *filter;
  gint *suspend_count = (gint *) &event_mock_subplugin::num_events[SUSPEND];
  guint critical_count = 0;
  guint handler, i;

  filter = gst_element_factory_make ("tensor_filter", NULL);
  ASSERT_TRUE (filter != nullptr);
  g_object_set (filter, "framework", event_mock_subplugin::mock_name, "suspend", 10, NULL);
  handler = g_log_set_handler (NULL, G_LOG_LEVEL_CRITICAL, _count_critical_logs, &critical_count);

  EXPECT_EQ (gst_element_set_state (filter, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);
  for (i = 0; i < 5000 && g_atomic_int_get (suspend_count) == 0; i++)
    g_usleep (1000);
  EXPECT_EQ (g_atomic_int_get (suspend_count), 1);

  EXPECT_EQ (gst_element_set_state (filter, GST_STATE_READY), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (gst_element_set_state (filter, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (event_mock_subplugin::num_events[RESUME], 1U);

  EXPECT_EQ (gst_element_set_state (filter, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
  g_log_remove_handler (NULL, handler);
  gst_object_unref (filter);
  EXPECT_EQ (critical_count, 0U);
}

/**
 * @brief Setting is-updatable asks a C++ sub-plugin with no event data.
 */
TEST_F (testFilterEvent, isUpdatable)
{
  GValue value = G_VALUE_INIT;

  g_value_init (&value, G_TYPE_BOOLEAN);
  g_value_set_boolean (&value, TRUE);
  EXPECT_TRUE (gst_tensor_filter_common_set_property (&priv, PROP_IS_UPDATABLE, &value, NULL));
  g_value_unset (&value);

  EXPECT_EQ (event_mock_subplugin::num_events[RELOAD_MODEL], 1U);
  EXPECT_TRUE (event_mock_subplugin::last_data == nullptr);
  EXPECT_TRUE (_event_record_is_zeroed ());
  EXPECT_TRUE (priv.is_updatable);
}

/**
 * @brief A C++ sub-plugin not supporting RELOAD_MODEL is not updatable.
 */
TEST_F (testFilterEvent, isUpdatableUnsupported_n)
{
  GValue value = G_VALUE_INIT;

  event_mock_subplugin::event_ret = -ENOENT;
  g_value_init (&value, G_TYPE_BOOLEAN);
  g_value_set_boolean (&value, TRUE);
  EXPECT_TRUE (gst_tensor_filter_common_set_property (&priv, PROP_IS_UPDATABLE, &value, NULL));
  g_value_unset (&value);

  EXPECT_EQ (event_mock_subplugin::num_events[RELOAD_MODEL], 1U);
  EXPECT_FALSE (priv.is_updatable);
}

/**
 * @brief Test fixture sending the model update event to a running tensor_filter.
 */
class testFilterUpdateModelEvent : public testFilterEvent
{
  protected:
  GstElement *filter;
  GstPad *sinkpad;

  /** @brief start a tensor_filter with the mock and a model */
  void SetUp () override
  {
    testFilterEvent::SetUp ();

    filter = gst_element_factory_make ("tensor_filter", NULL);
    ASSERT_TRUE (filter != nullptr);
    g_object_set (filter, "framework", event_mock_subplugin::mock_name, "model",
        "first.model", NULL);
    sinkpad = gst_element_get_static_pad (filter, "sink");
    ASSERT_TRUE (sinkpad != nullptr);
    ASSERT_EQ (gst_element_set_state (filter, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);
  }

  /** @brief stop the tensor_filter */
  void TearDown () override
  {
    if (sinkpad)
      gst_object_unref (sinkpad);
    if (filter) {
      EXPECT_EQ (gst_element_set_state (filter, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
      gst_object_unref (filter);
    }

    testFilterEvent::TearDown ();
  }

  /**
   * @brief Send a custom downstream event to the sink pad of the filter.
   * @param structure the content of the event (transfer full)
   * @return the result of the event handler
   */
  gboolean sendEvent (GstStructure *structure)
  {
    return gst_pad_send_event (
        sinkpad, gst_event_new_custom (GST_EVENT_CUSTOM_DOWNSTREAM, structure));
  }

  /**
   * @brief Tell whether the model property of the filter is the given one.
   */
  gboolean modelIs (const gchar *expected)
  {
    g_autofree gchar *model = NULL;

    g_object_get (filter, "model", &model, NULL);
    return g_strcmp0 (model, expected) == 0;
  }
};

/**
 * @brief The model update event sets the model and the sub-plugin reloads it.
 */
TEST_F (testFilterUpdateModelEvent, update)
{
  guint reloads;

  g_object_set (filter, "is-updatable", TRUE, NULL);
  reloads = event_mock_subplugin::num_events[RELOAD_MODEL];

  EXPECT_TRUE (sendEvent (gst_structure_new (
      "evt_update_model", "model_files", G_TYPE_STRING, "second.model", NULL)));

  EXPECT_TRUE (modelIs ("second.model"));
  EXPECT_EQ (event_mock_subplugin::num_events[RELOAD_MODEL], reloads + 1);
  ASSERT_EQ (event_mock_subplugin::last_event.num_models, 1);
  EXPECT_STREQ (event_mock_subplugin::last_event.model_files[0], "second.model");

  EXPECT_TRUE (sendEvent (gst_structure_new ("evt_update_model", "model_files",
      G_TYPE_STRING, "third.model,fourth.model", NULL)));

  EXPECT_TRUE (modelIs ("third.model,fourth.model"));
  EXPECT_EQ (event_mock_subplugin::num_events[RELOAD_MODEL], reloads + 2);
  ASSERT_EQ (event_mock_subplugin::last_event.num_models, 2);
  EXPECT_STREQ (event_mock_subplugin::last_event.model_files[0], "third.model");
  EXPECT_STREQ (event_mock_subplugin::last_event.model_files[1], "fourth.model");
}

/**
 * @brief The model update event is refused when the filter is not updatable.
 */
TEST_F (testFilterUpdateModelEvent, notUpdatable_n)
{
  EXPECT_FALSE (sendEvent (gst_structure_new (
      "evt_update_model", "model_files", G_TYPE_STRING, "second.model", NULL)));

  EXPECT_TRUE (modelIs ("first.model"));
  EXPECT_EQ (event_mock_subplugin::num_events[RELOAD_MODEL], 0U);
}

/**
 * @brief The model update event without model files is refused.
 */
TEST_F (testFilterUpdateModelEvent, noModelFiles_n)
{
  guint reloads;

  g_object_set (filter, "is-updatable", TRUE, NULL);
  reloads = event_mock_subplugin::num_events[RELOAD_MODEL];

  EXPECT_FALSE (sendEvent (gst_structure_new_empty ("evt_update_model")));

  EXPECT_TRUE (modelIs ("first.model"));
  EXPECT_EQ (event_mock_subplugin::num_events[RELOAD_MODEL], reloads);
}

/**
 * @brief The model update event with model files that are not a string is refused.
 */
TEST_F (testFilterUpdateModelEvent, modelFilesNotString_n)
{
  guint reloads;

  g_object_set (filter, "is-updatable", TRUE, NULL);
  reloads = event_mock_subplugin::num_events[RELOAD_MODEL];

  EXPECT_FALSE (sendEvent (
      gst_structure_new ("evt_update_model", "model_files", G_TYPE_INT, 1, NULL)));

  EXPECT_TRUE (modelIs ("first.model"));
  EXPECT_EQ (event_mock_subplugin::num_events[RELOAD_MODEL], reloads);
}

/**
 * @brief The model update event with a NULL string as model files is refused.
 */
TEST_F (testFilterUpdateModelEvent, modelFilesNull_n)
{
  guint reloads;

  g_object_set (filter, "is-updatable", TRUE, NULL);
  reloads = event_mock_subplugin::num_events[RELOAD_MODEL];

  EXPECT_FALSE (sendEvent (gst_structure_new (
      "evt_update_model", "model_files", G_TYPE_STRING, NULL, NULL)));

  EXPECT_TRUE (modelIs ("first.model"));
  EXPECT_EQ (event_mock_subplugin::num_events[RELOAD_MODEL], reloads);
}

/**
 * @brief A custom event of another name does not update the model.
 */
TEST_F (testFilterUpdateModelEvent, otherEvent_n)
{
  guint reloads;

  g_object_set (filter, "is-updatable", TRUE, NULL);
  reloads = event_mock_subplugin::num_events[RELOAD_MODEL];

  sendEvent (gst_structure_new (
      "evt_other", "model_files", G_TYPE_STRING, "second.model", NULL));

  EXPECT_TRUE (modelIs ("first.model"));
  EXPECT_EQ (event_mock_subplugin::num_events[RELOAD_MODEL], reloads);
}

/**
 * @brief State of the sub-plugin checking what runs while it is being unloaded.
 */
static struct {
  std::atomic<int> unloading; /**< SUSPEND or close in progress */
  std::atomic<int> unload_entered; /**< number of SUSPEND and close calls started */
  std::atomic<int> overlaps; /**< invokes run while unloading */
  std::atomic<int> invokes; /**< number of invokes */
  std::atomic<int> opens; /**< number of opens */
  std::atomic<int> resumes; /**< number of RESUME calls */
  int suspend_ret; /**< return value of SUSPEND */
} unload_mock;

static const char unload_mock_name[] = "unload_mock_subplugin";
static const gulong unload_mock_delay_us = 500000;

/**
 * @brief Mark the sub-plugin busy unloading for a while.
 */
static void
unload_mock_busy (void)
{
  unload_mock.unloading = 1;
  unload_mock.unload_entered++;
  g_usleep (unload_mock_delay_us);
  unload_mock.unloading = 0;
}

/**
 * @brief open callback of the unload mock.
 */
static int
unload_mock_open (const GstTensorFilterProperties *prop, void **private_data)
{
  UNUSED (prop);
  *private_data = g_new0 (int, 1);
  unload_mock.opens++;
  return 0;
}

/**
 * @brief close callback of the unload mock.
 */
static void
unload_mock_close (const GstTensorFilterProperties *prop, void **private_data)
{
  UNUSED (prop);
  unload_mock_busy ();
  g_free (*private_data);
  *private_data = NULL;
}

/**
 * @brief invoke callback of the unload mock, copying the input.
 */
static int
unload_mock_invoke (const GstTensorFilterFramework *self, GstTensorFilterProperties *prop,
    void *private_data, const GstTensorMemory *input, GstTensorMemory *output)
{
  UNUSED (self);
  UNUSED (prop);
  if (unload_mock.unloading.load () || !private_data)
    unload_mock.overlaps++;
  memcpy (output[0].data, input[0].data, MIN (input[0].size, output[0].size));
  unload_mock.invokes++;
  return 0;
}

/**
 * @brief getFrameworkInfo callback of the unload mock.
 */
static int
unload_mock_get_fw_info (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    GstTensorFilterFrameworkInfo *info)
{
  UNUSED (self);
  UNUSED (prop);
  UNUSED (private_data);
  memset (info, 0, sizeof (*info));
  info->name = unload_mock_name;
  info->run_without_model = 1;
  return 0;
}

/**
 * @brief getModelInfo callback of the unload mock: one uint8 tensor of 4.
 */
static int
unload_mock_get_model_info (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    model_info_ops ops, GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  GstTensorsInfo *infos[] = { in_info, out_info };
  guint i;

  UNUSED (self);
  UNUSED (prop);
  UNUSED (private_data);
  if (ops != GET_IN_OUT_INFO)
    return -ENOENT;

  for (i = 0; i < G_N_ELEMENTS (infos); i++) {
    gst_tensors_info_init (infos[i]);
    infos[i]->num_tensors = 1;
    infos[i]->info[0].type = _NNS_UINT8;
    infos[i]->info[0].dimension[0] = 4;
  }
  return 0;
}

/**
 * @brief eventHandler callback of the unload mock.
 */
static int
unload_mock_event (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data, event_ops ops,
    GstTensorFilterFrameworkEventData *data)
{
  UNUSED (self);
  UNUSED (prop);
  UNUSED (private_data);
  UNUSED (data);
  if (ops == SUSPEND) {
    if (unload_mock.suspend_ret != 0)
      return unload_mock.suspend_ret;
    unload_mock_busy ();
    return 0;
  }
  if (ops == RESUME) {
    unload_mock.resumes++;
    return 0;
  }
  return -ENOENT;
}

/**
 * @brief Wait until @a counter reaches @a value, for at most 5 seconds.
 */
static gboolean
_wait_counter (std::atomic<int> &counter, int value)
{
  guint i;

  for (i = 0; i < 5000 && counter.load () < value; i++)
    g_usleep (1000);

  return counter.load () >= value;
}

/**
 * @brief Push a 4-byte buffer to appsrc.
 */
static GstFlowReturn
_push_4bytes (GstElement *src)
{
  static const guint8 data[4] = { 1, 2, 3, 4 };
  GstBuffer *buf = gst_buffer_new_allocate (NULL, sizeof (data), NULL);

  gst_buffer_fill (buf, 0, data, sizeof (data));
  return gst_app_src_push_buffer (GST_APP_SRC (src), buf);
}

/**
 * @brief Push a buffer into tensor_filter with the suspend watchdog while the
 *        watchdog is unloading the sub-plugin.
 * @param suspend_ret the value SUSPEND returns (0: suspended, else: closed)
 */
static void
_run_unload_race (int suspend_ret)
{
  static GstTensorFilterFramework fw;
  GstElement *pipeline, *src;
  int entered;

  memset (&fw, 0, sizeof (fw));
  fw.version = GST_TENSOR_FILTER_FRAMEWORK_V1;
  fw.open = unload_mock_open;
  fw.close = unload_mock_close;
  fw.invoke = unload_mock_invoke;
  fw.getFrameworkInfo = unload_mock_get_fw_info;
  fw.getModelInfo = unload_mock_get_model_info;
  fw.eventHandler = unload_mock_event;

  unload_mock.unloading = 0;
  unload_mock.unload_entered = 0;
  unload_mock.overlaps = 0;
  unload_mock.invokes = 0;
  unload_mock.opens = 0;
  unload_mock.resumes = 0;
  unload_mock.suspend_ret = suspend_ret;
  ASSERT_TRUE (nnstreamer_filter_probe (&fw));

  pipeline = gst_parse_launch (
      "appsrc name=src caps=other/tensors,format=static,num_tensors=1,"
      "dimensions=(string)4,types=(string)uint8,framerate=0/1 ! "
      "tensor_filter framework=unload_mock_subplugin suspend=50 ! fakesink async=false",
      NULL);
  src = pipeline ? gst_bin_get_by_name (GST_BIN (pipeline), "src") : nullptr;
  if (!src) {
    if (pipeline)
      gst_object_unref (pipeline);
    nnstreamer_filter_exit (unload_mock_name);
    FAIL () << "Failed to create the pipeline.";
  }
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (_push_4bytes (src), GST_FLOW_OK);
  EXPECT_TRUE (_wait_counter (unload_mock.invokes, 1));

  /* The watchdog unloads 50 ms after the invoke; push while it is unloading. */
  entered = unload_mock.unload_entered.load ();
  EXPECT_TRUE (_wait_counter (unload_mock.unload_entered, entered + 1));
  EXPECT_EQ (_push_4bytes (src), GST_FLOW_OK);
  EXPECT_TRUE (_wait_counter (unload_mock.invokes, 2));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (src);
  gst_object_unref (pipeline);
  nnstreamer_filter_exit (unload_mock_name);

  EXPECT_EQ (unload_mock.overlaps.load (), 0);
  EXPECT_EQ (unload_mock.invokes.load (), 2);
}

/**
 * @brief A buffer arriving while the suspend watchdog suspends the sub-plugin
 *        is invoked only after the model is suspended and resumed.
 */
TEST (testFilterSuspendWatchdog, invokeWhileSuspending)
{
  _run_unload_race (0);

  EXPECT_EQ (unload_mock.opens.load (), 1);
  EXPECT_GE (unload_mock.resumes.load (), 1);
}

/**
 * @brief A buffer arriving while the suspend watchdog closes a sub-plugin that
 *        refuses SUSPEND is invoked only after the model is opened again.
 */
TEST (testFilterSuspendWatchdog, invokeWhileClosing_n)
{
  _run_unload_race (-ENOENT);

  EXPECT_GE (unload_mock.opens.load (), 2);
  EXPECT_EQ (unload_mock.resumes.load (), 0);
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
