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
#include <errno.h>
#include <glib.h>
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
      num_events[ops]++;

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
