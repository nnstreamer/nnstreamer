/**
 * @file        unittest_decoder_plugin_data.cc
 * @date        17 Sep 2026
 * @brief       Unit test for the lifetime of the sub-plugin private data in tensor_decoder
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/base/gstbasetransform.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <string.h>

#define TEST_TIMEOUT_MS (5000U)

/** @brief Private data of the lifedec sub-plugin */
typedef struct {
  guint magic;
  guint options;
  guint order;
  gboolean ready; /**< set by getOutCaps (), which decode () needs first, like bounding_boxes */
  gboolean refuse; /**< getOutCaps () refuses every config */
  guint dim; /**< the first dimension getOutCaps () saw, which it holds on to like bounding_boxes does with max_detection */
} lifedec_data_s;

#define LIFEDEC_ALIVE (0x11223344U)
#define LIFEDEC_DEAD (0xdeadbeefU)
#define LIFEDEC_CAPS \
  "other/tensors,format=static,num_tensors=1,dimensions=(string)4,types=(string)uint8,framerate=(fraction)0/1"

/** @brief Where a lifedec sub-plugin call stops until the test releases it */
typedef enum {
  LIFEDEC_HOLD_NONE = 0,
  LIFEDEC_HOLD_CAPS,
  LIFEDEC_HOLD_DECODE,
  LIFEDEC_HOLD_SIZE,
  LIFEDEC_HOLD_OPTION,
} lifedec_hold_e;

/** @brief Shared state of the lifedec sub-plugin and the tests */
static struct {
  GMutex lock;
  GCond cond;
  guint inits;
  guint exits;
  guint caps_calls;
  GSList *released_data;
  lifedec_hold_e hold;
  gboolean entered;
  gboolean released;
  gboolean done;
  guint magic_seen;
  guint options_seen;
  guint order_seen;
  guint dim_seen;
} lifedec;

/** @brief Reset the lifedec counters */
static void
lifedec_reset (void)
{
  g_mutex_lock (&lifedec.lock);
  lifedec.inits = lifedec.exits = lifedec.caps_calls = 0;
  g_mutex_unlock (&lifedec.lock);
}

/** @brief Free the private data released by exit (), kept to observe a use after release */
static void
lifedec_free_released (void)
{
  g_mutex_lock (&lifedec.lock);
  g_slist_free_full (lifedec.released_data, g_free);
  lifedec.released_data = NULL;
  g_mutex_unlock (&lifedec.lock);
}

/** @brief Make the next call at @a site stop until lifedec_release () */
static void
lifedec_hold (lifedec_hold_e site)
{
  g_mutex_lock (&lifedec.lock);
  lifedec.hold = site;
  lifedec.entered = lifedec.released = lifedec.done = FALSE;
  lifedec.magic_seen = lifedec.options_seen = lifedec.order_seen = 0;
  lifedec.dim_seen = 0;
  g_mutex_unlock (&lifedec.lock);
}

/** @brief Let a held call return */
static void
lifedec_release (void)
{
  g_mutex_lock (&lifedec.lock);
  lifedec.released = TRUE;
  g_cond_broadcast (&lifedec.cond);
  g_mutex_unlock (&lifedec.lock);
}

/** @brief Wait until @a flag is set or @a ms passes */
static gboolean
lifedec_wait_flag (gboolean *flag, guint ms)
{
  gint64 end = g_get_monotonic_time () + ms * G_TIME_SPAN_MILLISECOND;
  gboolean ret;

  g_mutex_lock (&lifedec.lock);
  while (!*flag && g_cond_wait_until (&lifedec.cond, &lifedec.lock, end))
    ;
  ret = *flag;
  g_mutex_unlock (&lifedec.lock);
  return ret;
}

/** @brief Stop in a call at @a site if it is held, then read the private data the call started with */
static void
lifedec_enter (lifedec_hold_e site, void **pdata)
{
  lifedec_data_s *data = (lifedec_data_s *) *pdata;

  g_mutex_lock (&lifedec.lock);
  if (lifedec.hold == site) {
    lifedec.hold = LIFEDEC_HOLD_NONE;
    lifedec.entered = TRUE;
    g_cond_broadcast (&lifedec.cond);
    while (!lifedec.released)
      g_cond_wait (&lifedec.cond, &lifedec.lock);
    lifedec.magic_seen = data->magic;
  }
  g_mutex_unlock (&lifedec.lock);
}

/** @brief Mark a test thread finished */
static void
lifedec_set_done (void)
{
  g_mutex_lock (&lifedec.lock);
  lifedec.done = TRUE;
  g_cond_broadcast (&lifedec.cond);
  g_mutex_unlock (&lifedec.lock);
}

/** @brief lifedec init callback */
static int
lifedec_init (void **pdata)
{
  lifedec_data_s *data = g_new0 (lifedec_data_s, 1);

  data->magic = LIFEDEC_ALIVE;
  *pdata = data;

  g_mutex_lock (&lifedec.lock);
  lifedec.inits++;
  g_mutex_unlock (&lifedec.lock);
  return TRUE;
}

/** @brief lifedec exit callback, marks the data dead and keeps it until the test frees it */
static void
lifedec_exit (void **pdata)
{
  lifedec_data_s *data = (lifedec_data_s *) *pdata;

  g_mutex_lock (&lifedec.lock);
  lifedec.exits++;
  if (data) {
    data->magic = LIFEDEC_DEAD;
    lifedec.released_data = g_slist_prepend (lifedec.released_data, data);
  }
  g_mutex_unlock (&lifedec.lock);
  *pdata = NULL;
}

/**
 * @brief lifedec setOption callback. "replace" swaps the private data like the python3 decoder loading a script, and "refuse" makes getOutCaps () refuse every config.
 */
static int
lifedec_setOption (void **pdata, int opNum, const char *param)
{
  lifedec_data_s *data;

  lifedec_enter (LIFEDEC_HOLD_OPTION, pdata);

  if (g_strcmp0 (param, "replace") == 0) {
    lifedec_exit (pdata);
    lifedec_init (pdata);
  }

  data = (lifedec_data_s *) *pdata;
  data->options |= 1U << opNum;
  data->order = data->order * 10 + opNum + 1;
  if (g_strcmp0 (param, "refuse") == 0)
    data->refuse = TRUE;
  return TRUE;
}

/** @brief lifedec getOutCaps callback, whose caps carry the options the private data has */
static GstCaps *
lifedec_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  lifedec_data_s *data;

  lifedec_enter (LIFEDEC_HOLD_CAPS, pdata);

  data = (lifedec_data_s *) *pdata;
  g_mutex_lock (&lifedec.lock);
  lifedec.caps_calls++;
  lifedec.options_seen = data->options;
  lifedec.order_seen = data->order;
  g_mutex_unlock (&lifedec.lock);

  if (data->refuse)
    return NULL;

  /* the first config decides what decode () works on, as in bounding_boxes */
  if (data->dim == 0)
    data->dim = config->info.info[0].dimension[0];

  data->ready = TRUE;
  return gst_caps_new_simple ("application/octet-stream", "options", G_TYPE_INT,
      (gint) data->options, NULL);
}

/** @brief lifedec getTransformSize callback, leaves the output size to decode () */
static size_t
lifedec_getTransformSize (void **pdata, const GstTensorsConfig *config,
    GstCaps *caps, size_t size, GstCaps *othercaps, GstPadDirection direction)
{
  lifedec_enter (LIFEDEC_HOLD_SIZE, pdata);
  return 0;
}

/** @brief lifedec decode callback */
static GstFlowReturn
lifedec_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  lifedec_data_s *data;

  lifedec_enter (LIFEDEC_HOLD_DECODE, pdata);

  data = (lifedec_data_s *) *pdata;
  g_mutex_lock (&lifedec.lock);
  lifedec.dim_seen = data->dim;
  g_mutex_unlock (&lifedec.lock);
  return data->ready ? GST_FLOW_OK : GST_FLOW_ERROR;
}

/** @brief Thread body switching the mode to lifedec2 */
static gpointer
lifedec_set_mode_thread (gpointer element)
{
  g_object_set (G_OBJECT (element), "mode", "lifedec2", NULL);
  lifedec_set_done ();
  return NULL;
}

/** @brief Thread body setting option1 to "replace" */
static gpointer
lifedec_set_option_thread (gpointer element)
{
  g_object_set (G_OBJECT (element), "option1", "replace", NULL);
  lifedec_set_done ();
  return NULL;
}

/** @brief Thread body running a sink-to-src caps transform of the decoder */
static gpointer
lifedec_caps_thread (gpointer element)
{
  GstBaseTransform *trans = GST_BASE_TRANSFORM (element);
  GstCaps *caps = gst_caps_from_string (LIFEDEC_CAPS);
  GstCaps *result;

  result = GST_BASE_TRANSFORM_GET_CLASS (trans)->transform_caps (
      trans, GST_PAD_SINK, caps, NULL);
  gst_caps_unref (caps);
  return result;
}

/** @brief Thread body pushing one 4-byte tensor into the harness */
static gpointer
lifedec_push_thread (gpointer harness)
{
  GstHarness *h = (GstHarness *) harness;

  return GINT_TO_POINTER (gst_harness_push (h, gst_harness_create_buffer (h, 4)));
}

/** @brief Make a harness around a tensor_decoder whose mode is already set, so its pads can link */
static GstHarness *
lifedec_harness_new (void)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  GstHarness *h;

  if (!dec)
    return NULL;

  g_object_set (dec, "mode", "lifedec", NULL);
  h = gst_harness_new_with_element (dec, "sink", "src");
  gst_object_unref (dec);
  gst_harness_set_src_caps_str (h, LIFEDEC_CAPS);
  return h;
}

/**
 * @brief Test fixture registering the lifedec sub-plugins
 */
class tensorDecoderPluginData : public ::testing::Test
{
  protected:
  GstTensorDecoderDef def[2];

  /** @brief Register lifedec and lifedec2, two modes sharing the callbacks */
  void SetUp () override
  {
    guint i;

    memset (def, 0, sizeof (def));
    for (i = 0; i < 2; i++) {
      def[i].modename = (char *) (i == 0 ? "lifedec" : "lifedec2");
      def[i].init = lifedec_init;
      def[i].exit = lifedec_exit;
      def[i].setOption = lifedec_setOption;
      def[i].getOutCaps = lifedec_getOutCaps;
      def[i].getTransformSize = lifedec_getTransformSize;
      def[i].decode = lifedec_decode;
      ASSERT_TRUE (nnstreamer_decoder_probe (&def[i]));
    }
    lifedec_reset ();
    lifedec_hold (LIFEDEC_HOLD_NONE);
  }

  /** @brief Unregister the modes and free what they released */
  void TearDown () override
  {
    nnstreamer_decoder_exit ("lifedec");
    nnstreamer_decoder_exit ("lifedec2");
    lifedec_free_released ();
  }
};

/**
 * @brief Setting the same mode again releases the private data before init () allocates a new one
 */
TEST_F (tensorDecoderPluginData, setSameModeReleasesPrevious)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  ASSERT_TRUE (dec != NULL);

  g_object_set (dec, "mode", "lifedec", NULL);
  g_object_set (dec, "mode", "lifedec", NULL);
  g_object_set (dec, "mode", "lifedec", NULL);
  EXPECT_EQ (lifedec.inits, 3U);
  EXPECT_EQ (lifedec.exits, 2U);

  gst_object_unref (dec);
  EXPECT_EQ (lifedec.exits, 3U);
}

/**
 * @brief Switching to another mode releases the private data of the previous one
 */
TEST_F (tensorDecoderPluginData, setOtherModeReleasesPrevious)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  gchar *mode = NULL;
  ASSERT_TRUE (dec != NULL);

  g_object_set (dec, "mode", "lifedec", NULL);
  g_object_set (dec, "mode", "lifedec2", NULL);
  EXPECT_EQ (lifedec.inits, 2U);
  EXPECT_EQ (lifedec.exits, 1U);

  g_object_get (dec, "mode", &mode, NULL);
  EXPECT_STREQ (mode, "lifedec2");
  g_free (mode);

  gst_object_unref (dec);
  EXPECT_EQ (lifedec.exits, 2U);
}

/**
 * @brief An unknown mode releases the private data of the previous one and leaves no mode
 */
TEST_F (tensorDecoderPluginData, setUnknownModeReleases_n)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  gchar *mode = NULL;
  ASSERT_TRUE (dec != NULL);

  g_object_set (dec, "mode", "lifedec", NULL);
  g_object_set (dec, "mode", "no-such-decoder-mode", NULL);
  EXPECT_EQ (lifedec.inits, 1U);
  EXPECT_EQ (lifedec.exits, 1U);

  g_object_get (dec, "mode", &mode, NULL);
  EXPECT_STREQ (mode, "");
  g_free (mode);

  gst_object_unref (dec);
  EXPECT_EQ (lifedec.exits, 1U);
}

/**
 * @brief A mode change does not wait for getOutCaps (), and the call keeps its private data until it returns
 */
TEST_F (tensorDecoderPluginData, capsQueryKeepsDataOnModeChange)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  GThread *caps_thread, *mode_thread;
  GstCaps *result;
  ASSERT_TRUE (dec != NULL);

  g_object_set (dec, "mode", "lifedec", NULL);
  lifedec_reset ();
  lifedec_hold (LIFEDEC_HOLD_CAPS);

  caps_thread = g_thread_new ("caps", lifedec_caps_thread, dec);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.entered, TEST_TIMEOUT_MS));

  mode_thread = g_thread_new ("mode", lifedec_set_mode_thread, dec);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.done, TEST_TIMEOUT_MS));
  EXPECT_EQ (lifedec.inits, 1U);
  EXPECT_EQ (lifedec.exits, 0U);

  lifedec_release ();
  result = (GstCaps *) g_thread_join (caps_thread);
  g_thread_join (mode_thread);

  EXPECT_EQ (lifedec.magic_seen, LIFEDEC_ALIVE);
  EXPECT_EQ (lifedec.exits, 1U);
  ASSERT_TRUE (result != NULL);
  EXPECT_FALSE (gst_caps_is_empty (result));
  gst_caps_unref (result);

  gst_object_unref (dec);
  EXPECT_EQ (lifedec.exits, 2U);
}

/**
 * @brief An option change does not wait for getOutCaps (), and the call keeps its private data until it returns
 */
TEST_F (tensorDecoderPluginData, capsQueryKeepsDataOnOptionChange)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  GThread *caps_thread, *option_thread;
  GstCaps *result;
  ASSERT_TRUE (dec != NULL);

  g_object_set (dec, "mode", "lifedec", NULL);
  lifedec_hold (LIFEDEC_HOLD_CAPS);

  caps_thread = g_thread_new ("caps", lifedec_caps_thread, dec);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.entered, TEST_TIMEOUT_MS));

  option_thread = g_thread_new ("option", lifedec_set_option_thread, dec);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.done, TEST_TIMEOUT_MS));

  lifedec_release ();
  result = (GstCaps *) g_thread_join (caps_thread);
  g_thread_join (option_thread);

  EXPECT_EQ (lifedec.magic_seen, LIFEDEC_ALIVE);
  if (result)
    gst_caps_unref (result);

  /* the next call sees the new data with option1 */
  lifedec_hold (LIFEDEC_HOLD_NONE);
  result = (GstCaps *) lifedec_caps_thread (dec);
  EXPECT_EQ (lifedec.options_seen, 1U << 0);
  if (result)
    gst_caps_unref (result);

  gst_object_unref (dec);
  EXPECT_EQ (lifedec.inits, lifedec.exits);
}

/**
 * @brief A mode change does not wait for decode (), and the call keeps its private data until it returns
 */
TEST_F (tensorDecoderPluginData, decodeKeepsDataOnModeChange)
{
  GstHarness *h = lifedec_harness_new ();
  GThread *push_thread, *mode_thread;
  ASSERT_TRUE (h != NULL);

  lifedec_reset ();
  lifedec_hold (LIFEDEC_HOLD_DECODE);

  push_thread = g_thread_new ("push", lifedec_push_thread, h);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.entered, TEST_TIMEOUT_MS));

  mode_thread = g_thread_new ("mode", lifedec_set_mode_thread, h->element);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.done, TEST_TIMEOUT_MS));
  EXPECT_EQ (lifedec.exits, 0U);

  lifedec_release ();
  EXPECT_EQ (GPOINTER_TO_INT (g_thread_join (push_thread)), GST_FLOW_OK);
  g_thread_join (mode_thread);

  EXPECT_EQ (lifedec.magic_seen, LIFEDEC_ALIVE);
  EXPECT_EQ (lifedec.exits, 1U);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);

  gst_harness_teardown (h);
}

/**
 * @brief A mode change does not wait for getTransformSize (), and the call keeps its private data until it returns
 */
TEST_F (tensorDecoderPluginData, transformSizeKeepsDataOnModeChange)
{
  GstHarness *h = lifedec_harness_new ();
  GThread *push_thread, *mode_thread;
  ASSERT_TRUE (h != NULL);

  lifedec_reset ();
  lifedec_hold (LIFEDEC_HOLD_SIZE);

  push_thread = g_thread_new ("push", lifedec_push_thread, h);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.entered, TEST_TIMEOUT_MS));

  mode_thread = g_thread_new ("mode", lifedec_set_mode_thread, h->element);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.done, TEST_TIMEOUT_MS));
  EXPECT_EQ (lifedec.exits, 0U);

  lifedec_release ();
  EXPECT_EQ (GPOINTER_TO_INT (g_thread_join (push_thread)), GST_FLOW_OK);
  g_thread_join (mode_thread);

  EXPECT_EQ (lifedec.magic_seen, LIFEDEC_ALIVE);
  EXPECT_EQ (lifedec.exits, 1U);

  gst_harness_teardown (h);
}

/**
 * @brief Options set after the mode reach the sub-plugin in the order they were
 * set, and a mode set gives them in their numbered order
 */
TEST_F (tensorDecoderPluginData, optionOrder)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  GstCaps *result;
  ASSERT_TRUE (dec != NULL);

  g_object_set (dec, "mode", "lifedec", NULL);
  g_object_set (dec, "option3", "any", NULL);
  g_object_set (dec, "option1", "any", NULL);
  g_object_set (dec, "option2", "any", NULL);

  result = (GstCaps *) lifedec_caps_thread (dec);
  EXPECT_EQ (lifedec.order_seen, 312U);
  if (result)
    gst_caps_unref (result);

  g_object_set (dec, "mode", "lifedec", NULL);
  result = (GstCaps *) lifedec_caps_thread (dec);
  EXPECT_EQ (lifedec.order_seen, 123U);
  if (result)
    gst_caps_unref (result);

  gst_object_unref (dec);
}

/**
 * @brief Of two option sets that overlap, the private data of the later one stays and carries both options
 */
TEST_F (tensorDecoderPluginData, overlappingOptionSets)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  GThread *option_thread;
  GstCaps *result;
  ASSERT_TRUE (dec != NULL);

  g_object_set (dec, "mode", "lifedec", NULL);
  lifedec_hold (LIFEDEC_HOLD_OPTION);

  option_thread = g_thread_new ("option", lifedec_set_option_thread, dec);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.entered, TEST_TIMEOUT_MS));

  g_object_set (dec, "option2", "any", NULL);

  lifedec_release ();
  g_thread_join (option_thread);

  result = (GstCaps *) lifedec_caps_thread (dec);
  EXPECT_EQ (lifedec.options_seen, (1U << 0) | (1U << 1));
  if (result)
    gst_caps_unref (result);

  gst_object_unref (dec);
  EXPECT_EQ (lifedec.inits, lifedec.exits);
}

/**
 * @brief New tensor caps mid-stream replace the private data once and the stream goes on
 */
TEST_F (tensorDecoderPluginData, renegotiateReplacesData)
{
  GstHarness *h = lifedec_harness_new ();
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);
  EXPECT_EQ (lifedec.inits, 1U);
  EXPECT_EQ (lifedec.exits, 0U);

  gst_harness_set_src_caps_str (h,
      "other/tensors,format=static,num_tensors=1,dimensions=(string)8,types=(string)uint8,framerate=(fraction)0/1");
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 8)), GST_FLOW_OK);
  EXPECT_EQ (lifedec.inits, 2U);
  EXPECT_EQ (lifedec.exits, 1U);
  EXPECT_EQ (gst_harness_buffers_received (h), 2U);

  gst_harness_teardown (h);
  EXPECT_EQ (lifedec.exits, 2U);
}

/**
 * @brief A buffer whose size does not match the caps is refused without reaching decode ()
 */
TEST_F (tensorDecoderPluginData, decodeWrongSize_n)
{
  GstHarness *h = lifedec_harness_new ();
  ASSERT_TRUE (h != NULL);

  lifedec_hold (LIFEDEC_HOLD_DECODE);
  lifedec_release ();

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 3)), GST_FLOW_ERROR);
  EXPECT_FALSE (lifedec.entered);

  gst_harness_teardown (h);
}

/**
 * @brief An unknown mode after a valid one leaves no sub-plugin to answer a caps query
 */
TEST_F (tensorDecoderPluginData, unknownModeRefusesCaps_n)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  ASSERT_TRUE (dec != NULL);

  g_object_set (dec, "mode", "lifedec", NULL);
  g_object_set (dec, "mode", "no-such-decoder-mode", NULL);
  EXPECT_TRUE (lifedec_caps_thread (dec) == NULL);

  gst_object_unref (dec);
}

/**
 * @brief An option set on a negotiated stream gives the new private data the stream config before any buffer uses it
 */
TEST_F (tensorDecoderPluginData, optionSetKeepsDecoding)
{
  GstHarness *h = lifedec_harness_new ();
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);

  lifedec_reset ();
  g_object_set (h->element, "option1", "any", NULL);
  EXPECT_EQ (lifedec.inits, 1U);
  EXPECT_EQ (lifedec.caps_calls, 1U);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 2U);

  gst_harness_teardown (h);
}

/**
 * @brief A mode set on a negotiated stream gives the new private data the stream config before any buffer uses it
 */
TEST_F (tensorDecoderPluginData, modeSetKeepsDecoding)
{
  GstHarness *h = lifedec_harness_new ();
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);

  g_object_set (h->element, "mode", "lifedec2", NULL);
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 2U);

  gst_harness_teardown (h);
}

/**
 * @brief An option set on a negotiated stream renegotiates, so output caps that depend on the option follow it
 */
TEST_F (tensorDecoderPluginData, optionSetRenegotiates)
{
  GstHarness *h = lifedec_harness_new ();
  GstCaps *caps;
  gint options = -1;
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);

  g_object_set (h->element, "option2", "any", NULL);
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);

  caps = gst_pad_get_current_caps (h->sinkpad);
  ASSERT_TRUE (caps != NULL);
  EXPECT_TRUE (gst_structure_get_int (gst_caps_get_structure (caps, 0), "options", &options));
  EXPECT_EQ (options, 1 << 1);
  gst_caps_unref (caps);

  gst_harness_teardown (h);
}

/**
 * @brief An option that makes the sub-plugin refuse the negotiated config stops the stream instead of decoding with it
 */
TEST_F (tensorDecoderPluginData, optionSetRefusedStops_n)
{
  GstHarness *h = lifedec_harness_new ();
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);

  g_object_set (h->element, "option1", "refuse", NULL);
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_NOT_NEGOTIATED);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);

  gst_harness_teardown (h);
}

/**
 * @brief Caps that change while an option set builds new private data leave the data of the caps the stream ended up with
 */
TEST_F (tensorDecoderPluginData, optionSetDuringRenegotiation)
{
  GstHarness *h = lifedec_harness_new ();
  GThread *option_thread;
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);

  lifedec_hold (LIFEDEC_HOLD_OPTION);
  option_thread = g_thread_new ("option", lifedec_set_option_thread, h->element);
  EXPECT_TRUE (lifedec_wait_flag (&lifedec.entered, TEST_TIMEOUT_MS));

  /* the stream moves to another tensor size while the option is being applied */
  gst_harness_set_src_caps_str (h,
      "other/tensors,format=static,num_tensors=1,dimensions=(string)8,types=(string)uint8,framerate=(fraction)0/1");

  lifedec_release ();
  g_thread_join (option_thread);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 8)), GST_FLOW_OK);
  EXPECT_EQ (lifedec.dim_seen, 8U);
  EXPECT_EQ (lifedec.options_seen, 1U << 0);
  EXPECT_EQ (gst_harness_buffers_received (h), 2U);

  gst_harness_teardown (h);
}

/**
 * @brief An option set before the stream is negotiated neither calls getOutCaps () nor needs a config
 */
TEST_F (tensorDecoderPluginData, optionSetBeforeNegotiation)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  ASSERT_TRUE (dec != NULL);

  g_object_set (dec, "mode", "lifedec", NULL);
  g_object_set (dec, "option1", "any", NULL);
  EXPECT_EQ (lifedec.inits, 2U);
  EXPECT_EQ (lifedec.caps_calls, 0U);

  gst_object_unref (dec);
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
