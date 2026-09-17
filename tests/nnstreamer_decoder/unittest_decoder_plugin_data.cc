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
#include <gst/gst.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <string.h>

/** @brief Private data of the lifedec sub-plugin */
typedef struct {
  guint magic;
} lifedec_data_s;

#define LIFEDEC_ALIVE (0x11223344U)
#define LIFEDEC_DEAD (0xdeadbeefU)

/** @brief Shared state of the lifedec sub-plugin and the tests */
static struct {
  GMutex lock;
  guint inits;
  guint exits;
  GSList *released_data;
} lifedec;

/** @brief Reset the lifedec counters */
static void
lifedec_reset (void)
{
  g_mutex_lock (&lifedec.lock);
  lifedec.inits = lifedec.exits = 0;
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

/** @brief lifedec getOutCaps callback */
static GstCaps *
lifedec_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  return gst_caps_from_string ("application/octet-stream");
}

/** @brief lifedec decode callback */
static GstFlowReturn
lifedec_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  return GST_FLOW_OK;
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
      def[i].getOutCaps = lifedec_getOutCaps;
      def[i].decode = lifedec_decode;
      ASSERT_TRUE (nnstreamer_decoder_probe (&def[i]));
    }
    lifedec_reset ();
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
