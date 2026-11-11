/**
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file        unittest_decoder_image_labeling.cc
 * @date        04 Nov 2026
 * @brief       Unit test for the image_labeling mode of tensor_decoder
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <string.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api_util.h>

/* The labels written to the label file, one per line */
static const gchar *test_labels[] = { "background", "cat", "dog", "bird", NULL };
#define NUM_LABELS (G_N_ELEMENTS (test_labels) - 1)

/**
 * @brief Test fixture holding an instance of the image_labeling decoder and a label file.
 */
class tensorDecoderImageLabeling : public ::testing::Test
{
  protected:
  const GstTensorDecoderDef *decoder;
  void *pdata;
  gchar *label_file;

  /**
   * @brief Initialize the decoder instance and write the label file.
   */
  void SetUp () override
  {
    gchar *contents;
    gchar **labels = (gchar **) test_labels;

    pdata = NULL;
    label_file = getTempFilename ();
    ASSERT_TRUE (label_file != NULL);

    contents = g_strjoinv ("\n", labels);
    ASSERT_TRUE (g_file_set_contents (label_file, contents, -1, NULL));
    g_free (contents);

    decoder = nnstreamer_decoder_find ("image_labeling");
    ASSERT_TRUE (decoder != NULL);
    ASSERT_TRUE (decoder->init (&pdata));
  }

  /**
   * @brief Release the decoder instance and remove the label file.
   */
  void TearDown () override
  {
    if (pdata != NULL)
      decoder->exit (&pdata);
    removeTempFile (&label_file);
  }

  /**
   * @brief Decode one score tensor into a new buffer.
   * @param[in] config The tensors config of the input
   * @param[in] scores The score tensor, sized as config describes
   * @param[out] label The decoded label, to be released with g_free () (NULL if decoding fails)
   */
  GstFlowReturn decode (const GstTensorsConfig *config, const void *scores, gchar **label)
  {
    GstTensorMemory input;
    GstBuffer *outbuf = gst_buffer_new ();
    GstMapInfo map;
    GstFlowReturn ret;

    *label = NULL;
    input.size = gst_tensor_info_get_size (&config->info.info[0]);
    input.data = (void *) scores;

    ret = decoder->decode (&pdata, config, &input, outbuf);
    if (ret == GST_FLOW_OK && gst_buffer_map (outbuf, &map, GST_MAP_READ)) {
      *label = g_strndup ((const gchar *) map.data, map.size);
      gst_buffer_unmap (outbuf, &map);
    }

    gst_buffer_unref (outbuf);
    return ret;
  }
};

/**
 * @brief Describe one score tensor of the given dimensions.
 */
static void
setConfig (GstTensorsConfig *config, tensor_type type, guint d0, guint d1 = 1,
    guint d2 = 0, guint d3 = 0)
{
  GstTensorInfo *info;

  gst_tensors_config_init (config);
  config->rate_n = 0;
  config->rate_d = 1;
  config->info.num_tensors = 1;

  info = gst_tensors_info_get_nth_info (&config->info, 0);
  info->type = type;
  info->dimension[0] = d0;
  info->dimension[1] = d1;
  info->dimension[2] = d2;
  info->dimension[3] = d3;
}

/**
 * @brief Fill a score tensor of the given type with low scores and put a high score at best.
 */
static void
fillScores (guint8 *scores, tensor_type type, guint num, guint best)
{
  gsize bpe = gst_tensor_get_element_size (type);
  guint i;

  for (i = 0; i < num; i++) {
    guint8 *cursor = scores + i * bpe;
    gboolean is_best = (i == best);

    switch (type) {
      case _NNS_INT8:
        *(int8_t *) cursor = is_best ? 100 : -100;
        break;
      case _NNS_UINT8:
        *(uint8_t *) cursor = is_best ? 200 : 1;
        break;
      case _NNS_INT16:
        *(int16_t *) cursor = is_best ? 1000 : -1000;
        break;
      case _NNS_UINT16:
        *(uint16_t *) cursor = is_best ? 1000 : 1;
        break;
      case _NNS_INT32:
        *(int32_t *) cursor = is_best ? 100000 : -100000;
        break;
      case _NNS_UINT32:
        *(uint32_t *) cursor = is_best ? 100000U : 1U;
        break;
      case _NNS_INT64:
        *(int64_t *) cursor = is_best ? 100000 : -100000;
        break;
      case _NNS_UINT64:
        *(uint64_t *) cursor = is_best ? 100000U : 1U;
        break;
      case _NNS_FLOAT32:
        *(float *) cursor = is_best ? 0.9f : 0.01f;
        break;
      case _NNS_FLOAT64:
        *(double *) cursor = is_best ? 0.9 : 0.01;
        break;
      default:
        memset (cursor, 0, bpe);
        break;
    }
  }
}

/**
 * @brief The label of the highest score is decoded for every supported type.
 */
TEST_F (tensorDecoderImageLabeling, decodeMaxLabel)
{
  const tensor_type types[] = { _NNS_INT8, _NNS_UINT8, _NNS_INT16, _NNS_UINT16,
    _NNS_INT32, _NNS_UINT32, _NNS_INT64, _NNS_UINT64, _NNS_FLOAT32, _NNS_FLOAT64 };
  uint64_t scores[NUM_LABELS];
  GstTensorsConfig config;
  gchar *label;
  guint t, best;

  ASSERT_TRUE (decoder->setOption (&pdata, 0, label_file));

  for (t = 0; t < G_N_ELEMENTS (types); t++) {
    for (best = 0; best < NUM_LABELS; best++) {
      setConfig (&config, types[t], NUM_LABELS);
      fillScores ((guint8 *) scores, types[t], NUM_LABELS, best);

      EXPECT_EQ (decode (&config, scores, &label), GST_FLOW_OK);
      EXPECT_STREQ (label, test_labels[best]) << "type " << types[t];
      g_free (label);
      gst_tensors_config_free (&config);
    }
  }
}

/**
 * @brief More scores than labels decode as long as the best score has a label.
 */
TEST_F (tensorDecoderImageLabeling, decodeMoreScoresBestHasLabel)
{
  const guint num = NUM_LABELS + 2;
  float scores[NUM_LABELS + 2];
  GstTensorsConfig config;
  gchar *label;

  ASSERT_TRUE (decoder->setOption (&pdata, 0, label_file));
  setConfig (&config, _NNS_FLOAT32, num);
  fillScores ((guint8 *) scores, _NNS_FLOAT32, num, NUM_LABELS - 1);

  EXPECT_EQ (decode (&config, scores, &label), GST_FLOW_OK);
  EXPECT_STREQ (label, test_labels[NUM_LABELS - 1]);
  g_free (label);
  gst_tensors_config_free (&config);
}

/**
 * @brief The best score past the last label is an error, not an abort.
 */
TEST_F (tensorDecoderImageLabeling, decodeMoreScoresThanLabels_n)
{
  const guint num = NUM_LABELS + 2;
  float scores[NUM_LABELS + 2];
  GstTensorsConfig config;
  gchar *label;
  guint best;

  ASSERT_TRUE (decoder->setOption (&pdata, 0, label_file));
  setConfig (&config, _NNS_FLOAT32, num);

  for (best = NUM_LABELS; best < num; best++) {
    fillScores ((guint8 *) scores, _NNS_FLOAT32, num, best);
    EXPECT_EQ (decode (&config, scores, &label), GST_FLOW_ERROR);
    EXPECT_TRUE (label == NULL);
  }

  gst_tensors_config_free (&config);
}

/**
 * @brief Decoding without a label file is an error, not an abort.
 */
TEST_F (tensorDecoderImageLabeling, decodeWithoutLabels_n)
{
  float scores[NUM_LABELS];
  GstTensorsConfig config;
  gchar *label;

  setConfig (&config, _NNS_FLOAT32, NUM_LABELS);
  fillScores ((guint8 *) scores, _NNS_FLOAT32, NUM_LABELS, 0);

  EXPECT_EQ (decode (&config, scores, &label), GST_FLOW_ERROR);
  EXPECT_TRUE (label == NULL);
  gst_tensors_config_free (&config);
}

/**
 * @brief A score type the decoder does not search is refused.
 */
TEST_F (tensorDecoderImageLabeling, decodeUnsupportedType_n)
{
  guint16 scores[NUM_LABELS] = { 0 };
  GstTensorsConfig config;
  gchar *label;

  ASSERT_TRUE (decoder->setOption (&pdata, 0, label_file));
  setConfig (&config, _NNS_FLOAT16, NUM_LABELS);

  EXPECT_EQ (decode (&config, scores, &label), GST_FLOW_NOT_SUPPORTED);
  EXPECT_TRUE (label == NULL);
  gst_tensors_config_free (&config);
}

/**
 * @brief A label file that cannot be read is refused.
 */
TEST_F (tensorDecoderImageLabeling, setOptionMissingFile_n)
{
  gchar *missing = g_strconcat (label_file, ".missing", NULL);

  EXPECT_FALSE (decoder->setOption (&pdata, 0, missing));
  g_free (missing);
}

/**
 * @brief An N:1 score tensor negotiates a utf8 text stream.
 */
TEST_F (tensorDecoderImageLabeling, getOutCapsN1)
{
  GstTensorsConfig config;
  GstCaps *caps, *expected;

  setConfig (&config, _NNS_FLOAT32, NUM_LABELS);
  caps = decoder->getOutCaps (&pdata, &config);
  ASSERT_TRUE (caps != NULL);

  expected = gst_caps_from_string ("text/x-raw, format = (string) utf8");
  EXPECT_TRUE (gst_caps_can_intersect (caps, expected));

  gst_caps_unref (expected);
  gst_caps_unref (caps);
  gst_tensors_config_free (&config);
}

/**
 * @brief Score tensors that are not N:1 are refused.
 */
TEST_F (tensorDecoderImageLabeling, getOutCapsNotN1_n)
{
  GstTensorsConfig config;

  setConfig (&config, _NNS_FLOAT32, NUM_LABELS, 1, 1, 1);
  EXPECT_TRUE (decoder->getOutCaps (&pdata, &config) == NULL);
  gst_tensors_config_free (&config);

  setConfig (&config, _NNS_FLOAT32, NUM_LABELS, 2);
  EXPECT_TRUE (decoder->getOutCaps (&pdata, &config) == NULL);
  gst_tensors_config_free (&config);

  setConfig (&config, _NNS_FLOAT32, 0);
  EXPECT_TRUE (decoder->getOutCaps (&pdata, &config) == NULL);
  gst_tensors_config_free (&config);
}

/**
 * @brief Push one float32 score frame through tensor_decoder.
 * @param[in] label_file The label file for option1
 * @param[in] num The number of scores
 * @param[in] best The index of the highest score
 * @param[out] label The label the appsink received, to be released with g_free () (NULL if none)
 * @return TRUE if the pipeline posted an error, FALSE if it reached EOS
 */
static gboolean
runPipeline (const gchar *label_file, guint num, guint best, gchar **label)
{
  GstElement *pipeline, *src, *sink;
  GstBuffer *buf;
  GstMessage *msg;
  GstSample *sample;
  GstMapInfo map;
  gchar *desc;
  gboolean got_error = FALSE;

  *label = NULL;
  desc = g_strdup_printf ("appsrc name=src caps=other/tensors,num_tensors=1,"
                          "dimensions=(string)%u:1,types=(string)float32,format=static,framerate=(fraction)0/1 ! "
                          "tensor_decoder mode=image_labeling option1=%s ! "
                          "appsink name=sink sync=false",
      num, label_file);
  pipeline = gst_parse_launch (desc, NULL);
  g_free (desc);
  if (pipeline == NULL)
    return FALSE;

  src = gst_bin_get_by_name (GST_BIN (pipeline), "src");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);

  buf = gst_buffer_new_allocate (NULL, num * sizeof (float), NULL);
  if (gst_buffer_map (buf, &map, GST_MAP_WRITE)) {
    fillScores (map.data, _NNS_FLOAT32, num, best);
    gst_buffer_unmap (buf, &map);
  }
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (src), buf), GST_FLOW_OK);
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (src)), GST_FLOW_OK);

  /* appsink holds EOS back until its queued samples are pulled */
  sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 5 * GST_SECOND);
  if (sample != NULL) {
    buf = gst_sample_get_buffer (sample);
    if (buf != NULL && gst_buffer_map (buf, &map, GST_MAP_READ)) {
      *label = g_strndup ((const gchar *) map.data, map.size);
      gst_buffer_unmap (buf, &map);
    }
    gst_sample_unref (sample);
  }

  msg = gst_bus_timed_pop_filtered (GST_ELEMENT_BUS (pipeline), 5 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  EXPECT_TRUE (msg != NULL);
  if (msg != NULL) {
    got_error = (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR);
    gst_message_unref (msg);
  }

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (src);
  gst_object_unref (sink);
  gst_object_unref (pipeline);
  return got_error;
}

/**
 * @brief tensor_decoder passes the label of the highest score downstream.
 */
TEST_F (tensorDecoderImageLabeling, pipelineMaxLabel)
{
  gchar *label;

  EXPECT_FALSE (runPipeline (label_file, NUM_LABELS, 2, &label));
  EXPECT_STREQ (label, test_labels[2]);
  g_free (label);
}

/**
 * @brief tensor_decoder posts an error, and the process survives, when the best score has no label.
 */
TEST_F (tensorDecoderImageLabeling, pipelineMoreScoresThanLabels_n)
{
  gchar *label;

  EXPECT_TRUE (runPipeline (label_file, NUM_LABELS + 1, NUM_LABELS, &label));
  EXPECT_TRUE (label == NULL);
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
    g_warning ("catch testing::internal::ClassUniqueToAlwaysTrue");
  }

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch testing::internal::GoogleTestFailureException");
  }

  return result;
}
