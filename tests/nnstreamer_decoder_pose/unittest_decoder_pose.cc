/**
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file        unittest_decoder_pose.cc
 * @date        10 Sep 2026
 * @brief       Unit test for the pose_estimation mode of tensor_decoder and
 *              the label file loader of the image-based decoder modes
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <string.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_subplugin.h>

/* The number of labels pose_estimation assumes without a label file */
#define DEFAULT_LABELS (14U)

#define LABELS (2U)
#define GRID (5U)
#define HEATMAP_ELEMENTS (LABELS * GRID * GRID)
#define OFFSET_ELEMENTS (2U * HEATMAP_ELEMENTS)

#define OUT_SIZE (40U)
#define OUT_PIXELS (OUT_SIZE * OUT_SIZE)

#define POSE_PIXEL (0xFFFFFFFFU)

/** The two labels, connected to each other */
#define LABEL_TEXT "a 1\nb 0\n"
/** The first label connects to an id one past the last label */
#define LABEL_TEXT_PAST_LAST "a 2\nb\n"

/** The shapes of posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite */
#define POSENET_LABELS (17U)
#define POSENET_GRID (9U)
#define POSENET_HEATMAP_ELEMENTS (POSENET_LABELS * POSENET_GRID * POSENET_GRID)
#define POSENET_OFFSET_ELEMENTS (2U * POSENET_HEATMAP_ELEMENTS)
#define POSENET_OUT_WIDTH (640U)
#define POSENET_OUT_HEIGHT (480U)

/** The 17 posenet keypoints, as documented in tensordec-pose.c */
#define POSENET_LABEL_TEXT                                                         \
  "nose 1 2 3 4\nleftEye 0 2 3\nrightEye 0 1 4\nleftEar 0 1\nrightEar 0 2\n"       \
  "leftShoulder 6 7 11\nrightShoulder 5 8 12\nleftElbow 5 9\nrightElbow 6 10\n"    \
  "leftWrist 7\nrightWrist 8\nleftHip 5 12 13\nrightHip 6 11 14\nleftKnee 11 15\n" \
  "rightKnee 12 16\nleftAnkle 13\nrightAnkle 14\n"

/**
 * @brief Test fixture holding an instance of the pose_estimation decoder.
 */
class tensorDecoderPose : public ::testing::Test
{
  protected:
  const GstTensorDecoderDef *decoder;
  void *pdata;
  gchar *label_file;

  /**
   * @brief Initialize the decoder instance.
   */
  void SetUp () override
  {
    pdata = NULL;
    label_file = NULL;
    decoder = nnstreamer_decoder_find ("pose_estimation");
    ASSERT_TRUE (decoder != NULL);
    ASSERT_TRUE (decoder->init (&pdata));
  }

  /**
   * @brief Release the decoder instance and the label file.
   */
  void TearDown () override
  {
    if (pdata != NULL)
      decoder->exit (&pdata);
    removeTempFile (&label_file);
  }

  /**
   * @brief Write a label file and hand it to the decoder as option3.
   */
  gboolean setLabels (const gchar *text)
  {
    removeTempFile (&label_file);
    label_file = getTempFilename ();
    if (label_file == NULL || !g_file_set_contents (label_file, text, -1, NULL))
      return FALSE;

    return decoder->setOption (&pdata, 2, label_file);
  }

  /**
   * @brief Whether the decoder negotiates an output for the given input.
   */
  gboolean accepts (const GstTensorsConfig *config)
  {
    GstCaps *caps = decoder->getOutCaps (&pdata, config);

    if (caps == NULL)
      return FALSE;

    gst_caps_unref (caps);
    return TRUE;
  }
};

/**
 * @brief Describe a heatmap tensor, and an offset tensor if offset_grid is not 0.
 */
static void
setPoseConfig (GstTensorsConfig *config, tensor_type type, guint labels,
    guint grid, guint offset_grid)
{
  GstTensorInfo *info;

  gst_tensors_config_init (config);
  config->rate_n = 0;
  config->rate_d = 1;
  config->info.num_tensors = (offset_grid > 0) ? 2 : 1;

  info = gst_tensors_info_get_nth_info (&config->info, 0);
  info->type = type;
  info->dimension[0] = labels;
  info->dimension[1] = grid;
  info->dimension[2] = grid;
  info->dimension[3] = 1;

  if (offset_grid > 0) {
    info = gst_tensors_info_get_nth_info (&config->info, 1);
    info->type = type;
    info->dimension[0] = 2 * labels;
    info->dimension[1] = offset_grid;
    info->dimension[2] = offset_grid;
    info->dimension[3] = 1;
  }
}

/**
 * @brief Decode one heatmap, allocated to its exact size, with the given decoder.
 */
static GstFlowReturn
decodeHeatmap (const GstTensorDecoderDef *decoder, void **pdata, const GstTensorsConfig *config)
{
  GstTensorMemory input;
  GstBuffer *outbuf = gst_buffer_new ();
  GstFlowReturn ret;

  input.size = gst_tensor_info_get_size (&config->info.info[0]);
  input.data = g_malloc0 (input.size);

  ret = decoder->decode (pdata, config, &input, outbuf);

  g_free (input.data);
  gst_buffer_unref (outbuf);
  return ret;
}

/**
 * @brief Set one cell of a heatmap tensor (label, x, y).
 */
static void
setHeatmap (float *heatmap, guint label, guint x, guint y, float value)
{
  heatmap[(y * GRID + x) * LABELS + label] = value;
}

/**
 * @brief Write the given floats to a new temp file.
 * @return the file name, to be released with removeTempFile ()
 */
static gchar *
writeTensorFile (const float *data, guint elements)
{
  gchar *name = getTempFilename ();

  if (name != NULL
      && !g_file_set_contents (name, (const gchar *) data, elements * sizeof (float), NULL))
    removeTempFile (&name);

  return name;
}

/**
 * @brief Run the pipeline to its end.
 * @return GST_MESSAGE_EOS or GST_MESSAGE_ERROR, GST_MESSAGE_ANY if the
 *         pipeline could not be built and GST_MESSAGE_UNKNOWN on a timeout.
 */
static GstMessageType
runPipeline (const gchar *pipeline_str)
{
  GstElement *pipeline = gst_parse_launch (pipeline_str, NULL);
  GstMessageType type = GST_MESSAGE_UNKNOWN;
  GstBus *bus;
  GstMessage *msg;

  if (pipeline == NULL)
    return GST_MESSAGE_ANY;

  /* A refused negotiation may fail the state change itself */
  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);

  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  if (msg != NULL) {
    type = GST_MESSAGE_TYPE (msg);
    gst_message_unref (msg);
  }
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);

  return type;
}

/**
 * @brief Decode a heatmap (and its offsets, if given) into a frame.
 * @param[in] heatmap HEATMAP_ELEMENTS floats
 * @param[in] offset OFFSET_ELEMENTS floats for heatmap-offset, NULL for heatmap-only
 * @param[in] label_text the content of the label file
 * @param[in] model_size option2 of the decoder
 * @param[out] frame OUT_PIXELS RGBA pixels drawn by the decoder
 */
static GstMessageType
decodePose (const float *heatmap, const float *offset, const gchar *label_text,
    guint model_size, uint32_t *frame)
{
  float tensors[HEATMAP_ELEMENTS + OFFSET_ELEMENTS];
  guint elements = HEATMAP_ELEMENTS;
  gchar *in_file, *label_file, *out_file, *pipeline_str, *dims, *types;
  gchar *content = NULL;
  gsize len = 0;
  GstMessageType ret = GST_MESSAGE_ANY;

  memcpy (tensors, heatmap, HEATMAP_ELEMENTS * sizeof (float));
  if (offset != NULL) {
    memcpy (tensors + HEATMAP_ELEMENTS, offset, OFFSET_ELEMENTS * sizeof (float));
    elements += OFFSET_ELEMENTS;
    dims = g_strdup_printf (
        "%u:%u:%u:1,%u:%u:%u:1", LABELS, GRID, GRID, 2 * LABELS, GRID, GRID);
    types = g_strdup ("float32,float32");
  } else {
    dims = g_strdup_printf ("%u:%u:%u:1", LABELS, GRID, GRID);
    types = g_strdup ("float32");
  }

  in_file = writeTensorFile (tensors, elements);
  label_file = getTempFilename ();
  out_file = getTempFilename ();

  if (in_file != NULL && label_file != NULL && out_file != NULL
      && g_file_set_contents (label_file, label_text, -1, NULL)) {
    pipeline_str = g_strdup_printf (
        "filesrc location=%s blocksize=%u ! application/octet-stream ! "
        "tensor_converter input-dim=%s input-type=%s ! "
        "tensor_decoder mode=pose_estimation option1=%u:%u option2=%u:%u "
        "option3=%s option4=%s ! "
        "filesink location=%s buffer-mode=unbuffered sync=false async=false",
        in_file, (guint) (elements * sizeof (float)), dims, types, OUT_SIZE,
        OUT_SIZE, model_size, model_size, label_file,
        offset != NULL ? "heatmap-offset" : "heatmap-only", out_file);

    ret = runPipeline (pipeline_str);
    g_free (pipeline_str);

    if (ret == GST_MESSAGE_EOS) {
      if (g_file_get_contents (out_file, &content, &len, NULL)
          && len == OUT_PIXELS * sizeof (uint32_t))
        memcpy (frame, content, len);
      else
        ret = GST_MESSAGE_ANY;
      g_free (content);
    }
  }

  g_free (dims);
  g_free (types);
  removeTempFile (&in_file);
  removeTempFile (&label_file);
  removeTempFile (&out_file);

  return ret;
}

/**
 * @brief Count the pixels the decoder has drawn.
 */
static guint
countDrawnPixels (const uint32_t *frame, guint pixels)
{
  guint i, count = 0;

  for (i = 0; i < pixels; i++) {
    if (frame[i] != 0U)
      count++;
  }

  return count;
}

/**
 * @brief A float32 heatmap with one row per label is accepted.
 */
TEST_F (tensorDecoderPose, acceptFloat32Heatmap)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, 0);
  EXPECT_TRUE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief A heatmap of another element type is refused.
 * @details The decoder reads the heatmap as float32, so a uint8 heatmap of the
 *          same dimensions used to be read four times past its end.
 */
TEST_F (tensorDecoderPose, refuseUint8Heatmap_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));

  setPoseConfig (&config, _NNS_UINT8, DEFAULT_LABELS, GRID, 0);
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief A heatmap whose first dimension is not the number of labels is refused.
 */
TEST_F (tensorDecoderPose, refuseLabelCountMismatch_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS - 1, GRID, 0);
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief A heatmap or an offset tensor with more than three dimensions is refused.
 */
TEST_F (tensorDecoderPose, refuseHigherRank_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, 0);
  config.info.info[0].dimension[3] = 2;
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);

  EXPECT_TRUE (decoder->setOption (&pdata, 3, "heatmap-offset"));

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, GRID);
  config.info.info[1].dimension[3] = 2;
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief A stream is refused while the model input size (option2) is not given.
 * @details The decoder divides by the model input size.
 */
TEST_F (tensorDecoderPose, refuseMissingModelSize_n)
{
  GstTensorsConfig config;

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, 0);
  EXPECT_FALSE (accepts (&config));

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (accepts (&config));

  EXPECT_TRUE (decoder->setOption (&pdata, 1, ""));
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief An offset tensor of the heatmap's grid is accepted in heatmap-offset mode.
 */
TEST_F (tensorDecoderPose, acceptOffsetGrid)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "heatmap-offset"));

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, GRID);
  EXPECT_TRUE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief An offset tensor smaller than the heatmap's grid is refused.
 * @details The offset of a keypoint is looked up at the heatmap cell it was
 *          found in, so a smaller offset tensor used to be read past its end.
 */
TEST_F (tensorDecoderPose, refuseOffsetGridMismatch_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "heatmap-offset"));

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, GRID);
  config.info.info[1].dimension[1] = GRID - 1;
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, GRID);
  config.info.info[1].dimension[2] = GRID - 1;
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief A heatmap of a single cell in a direction is refused in heatmap-offset mode.
 * @details heatmap-offset places a keypoint at its cell index over the number
 *          of cells minus one, which is a division by zero for a single cell.
 */
TEST_F (tensorDecoderPose, refuseSingleCellOffsetGrid_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "heatmap-offset"));

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, GRID);
  config.info.info[0].dimension[1] = config.info.info[1].dimension[1] = 1;
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, GRID);
  config.info.info[0].dimension[2] = config.info.info[1].dimension[2] = 1;
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);

  /* heatmap-only does not divide by the grid */
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "heatmap-only"));
  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, 1, 0);
  EXPECT_TRUE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief A single tensor is refused in heatmap-offset mode.
 */
TEST_F (tensorDecoderPose, refuseMissingOffsetTensor_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "heatmap-offset"));

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, 0);
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief A label file without a label is refused and the labels in use are kept.
 * @details Reading a file of 0 bytes succeeds without an error to report,
 *          and the error message used to be read from the NULL error. A file
 *          of a single newline used to be taken as a file of no labels. The
 *          labels in use come from a file, so a refusal that released them
 *          first leaves a dangling table that exit () frees again.
 */
TEST_F (tensorDecoderPose, refuseEmptyLabelFile_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (setLabels (LABEL_TEXT));
  EXPECT_FALSE (setLabels (""));
  EXPECT_FALSE (setLabels ("\n"));

  setPoseConfig (&config, _NNS_FLOAT32, LABELS, GRID, 0);
  EXPECT_TRUE (accepts (&config));
  gst_tensors_config_free (&config);

  setPoseConfig (&config, _NNS_FLOAT32, DEFAULT_LABELS, GRID, 0);
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief Count the critical messages of the GLib domain.
 */
static void
countGLibCritical (const gchar *, GLogLevelFlags, const gchar *, gpointer user_data)
{
  guint *count = (guint *) user_data;

  (*count)++;
}

/**
 * @brief A blank line in a label file is a keypoint without a label or connections.
 * @details The label was copied from the NULL first token of the blank line.
 */
TEST_F (tensorDecoderPose, loadLabelFileWithBlankLine)
{
  GstTensorsConfig config;
  guint critical = 0;
  guint handler = g_log_set_handler ("GLib",
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION),
      countGLibCritical, &critical);
  GLogLevelFlags fatal_mask;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (setLabels ("a 2\n\nb 0\n"));
  EXPECT_EQ (critical, 0U);

  /* The handler is live: a critical of the domain is counted */
  fatal_mask = g_log_set_always_fatal ((GLogLevelFlags) G_LOG_FATAL_MASK);
  g_log ("GLib", G_LOG_LEVEL_CRITICAL, "self-check of the counter");
  g_log_set_always_fatal (fatal_mask);
  EXPECT_EQ (critical, 1U);
  g_log_remove_handler ("GLib", handler);

  setPoseConfig (&config, _NNS_FLOAT32, 3, GRID, 0);
  EXPECT_TRUE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief A label file given again replaces the labels of the previous one.
 */
TEST_F (tensorDecoderPose, reloadLabelFile)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (setLabels (LABEL_TEXT));
  EXPECT_TRUE (setLabels ("a 1\nb 2\nc 0\n"));

  setPoseConfig (&config, _NNS_FLOAT32, 3, GRID, 0);
  EXPECT_TRUE (accepts (&config));
  gst_tensors_config_free (&config);

  setPoseConfig (&config, _NNS_FLOAT32, LABELS, GRID, 0);
  EXPECT_FALSE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief A heatmap is decoded when the options match the negotiated stream.
 */
TEST_F (tensorDecoderPose, decodeHeatmap)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "40:40"));
  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (setLabels (LABEL_TEXT));

  setPoseConfig (&config, _NNS_FLOAT32, LABELS, GRID, 0);
  EXPECT_EQ (decodeHeatmap (decoder, &pdata, &config), GST_FLOW_OK);
  gst_tensors_config_free (&config);
}

/**
 * @brief Decoding stops when the model input size is cleared after negotiation.
 * @details The decoder divides by the model input size, which an option set
 *          while streaming can clear without a renegotiation.
 */
TEST_F (tensorDecoderPose, decodeAfterModelSizeCleared_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "40:40"));
  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (setLabels (LABEL_TEXT));

  setPoseConfig (&config, _NNS_FLOAT32, LABELS, GRID, 0);
  EXPECT_TRUE (accepts (&config));

  EXPECT_TRUE (decoder->setOption (&pdata, 1, ""));
  EXPECT_EQ (decodeHeatmap (decoder, &pdata, &config), GST_FLOW_ERROR);
  gst_tensors_config_free (&config);
}

/**
 * @brief Decoding stops when a label file with more labels arrives after negotiation.
 * @details The decoder reads one heatmap row per label, so more labels than
 *          the negotiated heatmap has are read past its end.
 */
TEST_F (tensorDecoderPose, decodeAfterLabelsChanged_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "40:40"));
  EXPECT_TRUE (decoder->setOption (&pdata, 1, "5:5"));
  EXPECT_TRUE (setLabels (LABEL_TEXT));

  setPoseConfig (&config, _NNS_FLOAT32, LABELS, GRID, 0);
  EXPECT_TRUE (accepts (&config));

  EXPECT_TRUE (setLabels ("a 1\nb 2\nc 0\n"));
  EXPECT_EQ (decodeHeatmap (decoder, &pdata, &config), GST_FLOW_ERROR);
  gst_tensors_config_free (&config);
}

/**
 * @brief The posenet heatmap and offsets place a keypoint at its pixel in the output.
 * @details heatmap-offset takes the argmax cell over the grid size minus one,
 *          scaled to option2, adds the offsets read at that cell (y first, x
 *          after the offsets of all the labels) and scales the sum from
 *          option2 to option1. Nothing else pins that arithmetic with an
 *          option1 that differs from option2 in each direction.
 */
TEST_F (tensorDecoderPose, decodePosenetKeypointPosition)
{
  GstTensorsConfig config;
  GstTensorMemory input[2];
  float *heatmap, *offset;
  GstBuffer *outbuf;
  GstMapInfo map;
  GstFlowReturn ret;
  const uint32_t *frame;
  guint i;

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "640:480"));
  EXPECT_TRUE (decoder->setOption (&pdata, 1, "257:257"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "heatmap-offset"));
  ASSERT_TRUE (setLabels (POSENET_LABEL_TEXT));

  setPoseConfig (&config, _NNS_FLOAT32, POSENET_LABELS, POSENET_GRID, POSENET_GRID);
  EXPECT_TRUE (accepts (&config));

  input[0].size = POSENET_HEATMAP_ELEMENTS * sizeof (float);
  input[1].size = POSENET_OFFSET_ELEMENTS * sizeof (float);
  heatmap = (float *) g_malloc (input[0].size);
  offset = (float *) g_malloc0 (input[1].size);
  input[0].data = heatmap;
  input[1].data = offset;

  /* Every keypoint but nose (0) and leftEye (1) is below the 0.5 sigmoid */
  for (i = 0; i < POSENET_HEATMAP_ELEMENTS; i++)
    heatmap[i] = -10.0f;
  heatmap[(2 * POSENET_GRID + 4) * POSENET_LABELS + 0] = 10.0f;
  heatmap[(1 * POSENET_GRID + 6) * POSENET_LABELS + 1] = 10.0f;

  offset[(2 * POSENET_GRID + 4) * 2 * POSENET_LABELS + 0] = 5.0f;
  offset[(2 * POSENET_GRID + 4) * 2 * POSENET_LABELS + POSENET_LABELS + 0] = -3.0f;
  offset[(1 * POSENET_GRID + 6) * 2 * POSENET_LABELS + 1] = 2.0f;
  offset[(1 * POSENET_GRID + 6) * 2 * POSENET_LABELS + POSENET_LABELS + 1] = 1.0f;

  outbuf = gst_buffer_new ();
  ret = decoder->decode (&pdata, &config, input, outbuf);
  g_free (heatmap);
  g_free (offset);
  gst_tensors_config_free (&config);
  EXPECT_EQ (ret, GST_FLOW_OK);

  if (!gst_buffer_map (outbuf, &map, GST_MAP_READ)) {
    gst_buffer_unref (outbuf);
    FAIL () << "Cannot map the decoded frame";
  }
  if (map.size != POSENET_OUT_WIDTH * POSENET_OUT_HEIGHT * 4U) {
    gst_buffer_unmap (outbuf, &map);
    gst_buffer_unref (outbuf);
    FAIL () << "The decoded frame is not 640x480 RGBA";
  }
  frame = (const uint32_t *) map.data;

  /**
   * nose, cell (4, 2): x = (4 / 8 * 257 - 3) * 640 / 257 = 312.5,
   * y = (2 / 8 * 257 + 5) * 480 / 257 = 129.3, so (312, 129).
   * The line to leftEye leaves it up and right, and its label is drawn above
   * it, so the dot's left and bottom tips pin it to the pixel.
   */
  EXPECT_EQ (frame[129 * POSENET_OUT_WIDTH + 308], POSE_PIXEL);
  EXPECT_EQ (frame[129 * POSENET_OUT_WIDTH + 307], 0U);
  EXPECT_EQ (frame[133 * POSENET_OUT_WIDTH + 312], POSE_PIXEL);
  EXPECT_EQ (frame[134 * POSENET_OUT_WIDTH + 312], 0U);

  /**
   * leftEye, cell (6, 1): x = (6 / 8 * 257 + 1) * 640 / 257 = 482.4,
   * y = (1 / 8 * 257 + 2) * 480 / 257 = 63.7, so (482, 63).
   * The line comes from the bottom left, so the right and bottom tips pin it.
   */
  EXPECT_EQ (frame[63 * POSENET_OUT_WIDTH + 486], POSE_PIXEL);
  EXPECT_EQ (frame[63 * POSENET_OUT_WIDTH + 487], 0U);
  EXPECT_EQ (frame[67 * POSENET_OUT_WIDTH + 482], POSE_PIXEL);
  EXPECT_EQ (frame[68 * POSENET_OUT_WIDTH + 482], 0U);

  gst_buffer_unmap (outbuf, &map);
  gst_buffer_unref (outbuf);
}

/**
 * @brief The posenet tensors are refused with the 14 built-in keypoints.
 * @details Without option3 the decoder has 14 keypoints, which a heatmap of
 *          17 channels and offsets of 34 do not match.
 */
TEST_F (tensorDecoderPose, refusePosenetWithDefaultLabels_n)
{
  GstTensorsConfig config;

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "640:480"));
  EXPECT_TRUE (decoder->setOption (&pdata, 1, "257:257"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "heatmap-offset"));

  setPoseConfig (&config, _NNS_FLOAT32, POSENET_LABELS, POSENET_GRID, POSENET_GRID);
  EXPECT_FALSE (accepts (&config));

  /* The 17-channel heatmap alone is refused, with offsets fit for 14 keypoints */
  config.info.info[1].dimension[0] = 2 * DEFAULT_LABELS;
  EXPECT_FALSE (accepts (&config));
  config.info.info[1].dimension[0] = 2 * POSENET_LABELS;

  /* The same tensors are taken with the 17 posenet keypoints */
  EXPECT_TRUE (setLabels (POSENET_LABEL_TEXT));
  EXPECT_TRUE (accepts (&config));
  gst_tensors_config_free (&config);
}

/**
 * @brief Two connected keypoints are drawn with the line between them.
 */
TEST (tensorDecoderPosePipeline, drawConnection)
{
  float heatmap[HEATMAP_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  /* heatmap-only scales the grid by option1 / option2: 40 / 5 = 8 pixels a cell */
  setHeatmap (heatmap, 0, 1, 1, 1.0f);
  setHeatmap (heatmap, 1, 3, 3, 1.0f);
  ASSERT_EQ (decodePose (heatmap, NULL, LABEL_TEXT, GRID, frame), GST_MESSAGE_EOS);

  /* The keypoints are at (8, 8) and (24, 24), the line passes (16, 16) */
  EXPECT_EQ (frame[24 * OUT_SIZE + 24], POSE_PIXEL);
  EXPECT_EQ (frame[16 * OUT_SIZE + 16], POSE_PIXEL);
}

/**
 * @brief A connection to an id past the last label is not drawn.
 * @details The id equal to the number of labels passed the bounds test and
 *          was looked up one past the end of the keypoint array.
 */
TEST (tensorDecoderPosePipeline, skipConnectionPastLastLabel_n)
{
  float heatmap[HEATMAP_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setHeatmap (heatmap, 0, 1, 1, 1.0f);
  setHeatmap (heatmap, 1, 3, 3, 1.0f);
  ASSERT_EQ (decodePose (heatmap, NULL, LABEL_TEXT_PAST_LAST, GRID, frame), GST_MESSAGE_EOS);

  /* The labels are drawn, the line is not */
  EXPECT_GT (countDrawnPixels (frame, OUT_PIXELS), 0U);
  EXPECT_EQ (frame[16 * OUT_SIZE + 16], 0U);
}

/**
 * @brief The offset tensor moves a keypoint within its heatmap cell.
 * @details The offsets of a keypoint are read at the heatmap cell it was
 *          found in, y first and x after the offsets of all the labels.
 */
TEST (tensorDecoderPosePipeline, drawWithOffset)
{
  float heatmap[HEATMAP_ELEMENTS];
  float offset[OFFSET_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };
  guint i;

  for (i = 0; i < HEATMAP_ELEMENTS; i++)
    heatmap[i] = -10.0f;
  setHeatmap (heatmap, 0, 1, 1, 10.0f);
  setHeatmap (heatmap, 1, 2, 2, 10.0f);

  /* The cell (1, 1) holds label 0's y offset at 0 and its x offset at LABELS */
  offset[(1 * GRID + 1) * 2 * LABELS] = 3.0f;
  offset[(1 * GRID + 1) * 2 * LABELS + LABELS] = 2.0f;

  /* heatmap-offset scales the grid to option2: cell 1 of 5 is at 40 / 4 = 10 */
  ASSERT_EQ (decodePose (heatmap, offset, LABEL_TEXT, OUT_SIZE, frame), GST_MESSAGE_EOS);

  /* The first keypoint moves from (10, 10) to (12, 13), the second stays at (20, 20) */
  EXPECT_EQ (frame[13 * OUT_SIZE + 12], POSE_PIXEL);
  EXPECT_EQ (frame[20 * OUT_SIZE + 20], POSE_PIXEL);
  EXPECT_EQ (frame[10 * OUT_SIZE + 10], 0U);
}

/**
 * @brief A pipeline without the model input size fails instead of dividing by it.
 */
TEST (tensorDecoderPosePipeline, refuseMissingModelSize_n)
{
  float heatmap[HEATMAP_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setHeatmap (heatmap, 0, 1, 1, 1.0f);
  setHeatmap (heatmap, 1, 3, 3, 1.0f);
  EXPECT_EQ (decodePose (heatmap, NULL, LABEL_TEXT, 0, frame), GST_MESSAGE_ERROR);
}

/**
 * @brief A label file without a label is refused by the loader image_labeling shares.
 * @details loadImageLabels () serves image_labeling, bounding_boxes and
 *          tensor_region. Reading a file of 0 bytes succeeds without an error
 *          to report, and the error message used to be read from the NULL error.
 */
TEST (tensorDecoderLabelFile, refuseEmptyLabelFile_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("image_labeling");
  gchar *label_file = getTempFilename ();
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  ASSERT_TRUE (label_file != NULL);
  ASSERT_TRUE (g_file_set_contents (label_file, "", 0, NULL));
  ASSERT_TRUE (decoder->init (&pdata));

  EXPECT_FALSE (decoder->setOption (&pdata, 0, label_file));

  EXPECT_TRUE (g_file_set_contents (label_file, "\n", -1, NULL));
  EXPECT_FALSE (decoder->setOption (&pdata, 0, label_file));

  /* The decoder still takes a label file after refusing one */
  EXPECT_TRUE (g_file_set_contents (label_file, "orange\n", -1, NULL));
  EXPECT_TRUE (decoder->setOption (&pdata, 0, label_file));

  decoder->exit (&pdata);
  removeTempFile (&label_file);
}

/**
 * @brief Each option pose_estimation parses is described, and option4 names the modes it takes.
 */
TEST (tensorDecoderPoseOptions, describeOptions)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("pose_estimation");
  const gchar *modes[] = { "heatmap-only", "heatmap-offset" };
  GData *desc_list;
  const gchar *desc;
  void *pdata = NULL;
  guint i;

  ASSERT_TRUE (decoder != NULL);
  desc_list = subplugin_get_custom_property_desc (NNS_SUBPLUGIN_DECODER, "pose_estimation");
  ASSERT_TRUE (desc_list != NULL);

  for (i = 1; i <= 4; i++) {
    g_autofree gchar *prop = g_strdup_printf ("option%u", i);

    desc = (const gchar *) g_datalist_get_data (&desc_list, prop);
    ASSERT_TRUE (desc != NULL) << prop;
    EXPECT_GT (strlen (desc), 0U) << prop;
  }

  desc = (const gchar *) g_datalist_get_data (&desc_list, "option4");
  ASSERT_TRUE (decoder->init (&pdata));
  for (i = 0; i < G_N_ELEMENTS (modes); i++) {
    EXPECT_TRUE (strstr (desc, modes[i]) != NULL) << modes[i];
    EXPECT_TRUE (decoder->setOption (&pdata, 3, modes[i])) << modes[i];
  }
  decoder->exit (&pdata);
}

/**
 * @brief pose_estimation describes no option it ignores, and refuses a mode option4 does not name.
 */
TEST (tensorDecoderPoseOptions, describeIgnoredOption_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("pose_estimation");
  GData *desc_list;
  void *pdata = NULL;
  guint i;

  ASSERT_TRUE (decoder != NULL);
  desc_list = subplugin_get_custom_property_desc (NNS_SUBPLUGIN_DECODER, "pose_estimation");
  ASSERT_TRUE (desc_list != NULL);

  for (i = 5; i <= 9; i++) {
    g_autofree gchar *prop = g_strdup_printf ("option%u", i);

    EXPECT_TRUE (g_datalist_get_data (&desc_list, prop) == NULL) << prop;
  }

  ASSERT_TRUE (decoder->init (&pdata));
  EXPECT_FALSE (decoder->setOption (&pdata, 3, "heatmap"));
  decoder->exit (&pdata);
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
