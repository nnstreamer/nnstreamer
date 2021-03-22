/**
 * @file        unittest_decoder.cc
 * @date        22 Mar 2021
 * @brief       Unit test for tensor_decoder
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include <tensor_decoder_custom.h>
#include <flatbuffers/flexbuffers.h>

#define TEST_TIMEOUT_MS (1000U)

static int data_received;

/**
 * @brief custom callback function
 */
int tensor_decoder_custom_cb (const GstTensorMemory *input,
    const GstTensorsConfig *config, void * data, GstBuffer *out_buf)
{
  GstMapInfo out_info;
  GstMemory *out_mem;
  unsigned int i, num_tensors = config->info.num_tensors;
  gboolean need_alloc;
  size_t flex_size;
  flexbuffers::Builder fbb;

  data_received++;
  fbb.Map ([&]() {
    fbb.UInt ("num_tensors", num_tensors);
    fbb.Int ("rate_n", config->rate_n);
    fbb.Int ("rate_d", config->rate_d);
    for (i = 0; i < num_tensors; i++) {
      gchar * tensor_key = g_strdup_printf ("tensor_%d", i);
      gchar * tensor_name = NULL;

      if (config->info.info[i].name == NULL) {
        tensor_name = g_strdup ("");
      } else {
        tensor_name = g_strdup (config->info.info[i].name);
      }
      tensor_type type = config->info.info[i].type;

      fbb.Vector (tensor_key, [&] () {
        fbb += tensor_name;
        fbb += type;
        fbb.Vector (config->info.info[i].dimension, NNS_TENSOR_RANK_LIMIT);
        fbb.Blob (input[i].data, (size_t) input[i].size);
      });
      g_free (tensor_key);
      g_free (tensor_name);
    }
  });
  fbb.Finish ();
  flex_size = fbb.GetSize ();

  g_assert (out_buf);
  need_alloc = (gst_buffer_get_size (out_buf) == 0);

  if (need_alloc) {
    out_mem = gst_allocator_alloc (NULL, flex_size, NULL);
  } else {
    if (gst_buffer_get_size (out_buf) < flex_size) {
      gst_buffer_set_size (out_buf, flex_size);
    }
    out_mem = gst_buffer_get_all_memory (out_buf);
  }

  if (FALSE == gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    if (need_alloc)
      gst_allocator_free (NULL, out_mem);
    nns_loge ("Cannot map gst memory (tensor decoder custom)\n");
    return GST_FLOW_ERROR;
  }

  memcpy (out_info.data, fbb.GetBuffer ().data (), flex_size);

  gst_memory_unmap (out_mem, &out_info);

  if (need_alloc)
    gst_buffer_append_memory (out_buf, out_mem);

  return GST_FLOW_OK;
}

/**
 * @brief Test behavior: custom callback
 */
TEST (tensorDecoderCustom, normal0)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1, len2;
  char *tmp_flex_default = getTempFilename ();
  char *tmp_flex_custom = getTempFilename ();

  EXPECT_NE (tmp_flex_default, nullptr);
  EXPECT_NE (tmp_flex_custom, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=12 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=320,height=240 ! tensor_converter ! tee name=t "
      "t. ! queue ! tensor_decoder mode=flexbuf ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_decoder mode=custom-code option1=tdec ! filesink location=%s buffer-mode=unbuffered sync=false async=false ",
      tmp_flex_custom, tmp_flex_default);

  EXPECT_EQ (0, nnstreamer_decoder_custom_register ("tdec", tensor_decoder_custom_cb, NULL));

  data_received = 0;

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  _wait_pipeline_save_files (tmp_flex_default, content1, len1, 230522, TEST_TIMEOUT_MS);
  _wait_pipeline_save_files (tmp_flex_custom, content2, len2, 230522, TEST_TIMEOUT_MS);
  EXPECT_EQ (len1, len2);
  EXPECT_EQ (memcmp (content1, content2, len1), 0);
  g_free (content1);
  g_free (content2);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (1, data_received);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec"));

  gst_object_unref (pipeline);
  g_free (str_pipeline);
  g_free (tmp_flex_default);
  g_free (tmp_flex_custom);
}

/**
 * @brief Register custom callback with NULL parameter
 */
TEST (tensorDecoderCustom, invalidParam0_n)
{
  EXPECT_NE (0, nnstreamer_decoder_custom_register (NULL, tensor_decoder_custom_cb, NULL));
  EXPECT_NE (0, nnstreamer_decoder_custom_register ("tdec", NULL, NULL));
}

/**
 * @brief Register custom callback twice with same name
 */
TEST (tensorDecoderCustom, invalidParam1_n)
{
  EXPECT_EQ (0, nnstreamer_decoder_custom_register ("tdec", tensor_decoder_custom_cb, NULL));
  EXPECT_NE (0, nnstreamer_decoder_custom_register ("tdec", tensor_decoder_custom_cb, NULL));
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec"));
}

/**
 * @brief Unregister custom callback with NULL parameter
 */
TEST (tensorDecoderCustom, invalidParam2_n)
{
  EXPECT_NE (0, nnstreamer_decoder_custom_unregister (NULL));
}

/**
 * @brief Unregister custom callback which is not registered
 */
TEST (tensorDecoderCustom, invalidParam3_n)
{
  EXPECT_NE (0, nnstreamer_decoder_custom_unregister ("tdec"));
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
