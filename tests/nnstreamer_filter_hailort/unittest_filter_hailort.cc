/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_hailort.cc
 * @date    28 Oct 2026
 * @brief   Unit tests of the HailoRT tensor_filter sub-plugin on a mock HailoRT
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The sub-plugin source is linked into this binary against the mock in
 * mock/hailort_mock.cc, which reads a text model description instead of a
 * compiled .hef and needs no Hailo device. See mock/hailort_mock.h.
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>
#include <unittest_util.h>

#include <hailort_mock.h>

/** @brief A model of one uint8 4:4:3 input and the same output. */
static const gchar *IDENTITY_MODEL = "hailort-mock-hef\n"
                                     "network_group identity\n"
                                     "input in0 uint8 nhwc 4 4 3\n"
                                     "output out0 uint8 nhwc 4 4 3\n";

/** @brief A model of two inputs and three outputs of every layout. */
static const gchar *MULTI_MODEL = "hailort-mock-hef\n"
                                  "# two inputs, three outputs\n"
                                  "network_group multi\n"
                                  "input image uint8 nhwc 2 2 1\n"
                                  "input extra uint16 nhwc 1 1 4\n"
                                  "output scores float32 nhwc 1 1 4\n"
                                  "output planar uint8 nchw 2 3 1\n"
                                  "output boxes float32 nms 1 2 5\n";

/**
 * @brief Reset the mock around every case, so that an injected failure
 *        never reaches a case of another suite whatever the run order is.
 */
class hailort_mock_reset_listener : public testing::EmptyTestEventListener
{
  /** @brief Reset before a case. */
  void OnTestStart (const testing::TestInfo &) override
  {
    hailort_mock_reset ();
  }

  /** @brief Reset after a case. */
  void OnTestEnd (const testing::TestInfo &) override
  {
    hailort_mock_reset ();
  }
};

/**
 * @brief Fixture that writes mock models into a private directory.
 */
class nnstreamerFilterHailort : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  gchar *dir;
  void *data;
  GstTensorFilterProperties prop;
  const gchar *model_files[2];
  gchar *model;

  /** @brief Find the sub-plugin and create the model directory. */
  void SetUp () override
  {
    sp = nnstreamer_filter_find ("hailort");
    ASSERT_TRUE (sp != nullptr);
    dir = g_dir_make_tmp ("nns-hailort-XXXXXX", NULL);
    ASSERT_TRUE (dir != nullptr);
    data = nullptr;
    model = nullptr;
    live_before = hailort_mock_live_objects ();
  }

  /** @brief Close the filter, remove the models and check for leaks. */
  void TearDown () override
  {
    if (data)
      sp->close (&prop, &data);
    if (model) {
      g_remove (model);
      g_free (model);
    }
    g_rmdir (dir);
    g_free (dir);
    EXPECT_EQ (hailort_mock_live_objects (), live_before);
  }

  /** @brief Write a model and set the properties to use it. */
  void setModel (const gchar *content, const gchar *custom = nullptr)
  {
    if (model) {
      g_remove (model);
      g_free (model);
    }
    model = g_build_filename (dir, "model.hef", NULL);
    ASSERT_TRUE (g_file_set_contents (model, content, -1, NULL));

    memset (&prop, 0, sizeof (prop));
    prop.fwname = "hailort";
    model_files[0] = model;
    model_files[1] = nullptr;
    prop.model_files = model_files;
    prop.num_models = 1;
    prop.custom_properties = custom;
  }

  /** @brief Open the filter with the current properties. */
  int open ()
  {
    return sp->open (&prop, &data);
  }

  /** @brief Get the model information of the opened filter. */
  int modelInfo (GstTensorsInfo *in, GstTensorsInfo *out)
  {
    return sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, in, out);
  }

  /** @brief Invoke the opened filter. */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    return sp->invoke (sp, &prop, data, in, out);
  }

  private:
  int live_before;
};

/**
 * @brief Positive case: the framework information.
 */
TEST_F (nnstreamerFilterHailort, frameworkInfo)
{
  GstTensorFilterFrameworkInfo info;

  memset (&info, 0, sizeof (info));
  EXPECT_EQ (sp->getFrameworkInfo (sp, NULL, NULL, &info), 0);
  EXPECT_STREQ (info.name, "hailort");
  EXPECT_FALSE (info.run_without_model);
  EXPECT_TRUE (info.verify_model_path);
  EXPECT_FALSE (info.allocate_in_invoke);
  ASSERT_EQ (info.num_hw, 1);
  EXPECT_EQ (info.hw_list[0], ACCL_NPU);
}

/**
 * @brief Positive case: open a model, describe it and release every object.
 */
TEST_F (nnstreamerFilterHailort, openClose)
{
  GstTensorsInfo in, out;
  gchar *group_id;

  setModel (IDENTITY_MODEL);
  ASSERT_EQ (open (), 0);
  EXPECT_GT (hailort_mock_live_objects (), 0);

  gst_tensors_info_init (&in);
  gst_tensors_info_init (&out);
  ASSERT_EQ (modelInfo (&in, &out), 0);

  ASSERT_EQ (in.num_tensors, 1U);
  EXPECT_EQ (in.info[0].type, _NNS_UINT8);
  EXPECT_STREQ (in.info[0].name, "in0");
  EXPECT_EQ (in.info[0].dimension[0], 3U);
  EXPECT_EQ (in.info[0].dimension[1], 4U);
  EXPECT_EQ (in.info[0].dimension[2], 4U);
  EXPECT_EQ (in.info[0].dimension[3], 1U);
  ASSERT_EQ (out.num_tensors, 1U);
  EXPECT_EQ (out.info[0].type, _NNS_UINT8);
  EXPECT_STREQ (out.info[0].name, "out0");
  EXPECT_EQ (gst_tensor_info_get_size (&out.info[0]), 48U);

  /* Without group_id each filter gets a virtual device of its own. */
  group_id = hailort_mock_last_group_id ();
  EXPECT_STREQ (group_id, "UNIQUE");
  g_free (group_id);

  gst_tensors_info_free (&in);
  gst_tensors_info_free (&out);

  sp->close (&prop, &data);
  EXPECT_EQ (hailort_mock_live_objects (), 0);
  EXPECT_EQ (hailort_mock_calls ("hailo_release_vdevice"), 1);
  EXPECT_EQ (hailort_mock_calls ("hailo_release_hef"), 1);
  EXPECT_EQ (hailort_mock_calls ("hailo_release_input_vstreams"), 1);
  EXPECT_EQ (hailort_mock_calls ("hailo_release_output_vstreams"), 1);
}

/**
 * @brief Positive case: the output equals the input of the identity model.
 */
TEST_F (nnstreamerFilterHailort, invokeIdentity)
{
  guint8 in_data[48], out_data[48];
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out = { out_data, sizeof (out_data) };

  setModel (IDENTITY_MODEL);
  ASSERT_EQ (open (), 0);

  for (int frame = 0; frame < 3; frame++) {
    for (guint i = 0; i < sizeof (in_data); i++)
      in_data[i] = (guint8) (i * 3 + frame);
    memset (out_data, 0, sizeof (out_data));

    ASSERT_EQ (invoke (&in, &out), 0);
    EXPECT_EQ (memcmp (in_data, out_data, sizeof (in_data)), 0);
  }

  EXPECT_EQ (hailort_mock_calls ("hailo_vstream_write_raw_buffer"), 3);
  EXPECT_EQ (hailort_mock_calls ("hailo_vstream_read_raw_buffer"), 3);
}

/**
 * @brief Positive case: float32 output asks HailoRT to dequantize on the host.
 */
TEST_F (nnstreamerFilterHailort, invokeFloatOutput)
{
  GstTensorsInfo in_info, out_info;
  guint8 in_data[48];
  gfloat out_data[48];
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out = { out_data, sizeof (out_data) };

  setModel (IDENTITY_MODEL, "output_format:float32");
  ASSERT_EQ (open (), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (modelInfo (&in_info, &out_info), 0);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (out_info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (gst_tensor_info_get_size (&out_info.info[0]), sizeof (out_data));
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);

  for (guint i = 0; i < sizeof (in_data); i++)
    in_data[i] = (guint8) (255 - i);

  ASSERT_EQ (invoke (&in, &out), 0);
  for (guint i = 0; i < sizeof (in_data); i++)
    EXPECT_FLOAT_EQ (out_data[i], (gfloat) (255 - i));
}

/**
 * @brief Positive case: several inputs and outputs of every layout.
 */
TEST_F (nnstreamerFilterHailort, multipleInputsOutputs)
{
  GstTensorsInfo in_info, out_info;
  guint8 image[4] = { 10, 20, 30, 40 };
  guint16 extra[4] = { 1, 2, 3, 4 };
  gfloat scores[4];
  guint8 planar[6];
  gfloat boxes[10];
  GstTensorMemory in[2] = { { image, sizeof (image) }, { extra, sizeof (extra) } };
  GstTensorMemory out[3] = { { scores, sizeof (scores) },
    { planar, sizeof (planar) }, { boxes, sizeof (boxes) } };

  setModel (MULTI_MODEL);
  ASSERT_EQ (open (), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (modelInfo (&in_info, &out_info), 0);

  ASSERT_EQ (in_info.num_tensors, 2U);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (in_info.info[0].dimension[0], 1U);
  EXPECT_EQ (in_info.info[0].dimension[1], 2U);
  EXPECT_EQ (in_info.info[0].dimension[2], 2U);
  EXPECT_EQ (in_info.info[1].type, _NNS_UINT16);
  EXPECT_EQ (in_info.info[1].dimension[0], 4U);
  EXPECT_STREQ (in_info.info[1].name, "extra");

  ASSERT_EQ (out_info.num_tensors, 3U);
  EXPECT_EQ (out_info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (out_info.info[0].dimension[0], 4U);
  EXPECT_EQ (out_info.info[0].dimension[1], 1U);
  /* A layout other than NHWC is passed as a flat tensor of the frame size. */
  EXPECT_EQ (out_info.info[1].type, _NNS_UINT8);
  EXPECT_EQ (out_info.info[1].dimension[0], 6U);
  EXPECT_EQ (out_info.info[1].dimension[1], 0U);
  EXPECT_EQ (out_info.info[2].type, _NNS_FLOAT32);
  EXPECT_EQ (out_info.info[2].dimension[0], 10U);
  EXPECT_EQ (out_info.info[2].dimension[1], 0U);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);

  ASSERT_EQ (invoke (in, out), 0);
  for (guint j = 0; j < 4; j++)
    EXPECT_FLOAT_EQ (scores[j], (gfloat) image[j]);
  for (guint j = 0; j < 6; j++)
    EXPECT_EQ (planar[j], image[j % 4] + 1);
  for (guint j = 0; j < 10; j++)
    EXPECT_FLOAT_EQ (boxes[j], (gfloat) (image[j % 4] + 2));
}

/**
 * @brief Positive case: NHW and NC frames are described like NHWC ones.
 */
TEST_F (nnstreamerFilterHailort, nhwAndNcLayouts)
{
  GstTensorsInfo in_info, out_info;

  setModel ("hailort-mock-hef\n"
            "network_group g\n"
            "input mask uint8 nhw 2 3 1\n"
            "output logits uint16 nc 1 1 5\n");
  ASSERT_EQ (open (), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (modelInfo (&in_info, &out_info), 0);
  EXPECT_EQ (in_info.info[0].dimension[0], 1U);
  EXPECT_EQ (in_info.info[0].dimension[1], 3U);
  EXPECT_EQ (in_info.info[0].dimension[2], 2U);
  EXPECT_EQ (in_info.info[0].dimension[3], 1U);
  EXPECT_EQ (out_info.info[0].type, _NNS_UINT16);
  EXPECT_EQ (out_info.info[0].dimension[0], 5U);
  EXPECT_EQ (out_info.info[0].dimension[1], 1U);
  EXPECT_EQ (out_info.info[0].dimension[2], 1U);
  EXPECT_EQ (out_info.info[0].dimension[3], 1U);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Positive case: every custom property, with spaces around the words.
 */
TEST_F (nnstreamerFilterHailort, customProperties)
{
  GstTensorsInfo in_info, out_info;
  gchar *group_id;

  setModel (IDENTITY_MODEL, " input_format : uint16 , output_format:UINT8,group_id:shared:0 ");
  ASSERT_EQ (open (), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (modelInfo (&in_info, &out_info), 0);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT16);
  EXPECT_EQ (gst_tensor_info_get_size (&in_info.info[0]), 96U);
  EXPECT_EQ (out_info.info[0].type, _NNS_UINT8);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);

  group_id = hailort_mock_last_group_id ();
  EXPECT_STREQ (group_id, "shared:0");
  g_free (group_id);
}

/**
 * @brief Positive case: auto and an empty custom property keep the device types.
 */
TEST_F (nnstreamerFilterHailort, customPropertiesAuto)
{
  GstTensorsInfo in_info, out_info;

  setModel (MULTI_MODEL, "input_format:auto,output_format:auto");
  ASSERT_EQ (open (), 0);
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (modelInfo (&in_info, &out_info), 0);
  EXPECT_EQ (in_info.info[1].type, _NNS_UINT16);
  EXPECT_EQ (out_info.info[1].type, _NNS_UINT8);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  sp->close (&prop, &data);

  setModel (MULTI_MODEL, "");
  EXPECT_EQ (open (), 0);
}

/**
 * @brief Positive case: an operation the sub-plugin does not implement.
 */
TEST_F (nnstreamerFilterHailort, unsupportedOperations)
{
  GstTensorsInfo in_info, out_info;
  GstTensorFilterFrameworkEventData event;

  setModel (IDENTITY_MODEL);
  ASSERT_EQ (open (), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  EXPECT_EQ (sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in_info, &out_info), -ENOENT);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);

  memset (&event, 0, sizeof (event));
  EXPECT_EQ (sp->eventHandler (sp, &prop, data, RELOAD_MODEL, &event), -ENOENT);
}

/**
 * @brief Negative case: no model path.
 */
TEST_F (nnstreamerFilterHailort, noModel_n)
{
  setModel (IDENTITY_MODEL);

  model_files[0] = "";
  EXPECT_NE (open (), 0);

  /* Only the first of two models would be used. */
  model_files[0] = model;
  prop.num_models = 2;
  EXPECT_NE (open (), 0);

  prop.model_files = nullptr;
  prop.num_models = 0;
  EXPECT_NE (open (), 0);
  EXPECT_EQ (hailort_mock_calls ("hailo_create_vdevice"), 0);
}

/**
 * @brief Negative case: invalid custom properties.
 */
TEST_F (nnstreamerFilterHailort, invalidCustomProperties_n)
{
  const gchar *invalid[] = { "unknown:1", "input_format", "input_format:int8",
    "output_format:float16", "group_id:", "output_format:float32,", nullptr };

  for (int i = 0; invalid[i]; i++) {
    SCOPED_TRACE (invalid[i]);
    setModel (IDENTITY_MODEL, invalid[i]);
    EXPECT_NE (open (), 0);
  }

  EXPECT_EQ (hailort_mock_calls ("hailo_create_vdevice"), 0);
}

/**
 * @brief Negative case: a file that is not a model, or does not exist.
 */
TEST_F (nnstreamerFilterHailort, invalidModel_n)
{
  gchar *missing;

  setModel ("not a model\n");
  EXPECT_NE (open (), 0);

  missing = g_build_filename (dir, "missing.hef", NULL);
  model_files[0] = missing;
  EXPECT_NE (open (), 0);
  g_free (missing);

  EXPECT_EQ (hailort_mock_calls ("hailo_create_hef_file"), 2);
  /* An invalid model does not occupy a device. */
  EXPECT_EQ (hailort_mock_calls ("hailo_create_vdevice"), 0);
  EXPECT_EQ (hailort_mock_live_objects (), 0);
}

/**
 * @brief Negative case: a model of two network groups.
 */
TEST_F (nnstreamerFilterHailort, twoNetworkGroups_n)
{
  setModel ("hailort-mock-hef\n"
            "network_group first\n"
            "network_group second\n"
            "input in0 uint8 nhwc 4 4 3\n"
            "output out0 uint8 nhwc 4 4 3\n");
  EXPECT_NE (open (), 0);
  EXPECT_EQ (hailort_mock_calls ("hailo_configure_vdevice"), 1);
  EXPECT_EQ (hailort_mock_calls ("hailo_make_input_vstream_params"), 0);
}

/**
 * @brief Negative case: a model without inputs, and one without outputs.
 */
TEST_F (nnstreamerFilterHailort, noInputOrOutput_n)
{
  setModel ("hailort-mock-hef\n"
            "network_group g\n"
            "output out0 uint8 nhwc 4 4 3\n");
  EXPECT_NE (open (), 0);

  setModel ("hailort-mock-hef\n"
            "network_group g\n"
            "input in0 uint8 nhwc 4 4 3\n");
  EXPECT_NE (open (), 0);
  EXPECT_EQ (hailort_mock_calls ("hailo_create_input_vstreams"), 0);
}

/**
 * @brief Negative case: every HailoRT call of open() fails in turn, and every
 *        object created before it is released.
 */
TEST_F (nnstreamerFilterHailort, apiFailures_n)
{
  const gchar *functions[] = { "hailo_init_vdevice_params", "hailo_create_vdevice",
    "hailo_create_hef_file", "hailo_init_configure_params_by_vdevice",
    "hailo_configure_vdevice", "hailo_make_input_vstream_params",
    "hailo_make_output_vstream_params", "hailo_create_input_vstreams",
    "hailo_create_output_vstreams", "hailo_get_input_vstream_info",
    "hailo_get_input_vstream_user_format", "hailo_get_input_vstream_frame_size",
    "hailo_get_output_vstream_info", "hailo_get_output_vstream_user_format",
    "hailo_get_output_vstream_frame_size", nullptr };

  setModel (MULTI_MODEL);

  for (int i = 0; functions[i]; i++) {
    SCOPED_TRACE (functions[i]);
    hailort_mock_reset ();
    hailort_mock_fail (functions[i], HAILO_INVALID_OPERATION);

    EXPECT_NE (open (), 0);
    EXPECT_EQ (hailort_mock_calls (functions[i]), 1);
    EXPECT_EQ (hailort_mock_live_objects (), 0);
  }
}

/**
 * @brief Negative case: a status HailoRT has no message for.
 */
TEST_F (nnstreamerFilterHailort, statusWithoutMessage_n)
{
  setModel (IDENTITY_MODEL);
  hailort_mock_fail ("hailo_create_vdevice", HAILO_OUT_OF_PHYSICAL_DEVICES);
  EXPECT_NE (open (), 0);
}

/**
 * @brief Negative case: a host-side type NNStreamer has no type for.
 */
TEST_F (nnstreamerFilterHailort, unsupportedType_n)
{
  setModel (IDENTITY_MODEL);
  hailort_mock_override_user_type ((hailo_format_type_t) 100);
  EXPECT_NE (open (), 0);
  EXPECT_EQ (hailort_mock_live_objects (), 0);
}

/**
 * @brief Negative case: frame sizes that cannot describe a tensor.
 */
TEST_F (nnstreamerFilterHailort, invalidFrameSize_n)
{
  /* Not a multiple of sizeof (float) */
  setModel (IDENTITY_MODEL, "input_format:float32");
  hailort_mock_override_frame_size (6);
  EXPECT_NE (open (), 0);

  setModel (IDENTITY_MODEL);
  hailort_mock_override_frame_size (0);
  EXPECT_NE (open (), 0);

  /* More elements than a dimension can hold */
  if (sizeof (size_t) > 4) {
    hailort_mock_override_frame_size ((size_t) G_MAXUINT32 + 1);
    EXPECT_NE (open (), 0);
  }
  EXPECT_EQ (hailort_mock_live_objects (), 0);
}

/**
 * @brief Positive case: a frame size that is not h * w * c becomes a flat tensor.
 */
TEST_F (nnstreamerFilterHailort, frameSizeMismatchShape)
{
  GstTensorsInfo in_info, out_info;

  setModel (IDENTITY_MODEL);
  hailort_mock_override_frame_size (50);
  ASSERT_EQ (open (), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (modelInfo (&in_info, &out_info), 0);
  EXPECT_EQ (in_info.info[0].dimension[0], 50U);
  EXPECT_EQ (in_info.info[0].dimension[1], 0U);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Negative case: invoke without buffers or with buffers of another size.
 */
TEST_F (nnstreamerFilterHailort, invokeInvalidBuffers_n)
{
  guint8 in_data[48] = { 0 }, out_data[48];
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out = { out_data, sizeof (out_data) };
  GstTensorMemory small_in = { in_data, sizeof (in_data) - 1 };
  GstTensorMemory small_out = { out_data, sizeof (out_data) - 1 };

  setModel (IDENTITY_MODEL);
  ASSERT_EQ (open (), 0);

  EXPECT_NE (invoke (NULL, &out), 0);
  EXPECT_NE (invoke (&in, NULL), 0);
  EXPECT_NE (invoke (&small_in, &out), 0);
  EXPECT_NE (invoke (&in, &small_out), 0);
  EXPECT_EQ (hailort_mock_calls ("hailo_vstream_write_raw_buffer"), 0);

  /* Nothing reached HailoRT, so the filter keeps working. */
  EXPECT_EQ (invoke (&in, &out), 0);
}

/**
 * @brief Negative case: a failed write makes later invokes fail until reopened.
 */
TEST_F (nnstreamerFilterHailort, invokeWriteFailure_n)
{
  guint8 in_data[48] = { 0 }, out_data[48];
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out = { out_data, sizeof (out_data) };

  setModel (IDENTITY_MODEL);
  ASSERT_EQ (open (), 0);

  hailort_mock_fail ("hailo_vstream_write_raw_buffer", HAILO_TIMEOUT);
  EXPECT_NE (invoke (&in, &out), 0);

  hailort_mock_reset ();
  EXPECT_NE (invoke (&in, &out), 0);
  EXPECT_EQ (hailort_mock_calls ("hailo_vstream_write_raw_buffer"), 0);

  sp->close (&prop, &data);
  ASSERT_EQ (open (), 0);
  EXPECT_EQ (invoke (&in, &out), 0);
}

/**
 * @brief Negative case: after a failed read, a later invoke must not return
 *        the frame the earlier one left queued.
 */
TEST_F (nnstreamerFilterHailort, invokeReadFailure_n)
{
  guint8 in_data[48], out_data[48];
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out = { out_data, sizeof (out_data) };

  setModel (IDENTITY_MODEL);
  ASSERT_EQ (open (), 0);

  memset (in_data, 1, sizeof (in_data));
  hailort_mock_fail ("hailo_vstream_read_raw_buffer", HAILO_TIMEOUT);
  EXPECT_NE (invoke (&in, &out), 0);

  hailort_mock_reset ();
  memset (in_data, 2, sizeof (in_data));
  memset (out_data, 0, sizeof (out_data));
  EXPECT_NE (invoke (&in, &out), 0);
  EXPECT_EQ (out_data[0], 0);
  EXPECT_EQ (hailort_mock_calls ("hailo_vstream_read_raw_buffer"), 0);
}

/**
 * @brief Build a tensor_filter harness on the given model.
 */
static GstHarness *
_HarnessNew (const gchar *model, const gchar *custom)
{
  GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);
  GstHarness *h;

  g_object_set (filter, "framework", "hailort", "model", model, NULL);
  if (custom)
    g_object_set (filter, "custom", custom, NULL);

  h = gst_harness_new_with_element (filter, "sink", "src");
  gst_object_unref (filter);
  return h;
}

/**
 * @brief Positive case: a pipeline with the identity model.
 */
TEST_F (nnstreamerFilterHailort, pipelineIdentity)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstMapInfo map;
  guint8 expected[48];

  setModel (IDENTITY_MODEL);
  h = _HarnessNew (model, NULL);
  gst_harness_set_src_caps_str (h, "other/tensors,num_tensors=1,format=static,"
                                   "dimensions=3:4:4:1,types=uint8,framerate=0/1");

  for (guint i = 0; i < sizeof (expected); i++)
    expected[i] = (guint8) (200 - i);
  in_buf = gst_harness_create_buffer (h, sizeof (expected));
  gst_buffer_fill (in_buf, 0, expected, sizeof (expected));

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != nullptr);
  ASSERT_TRUE (gst_buffer_map (out_buf, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, sizeof (expected));
  EXPECT_EQ (memcmp (map.data, expected, sizeof (expected)), 0);
  gst_buffer_unmap (out_buf, &map);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
}

/**
 * @brief Positive case: the output caps of a float32 output.
 */
TEST_F (nnstreamerFilterHailort, pipelineFloatOutput)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstCaps *caps;
  GstTensorsConfig config;
  GstMapInfo map;
  guint8 values[48];

  setModel (IDENTITY_MODEL, "output_format:float32");
  h = _HarnessNew (model, "output_format:float32");
  gst_harness_set_src_caps_str (h, "other/tensors,num_tensors=1,format=static,"
                                   "dimensions=3:4:4:1,types=uint8,framerate=0/1");

  for (guint i = 0; i < sizeof (values); i++)
    values[i] = (guint8) i;
  in_buf = gst_harness_create_buffer (h, sizeof (values));
  gst_buffer_fill (in_buf, 0, values, sizeof (values));
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  caps = gst_pad_get_current_caps (h->sinkpad);
  ASSERT_TRUE (caps != nullptr);
  gst_tensors_config_init (&config);
  EXPECT_TRUE (gst_tensors_config_from_caps (&config, caps, TRUE));
  EXPECT_EQ (config.info.info[0].type, _NNS_FLOAT32);
  gst_tensors_config_free (&config);
  gst_caps_unref (caps);

  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != nullptr);
  ASSERT_TRUE (gst_buffer_map (out_buf, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, sizeof (values) * sizeof (gfloat));
  EXPECT_FLOAT_EQ (((gfloat *) map.data)[47], 47.0f);
  gst_buffer_unmap (out_buf, &map);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
}

/**
 * @brief Negative case: a pipeline with an invalid model does not start.
 */
TEST_F (nnstreamerFilterHailort, pipelineInvalidModel_n)
{
  GstElement *pipeline;
  gchar *description;

  setModel ("hailort-mock-hef\nnetwork_group a\nnetwork_group b\n");
  description = g_strdup_printf (
      "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=4,height=4 ! "
      "tensor_converter ! tensor_filter framework=hailort model=\"%s\" ! fakesink",
      model);
  pipeline = gst_parse_launch (description, NULL);
  g_free (description);
  ASSERT_TRUE (pipeline != nullptr);

  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
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

  testing::UnitTest::GetInstance ()->listeners ().Append (
      new hailort_mock_reset_listener ());
  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
