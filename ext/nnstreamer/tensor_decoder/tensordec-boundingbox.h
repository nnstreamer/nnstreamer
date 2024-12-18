/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "bounding boxes"
 * Copyright (C) 2018 Samsung Electronics Co. Ltd.
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 * Copyright 2021 NXP
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file        tensordec-boundingbox.h
 * @date        15 Nov 2018
 * @brief       NNStreamer tensor-decoder subplugin, "bounding boxes",
 *              which converts tensors to video stream w/ boxes on
 *              transparent background.
 *              This code is NYI/WIP and not compilable.
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 * option1: Decoder mode of bounding box.
 *          Available: yolov5
 *                     mobilenet-ssd (single shot multibox detector with priors.)
 *                     mobilenet-ssd-postprocess
 *                     ov-person-detection
 *                     tf-ssd (deprecated, recommend to use mobilenet-ssd-postprocess)
 *                     tflite-ssd (deprecated, recommend to use mobilenet-ssd)
 * option2: Location of label file
 *          This is independent from option1
 * option3: Any option1-dependent values
 *          !!This depends on option1 values!!
 *          for yolov5 and yolov8 mode:
 *            The option3 requires up to 3 numbers, which tell
 *              - whether the output values are scaled or not
 *                0: not scaled (default), 1: scaled (e.g., 0.0 ~ 1.0)
 *              - the threshold of confidence (optional, default set to 0.25)
 *              - the threshold of IOU (optional, default set to 0.45)
 *            An example of option3 is "option3=0:0.65:0.6"
 *          for mobilenet-ssd mode:
 *            The option3 definition scheme is, in order, the following:
 *                - box priors location file (mandatory)
 *                - Detection threshold (optional, default set to 0.5)
 *                - Y box scale (optional, default set to 10.0)
 *                - X box scale (optional, default set to 10.0)
 *                - h box scale (optional, default set to 5.0)
 *                - w box scale (optional, default set to 5.0)
 *                - IOU box valid threshold (optional, default set to 0.5)
 *            The default parameters value could be set in the following ways:
 *            option3=box-priors.txt:0.5:10.0:10.0:5.0:5.0:0.5
 *            option3=box-priors.txt
 *            option3=box-priors.txt::::::
 *
 *            It's possible to set only few values, using the default values for
 *            those not specified through the command line.
 *            You could specify respectively the detection and IOU thresholds to 0.65
 *            and 0.6 with the option3 parameter as follow:
 *            option3=box-priors.txt:0.65:::::0.6
 *          for mobilenet-ssd-postprocess mode:
 *            The option3 is required to have 5 integer numbers, which tell
 *            the tensor-dec how to interpret the given tensor inputs.
 *            The first 4 numbers separated by colon, ':', designate which
 *            are location:class:score:number of the tensors.
 *            The last number separated by comma, ',' from the first 4 numbers
 *            designate the threshold in percent.
 *            In other words, "option3=%i:%i:%i:%i,%i".
 *          for mp-palm-detection mode:
 *            The option3 is required to have 5 float numbers, as following
 *                - box score threshold (mandatory)
 *                - number of layers for anchor generation (optional, default set to 4)
 *                - minimum scale factor for anchor generation (optional, default set to 1.0)
 *                - maximum scale factor for anchor generation (optional, default set to 1.0)
 *                - X offset (optional, default set to 0.5)
 *                - Y offset (optional, default set to 0.5)
 *                - strides for each layer for anchor generation (optional, default set to 8:16:16:16)
 *            The default parameter value could be set in the following ways:
 *            option3=0.5
 *            option3=0.5:4:0.2:0.8
 *            option3=0.5:4:1.0:1.0:0.5:0.5:8:16:16:16
 *
 * option4: Video Output Dimension (WIDTH:HEIGHT)
 *          This is independent from option1
 * option5: Input Dimension (WIDTH:HEIGHT)
 *          This is independent from option1
 * option6: Whether to track result bounding boxes or not
 *          0 (default, do not track)
 *          1 (track result bounding boxes, with naive centroid based algorithm)
 * option7: Whether to log the result bounding boxes or not
 *          0 (default, do not log)
 *          1 (log result bounding boxes)
 * option8: Box Style (NYI)
 *
 * MAJOR TODO: Support other colorspaces natively from _decode for performance gain
 * (e.g., BGRA, ARGB, ...)
 *
 */

#ifndef _TENSORDECBB_H__
#define _TENSORDECBB_H__
#include <gst/gst.h>
#include <math.h> /* expf */
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include <nnstreamer_plugin_api_util.h>
#include "tensordecutil.h"

#define PIXEL_VALUE (0xFF0000FF) /* RED 100% in RGBA */

/**
 * @brief Option of bounding box
 */
enum class BoundingBoxOption {
  MODE = 0,
  LABEL_PATH = 1,
  INTERNAL = 2,
  VIDEO_SIZE = 3,
  INPUT_MODEL_SIZE = 4,
  TRACK = 5,
  LOG = 6,
  UNKNOWN,
};

/**
 * @brief There can be different schemes for bounding boxes.
 */
typedef enum {
  MOBILENET_SSD_BOUNDING_BOX = 0,
  MOBILENET_SSD_PP_BOUNDING_BOX = 1,
  OV_PERSON_DETECTION_BOUNDING_BOX = 2,
  OV_FACE_DETECTION_BOUNDING_BOX = 3,

  /* the modes started with 'OLDNAME_' is for backward compatibility. */
  OLDNAME_MOBILENET_SSD_BOUNDING_BOX = 4,
  OLDNAME_MOBILENET_SSD_PP_BOUNDING_BOX = 5,

  YOLOV5_BOUNDING_BOX = 6,

  MP_PALM_DETECTION_BOUNDING_BOX = 7,

  YOLOV8_BOUNDING_BOX = 8,

  BOUNDING_BOX_UNKNOWN,
} bounding_box_modes;

/**
 * @brief Structure for object centroid tracking.
 */
typedef struct {
  guint id;
  guint matched_box_idx;
  gint cx;
  gint cy;
  guint consecutive_disappeared_frames;
} centroid;

/**
 * @brief Structure for distances. {distance} : {centroids} x {boxes}
 */
typedef struct {
  guint centroid_idx;
  guint box_idx;
  guint64 distance;
} distanceArrayData;

/**
 * @brief anchor data
 */
typedef struct {
  float x_center;
  float y_center;
  float w;
  float h;
} anchor;

/** @brief Represents a detect object */
typedef struct {
  int valid;
  int class_id;
  int x;
  int y;
  int width;
  int height;
  gfloat prob;

  int tracking_id;
} detectedObject;


/**
 * @brief Apply NMS to the given results (objects[DETECTION_MAX])
 * @param[in/out] results The results to be filtered with nms
 */
void nms (GArray *results, gfloat threshold);

/**
 * @brief check the num_tensors is valid
 * @param[in] config The structure of tensors info to check.
 * @param[in] limit The limit of tensors number.
 * @return TRUE if tensors info is valid.
 */
int check_tensors (const GstTensorsConfig *config, const unsigned int limit);

/**
 * @brief	Interface for Bounding box's properties
 */
class BoxProperties
{
  public:
  virtual ~BoxProperties () = default;

  /* mandatory methods */
  virtual int setOptionInternal (const char *param) = 0;
  virtual int checkCompatible (const GstTensorsConfig *config) = 0;
  virtual GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input) = 0;

  void setInputWidth (guint width)
  {
    i_width = width;
  }
  void setInputHeight (guint height)
  {
    i_height = height;
  }
  void setTotalLabels (guint labels)
  {
    total_labels = labels;
  }

  guint getInputWidth ()
  {
    return i_width;
  }
  guint getInputHeight ()
  {
    return i_height;
  }
  gchar *name;

  protected:
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */

  guint max_detection;
  guint total_labels;
};

/**
 * @brief	Class for Bounding box tensor decoder
 */
class BoundingBox
{
  public:
  BoundingBox ();
  ~BoundingBox ();

  gboolean checkLabelProps ();
  int setBoxDecodingMode (const char *param);
  int setLabelPath (const char *param);
  int setVideoSize (const char *param);
  int setInputModelSize (const char *param);
  void draw (GstMapInfo *out_info, GArray *results);
  void logBoxes (GArray *results);
  void updateCentroids (GArray *boxes);

  int setOption (BoundingBoxOption opNum, const char *param);
  GstCaps *getOutCaps (const GstTensorsConfig *config);
  GstFlowReturn decode (const GstTensorsConfig *config,
      const GstTensorMemory *input, GstBuffer *outbuf);

  static BoxProperties *getProperties (const gchar *properties_name);
  static gboolean addProperties (BoxProperties *boxProperties);

  private:
  bounding_box_modes mode;
  BoxProperties *bdata;

  /* From option2 */
  imglabel_t labeldata;
  char *label_path;

  /* From option4 */
  guint width; /**< Output Video Width */
  guint height; /**< Output Video Height */

  /* From option6 (track or not) */
  gint is_track;
  guint centroids_last_id; /**< The last_id of centroid valid id is 1, 2, ... (not 0). */
  guint max_centroids_num; /**< The maximum number of centroids */
  guint consecutive_disappear_threshold; /**< The threshold of consecutive disappeared frames */

  GArray *centroids; /**< Array for centroids */
  GArray *distanceArray; /**< Array for distances */

  /* From option7 (log or not) */
  gint do_log;

  gboolean flag_use_label;

  /* Table for box properties data */
  inline static GHashTable *properties_table;
};

/**
 * @brief Apply NMS to the given results (objects[DETECTION_MAX])
 * @param[in/out] results The results to be filtered with nms
 */
void nms (GArray *results, gfloat threshold, bounding_box_modes mode = BOUNDING_BOX_UNKNOWN);

#endif /* _TENSORDECBB_H__ */
