#ifndef _TENSORDECBB_H__
#define _TENSORDECBB_H__

#include <math.h> /* expf */
#include "tensordecutil.h"

#define PIXEL_VALUE (0xFF0000FF) /* RED 100% in RGBA */

/**
 * @brief C++-Template-like box location calculation for box-priors for Mobilenet SSD Model
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] boxprior The box prior data from the box file of MOBILENET_SSD.
 * @param[in] boxinput Input Tensor Data (Boxes)
 * @param[in] detinput Input Tensor Data (Detection). Null if not available. (numtensor ==1)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mobilenet_ssd(_type, typename, boxprior, boxinput,                                      \
    detinput, config, results, i_width, i_height, max_detection)                                             \
  case typename:                                                                                             \
    {                                                                                                        \
      int d;                                                                                                 \
      _type *boxinput_ = (_type *) boxinput;                                                                 \
      size_t boxbpi = config->info.info[0].dimension[0];                                                     \
      _type *detinput_ = (_type *) detinput;                                                                 \
      size_t detbpi = config->info.info[1].dimension[0];                                                     \
      int num = (DETECTION_MAX > max_detection) ? max_detection : DETECTION_MAX;                             \
      detectedObject object = {                                                                              \
        .valid = FALSE, .class_id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .prob = .0, .tracking_id = 0 \
      };                                                                                                     \
      for (d = 0; d < num; d++) {                                                                            \
        _get_object_i_mobilenet_ssd (d, detbpi, boxprior, (boxinput_ + (d * boxbpi)),                        \
            (detinput_ + (d * detbpi)), (&object), i_width, i_height);                                       \
        if (object.valid == TRUE) {                                                                          \
          g_array_append_val (results, object);                                                              \
        }                                                                                                    \
      }                                                                                                      \
    }                                                                                                        \
    break


/** @brief Macro to simplify calling _get_objects_mobilenet_ssd */
#define _get_objects_mobilenet_ssd_(type, typename)                      \
  _get_objects_mobilenet_ssd (type, typename, box_priors, (boxes->data), \
      (detections->data), config, results, i_width, i_height, max_detection)


#define _expit(x) (1.f / (1.f + expf (-((float) x))))

/**
 * @brief C++-Template-like box location calculation for box-priors
 * @bug This is not macro-argument safe. Use paranthesis!
 * @param[in] bb The configuration, "bounding_boxes"
 * @param[in] index The index (3rd dimension of BOX_SIZE:1:DETECTION_MAX:1)
 * @param[in] total_labels The count of total labels. We can get this from input tensor info. (1st dimension of LABEL_SIZE:DETECTION_MAX:1:1)
 * @param[in] boxprior The box prior data from the box file of SSD.
 * @param[in] boxinputptr Cursor pointer of input + byte-per-index * index (box)
 * @param[in] detinputptr Cursor pointer of input + byte-per-index * index (detection)
 * @param[in] result The object returned. (pointer to object)
 */
#define _get_object_i_mobilenet_ssd(index, total_labels, boxprior,                \
    boxinputptr, detinputptr, result, i_width, i_height)                          \
  do {                                                                            \
    unsigned int c;                                                               \
    gfloat highscore = -FLT_MAX;                                                  \
    float y_scale = params[Y_SCALE_IDX];                                          \
    float x_scale = params[X_SCALE_IDX];                                          \
    float h_scale = params[H_SCALE_IDX];                                          \
    float w_scale = params[W_SCALE_IDX];                                          \
    result->valid = FALSE;                                                        \
    for (c = 1; c < total_labels; c++) {                                          \
      if (detinputptr[c] >= sigmoid_threshold) {                                  \
        gfloat score = _expit (detinputptr[c]);                                   \
        float ycenter                                                             \
            = boxinputptr[0] / y_scale * boxprior[2][index] + boxprior[0][index]; \
        float xcenter                                                             \
            = boxinputptr[1] / x_scale * boxprior[3][index] + boxprior[1][index]; \
        float h = (float) expf (boxinputptr[2] / h_scale) * boxprior[2][index];   \
        float w = (float) expf (boxinputptr[3] / w_scale) * boxprior[3][index];   \
        float ymin = ycenter - h / 2.f;                                           \
        float xmin = xcenter - w / 2.f;                                           \
        int x = xmin * i_width;                                                   \
        int y = ymin * i_height;                                                  \
        int width = w * i_width;                                                  \
        int height = h * i_height;                                                \
        if (highscore < score) {                                                  \
          result->class_id = c;                                                   \
          result->x = MAX (0, x);                                                 \
          result->y = MAX (0, y);                                                 \
          result->width = width;                                                  \
          result->height = height;                                                \
          result->prob = score;                                                   \
          result->valid = TRUE;                                                   \
        }                                                                         \
      }                                                                           \
    }                                                                             \
  } while (0);

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
 * @brief MOBILENET SSD PostProcess Output tensor feature mapping.
 */
typedef enum {
  MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS = 0,
  MOBILENET_SSD_PP_BBOX_IDX_CLASSES = 1,
  MOBILENET_SSD_PP_BBOX_IDX_SCORES = 2,
  MOBILENET_SSD_PP_BBOX_IDX_NUM = 3,
  MOBILENET_SSD_PP_BBOX_IDX_UNKNOWN
} mobilenet_ssd_pp_bbox_idx_t;

/**
 * @brief List of bounding-box decoding schemes in string
 */
static const char *bb_modes[] = {
  [MOBILENET_SSD_BOUNDING_BOX] = "mobilenet-ssd",
  [MOBILENET_SSD_PP_BOUNDING_BOX] = "mobilenet-ssd-postprocess",
  [OV_PERSON_DETECTION_BOUNDING_BOX] = "ov-person-detection",
  [OV_FACE_DETECTION_BOUNDING_BOX] = "ov-face-detection",
  [OLDNAME_MOBILENET_SSD_BOUNDING_BOX] = "tflite-ssd",
  [OLDNAME_MOBILENET_SSD_PP_BOUNDING_BOX] = "tf-ssd",
  [YOLOV5_BOUNDING_BOX] = "yolov5",
  [MP_PALM_DETECTION_BOUNDING_BOX] = "mp-palm-detection",
  [YOLOV8_BOUNDING_BOX] = "yolov8",
  NULL,
};

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
 * @brief C++-Template-like box location calculation for Tensorflow SSD model
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] numinput Input Tensor Data (The number of detections)
 * @param[in] classinput Input Tensor Data (Detected classes)
 * @param[in] scoreinput Input Tensor Data (Detection scores)
 * @param[in] boxesinput Input Tensor Data (Boxes)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mobilenet_ssd_pp(_type, typename, numinput, classinput,       \
    scoreinput, boxesinput, config, results, i_width, i_height)                    \
  case typename:                                                                   \
    {                                                                              \
      int d, num;                                                                  \
      size_t boxbpi;                                                               \
      _type *num_detection_ = (_type *) numinput;                                  \
      _type *classes_ = (_type *) classinput;                                      \
      _type *scores_ = (_type *) scoreinput;                                       \
      _type *boxes_ = (_type *) boxesinput;                                        \
      int locations_idx                                                            \
          = get_mobilenet_ssd_pp_tensor_idx (MOBILENET_SSD_PP_BBOX_IDX_LOCATIONS); \
      num = (int) num_detection_[0];                                               \
      results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), num);     \
      boxbpi = config->info.info[locations_idx].dimension[0];                      \
      for (d = 0; d < num; d++) {                                                  \
        _type x1, x2, y1, y2;                                                      \
        detectedObject object;                                                     \
        if (scores_[d] < threshold)                                                \
          continue;                                                                \
        object.valid = TRUE;                                                       \
        object.class_id = (int) classes_[d];                                       \
        x1 = MIN (MAX (boxes_[d * boxbpi + 1], 0), 1);                             \
        y1 = MIN (MAX (boxes_[d * boxbpi], 0), 1);                                 \
        x2 = MIN (MAX (boxes_[d * boxbpi + 3], 0), 1);                             \
        y2 = MIN (MAX (boxes_[d * boxbpi + 2], 0), 1);                             \
        object.x = (int) (x1 * i_width);                                           \
        object.y = (int) (y1 * i_height);                                          \
        object.width = (int) ((x2 - x1) * i_width);                                \
        object.height = (int) ((y2 - y1) * i_height);                              \
        object.prob = scores_[d];                                                  \
        g_array_append_val (results, object);                                      \
      }                                                                            \
    }                                                                              \
    break

/** @brief Macro to simplify calling _get_objects_mobilenet_ssd_pp */
#define _get_objects_mobilenet_ssd_pp_(type, typename)                                 \
  _get_objects_mobilenet_ssd_pp (type, typename, (mem_num->data), (mem_classes->data), \
      (mem_scores->data), (mem_boxes->data), config, results, i_width, i_height)


#define OV_PERSON_DETECTION_CONF_THRESHOLD (0.8)
/**
 * @brief C++-Template-like box location calculation for OpenVino Person Detection Model
 * @param[in] type The tensor type of inputptr
 * @param[in] intputptr Input tensor Data
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_persons_ov(type, inputptr, typename, results)                                                   \
  case typename:                                                                                             \
    {                                                                                                        \
      detectedObject object = {                                                                              \
        .valid = FALSE, .class_id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .prob = .0, .tracking_id = 0 \
      };                                                                                                     \
      type *typed_inputptr = (type *) inputptr;                                                              \
      guint d;                                                                                               \
                                                                                                             \
      for (d = 1; d <= DETECTION_MAX; ++d) {                                                                 \
        struct {                                                                                             \
          type image_id;                                                                                     \
          type label;                                                                                        \
          type conf;                                                                                         \
          type x_min;                                                                                        \
          type y_min;                                                                                        \
          type x_max;                                                                                        \
          type y_max;                                                                                        \
        } desc;                                                                                              \
                                                                                                             \
        memcpy (&desc, typed_inputptr, sizeof (desc));                                                       \
        typed_inputptr += (sizeof (desc) / sizeof (type));                                                   \
        object.valid = FALSE;                                                                                \
                                                                                                             \
        if ((int) desc.image_id < 0) {                                                                       \
          max_detection = (d - 1);                                                                           \
          break;                                                                                             \
        }                                                                                                    \
        object.class_id = -1;                                                                                \
        object.x = (int) (desc.x_min * (type) i_width);                                                      \
        object.y = (int) (desc.y_min * (type) i_height);                                                     \
        object.width = (int) ((desc.x_max - desc.x_min) * (type) i_width);                                   \
        object.height = (int) ((desc.y_max - desc.y_min) * (type) i_height);                                 \
        if (desc.conf < OV_PERSON_DETECTION_CONF_THRESHOLD)                                                  \
          continue;                                                                                          \
        object.prob = 1;                                                                                     \
        object.valid = TRUE;                                                                                 \
        g_array_append_val (results, object);                                                                \
      }                                                                                                      \
    }                                                                                                        \
    break

/**
 * @brief C++-Template-like box location calculation for Tensorflow model
 * @param[in] type The tensor type of inputptr
 * @param[in] typename nnstreamer enum corresponding to the type
 * @param[in] scoreinput Input Tensor Data (Detection scores)
 * @param[in] boxesinput Input Tensor Data (Boxes)
 * @param[in] config Tensor configs of the input tensors
 * @param[out] results The object returned. (GArray with detectedObject)
 */
#define _get_objects_mp_palm_detection(_type, typename, scoreinput, boxesinput, config) \
  case typename:                                                                        \
    {                                                                                   \
      int d_;                                                                           \
      _type *scores_ = (_type *) scoreinput;                                            \
      _type *boxes_ = (_type *) boxesinput;                                             \
      int num_ = max_detection;                                                         \
      size_t boxbpi_ = config->info.info[0].dimension[0];                               \
      results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), num_);         \
      for (d_ = 0; d_ < num_; d_++) {                                                   \
        gfloat y_center, x_center, h, w;                                                \
        gfloat ymin, xmin;                                                              \
        int y, x, width, height;                                                        \
        detectedObject object;                                                          \
        gfloat score = (gfloat) scores_[d_];                                            \
        _type *box = boxes_ + boxbpi_ * d_;                                             \
        anchor *a = &g_array_index (this->anchors, anchor, d_);                         \
        score = MAX (score, -100.0f);                                                   \
        score = MIN (score, 100.0f);                                                    \
        score = 1.0f / (1.0f + exp (-score));                                           \
        if (score < min_score_threshold)                                                \
          continue;                                                                     \
        y_center = (box[0] * 1.f) / i_height * a->h + a->y_center;                      \
        x_center = (box[1] * 1.f) / i_width * a->w + a->x_center;                       \
        h = (box[2] * 1.f) / i_height * a->h;                                           \
        w = (box[3] * 1.f) / i_width * a->w;                                            \
        ymin = y_center - h / 2.f;                                                      \
        xmin = x_center - w / 2.f;                                                      \
        y = ymin * i_height;                                                            \
        x = xmin * i_width;                                                             \
        width = w * i_width;                                                            \
        height = h * i_height;                                                          \
        object.class_id = 0;                                                            \
        object.x = MAX (0, x);                                                          \
        object.y = MAX (0, y);                                                          \
        object.width = width;                                                           \
        object.height = height;                                                         \
        object.prob = score;                                                            \
        object.valid = TRUE;                                                            \
        g_array_append_val (results, object);                                           \
      }                                                                                 \
    }                                                                                   \
    break

/** @brief Macro to simplify calling _get_objects_mp_palm_detection */
#define _get_objects_mp_palm_detection_(type, typename) \
  _get_objects_mp_palm_detection (type, typename, (detections->data), (boxes->data), config)

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

  protected:
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */

  guint max_detection;
  guint total_labels;
};


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
};

/** @brief Mathematic inverse of sigmoid function, aka logit */
static float
logit (float x)
{
  if (x <= 0.0f)
    return -INFINITY;

  if (x >= 1.0f)
    return INFINITY;

  return log (x / (1.0 - x));
}

class MobilenetSSD : public BoxProperties
{
  public:
  MobilenetSSD ();
  int mobilenet_ssd_loadBoxPrior ();

  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  static const int BOX_SIZE = 4;
  static const int DETECTION_MAX = 2034; /* add ssd_mobilenet v3 support */
  static const guint MAX_TENSORS = 2U;

  static const int THRESHOLD_IDX = 0;
  static const int Y_SCALE_IDX = 1;
  static const int X_SCALE_IDX = 2;
  static const int H_SCALE_IDX = 3;
  static const int W_SCALE_IDX = 4;
  static const int IOU_THRESHOLD_IDX = 5;
  static const int PARAMS_MAX = 6;

  static constexpr gfloat DETECTION_THRESHOLD_DEFAULT = 0.5f;
  static constexpr gfloat THRESHOLD_IOU_DEFAULT = 0.5f;
  static constexpr gfloat Y_SCALE_DEFAULT = 10.0f;
  static constexpr gfloat X_SCALE_DEFAULT = 10.0f;
  static constexpr gfloat H_SCALE_DEFAULT = 5.0f;
  static constexpr gfloat W_SCALE_DEFAULT = 5.0f;

  private:
  char *box_prior_path; /**< Box Prior file path */
  gfloat box_priors[BOX_SIZE][DETECTION_MAX + 1]; /** loaded box prior */
  gfloat params[PARAMS_MAX]; /** Post Processing parameters */
  gfloat sigmoid_threshold; /** Inverse value of valid detection threshold in sigmoid domain */
};

class MobilenetSSDPP : public BoxProperties
{
  public:
  MobilenetSSDPP ();
  int get_mobilenet_ssd_pp_tensor_idx (int idx);

  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  static const int BOX_SIZE = 4;
  static const guint DETECTION_MAX = 100;
  static const guint MAX_TENSORS = 4U;
  static const int LOCATIONS_IDX = 0;
  static const int CLASSES_IDX = 1;
  static const int SCORES_IDX = 2;
  static const int NUM_IDX = 3;

  static const gint LOCATIONS_DEFAULT = 3;
  static const gint CLASSES_DEFAULT = 1;
  static const gint SCORES_DEFAULT = 2;
  static const gint NUM_DEFAULT = 0;
  static constexpr gfloat THRESHOLD_DEFAULT = G_MINFLOAT;

  private:
  gint tensor_mapping[MAX_TENSORS]; /* Output tensor index mapping */
  gfloat threshold; /* Detection threshold */
};

class OVDetection : public BoxProperties
{
  public:
  int setOptionInternal (const char *param)
  {
    UNUSED (param);
    return TRUE;
  }
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  static const guint DETECTION_MAX = 200U;
  static const guint DEFAULT_MAX_TENSORS = 1;
  static const guint DEFAULT_SIZE_DETECTION_DESC = 7;
};

#define YOLO_DETECTION_CONF_THRESHOLD (0.25)
#define YOLO_DETECTION_IOU_THRESHOLD (0.45)

class YoloV5 : public BoxProperties
{
  public:
  YoloV5 ()
      : scaled_output (0), conf_threshold (YOLO_DETECTION_CONF_THRESHOLD),
        iou_threshold (YOLO_DETECTION_IOU_THRESHOLD)
  {
  }
  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  static const int DEFAULT_DETECTION_NUM_INFO = 5;

  private:
  /* From option3, whether the output values are scaled or not */
  int scaled_output;
  gfloat conf_threshold;
  gfloat iou_threshold;
};

class YoloV8 : public BoxProperties
{
  public:
  YoloV8 ()
      : scaled_output (0), conf_threshold (YOLO_DETECTION_CONF_THRESHOLD),
        iou_threshold (YOLO_DETECTION_IOU_THRESHOLD)
  {
  }
  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);
  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  static const int DEFAULT_DETECTION_NUM_INFO = 4;

  private:
  /* From option3, whether the output values are scaled or not */
  int scaled_output;
  gfloat conf_threshold;
  gfloat iou_threshold;
};

class MpPalmDetection : public BoxProperties
{
  public:
  MpPalmDetection ();
  ~MpPalmDetection ();
  void mp_palm_detection_generate_anchors ();
  int setOptionInternal (const char *param);
  int checkCompatible (const GstTensorsConfig *config);

  GArray *decode (const GstTensorsConfig *config, const GstTensorMemory *input);

  static const guint INFO_SIZE = 18;
  static const guint MAX_TENSORS = 2U;
  static const guint MAX_DETECTION = 2016;

  static const gint NUM_LAYERS_DEFAULT = 4;
  static constexpr gfloat MIN_SCALE_DEFAULT = 1.0;
  static constexpr gfloat MAX_SCALE_DEFAULT = 1.0;
  static constexpr gfloat OFFSET_X_DEFAULT = 0.5;
  static constexpr gfloat OFFSET_Y_DEFAULT = 0.5;
  static const gint STRIDE_0_DEFAULT = 8;
  static const gint STRIDE_1_DEFAULT = 16;
  static const gint STRIDE_2_DEFAULT = 16;
  static const gint STRIDE_3_DEFAULT = 16;
  static constexpr gfloat MIN_SCORE_THRESHOLD_DEFAULT = 0.5;

  static const int PARAMS_STRIDE_SIZE = 8;
  static const int PARAMS_MAX = 13;

  private:
  gint num_layers;
  /** Number of stride layers */
  gfloat min_scale; /** Minimum scale */
  gfloat max_scale; /** Maximum scale */
  gfloat offset_x; /** anchor X offset */
  gfloat offset_y; /** anchor Y offset */
  gint strides[PARAMS_MAX]; /** Stride data for each layers */
  gfloat min_score_threshold; /** minimum threshold of score */

  GArray *anchors;
};
#endif /* _TENSORDECBB_H__ */
