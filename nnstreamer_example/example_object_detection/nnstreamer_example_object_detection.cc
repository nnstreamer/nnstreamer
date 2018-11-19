/**
 * @file	nnstreamer_example_object_detection.cc
 * @date	22 October 2018
 * @brief	Tensor stream example with TF-Lite model for object detection
 * @author	HyoungJoo Ahn <hello.ahn@samsung.com>
 * @bug		No known bugs.
 *
 * Run example :
 * Before running this example, GST_PLUGIN_PATH should be updated for nnstreamer plug-in.
 * $ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:<nnstreamer plugin path>
 * $ ./nnstreamer_example_object_detection
 *
 * Required model and resources are stored at below link
 * https://github.com/nnsuite/testcases/tree/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <glib.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <math.h>
#include <cairo.h>
#include <cairo-gobject.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Macro for debug message.
 */
#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

/**
 * @brief Macro to check error case.
 */
#define _check_cond_err(cond) \
  if (!(cond)) { \
    _print_log ("app failed! [line : %d]", __LINE__); \
    goto error; \
  }

#define Y_SCALE         10.0f
#define X_SCALE         10.0f
#define H_SCALE         5.0f
#define W_SCALE         5.0f

#define VIDEO_WIDTH     640
#define VIDEO_HEIGHT    480
#define MODEL_WIDTH     300
#define MODEL_HEIGHT    300

#define BOX_SIZE        4
#define LABEL_SIZE      91
#define DETECTION_MAX   1917

/**
 * @brief Max objects in display.
 */
#define MAX_OBJECT_DETECTION 5

typedef struct
{
  gint x;
  gint y;
  gint width;
  gint height;
  gint class_id;
  gfloat prob;
} DetectedObject;

typedef struct
{
  gboolean valid;
  GstVideoInfo vinfo;
} CairoOverlayState;

/**
 * @brief Data structure for tflite model info.
 */
typedef struct
{
  gchar *model_path; /**< tflite model file path */
  gchar *label_path; /**< label file path */
  gchar *box_prior_path; /**< box prior file path */
  gfloat box_priors[BOX_SIZE][DETECTION_MAX]; /**< box prior */
  GList *labels; /**< list of loaded labels */
} TFLiteModelInfo;

/**
 * @brief Data structure for app.
 */
typedef struct
{
  GMainLoop *loop; /**< main event loop */
  GstElement *pipeline; /**< gst pipeline for data stream */
  GstBus *bus; /**< gst bus for data pipeline */
  gboolean running; /**< true when app is running */
  GMutex mutex; /**< mutex for processing */
  TFLiteModelInfo tflite_info; /**< tflite model info */
  CairoOverlayState overlay_state;
  std::vector < DetectedObject > detected_objects;
} AppData;

/**
 * @brief Data for pipeline and result.
 */
static AppData g_app;

/**
 * @brief Read strings from file.
 */
static gboolean
read_lines (const gchar * file_name, GList ** lines)
{
  std::ifstream file (file_name);
  if (!file) {
    _print_log ("Failed to open file %s", file_name);
    return FALSE;
  }

  std::string str;
  while (std::getline (file, str)) {
    *lines = g_list_append (*lines, g_strdup (str.c_str ()));
  }

  return TRUE;
}

/**
 * @brief Load box priors.
 */
static gboolean
tflite_load_box_priors (TFLiteModelInfo * tflite_info)
{
  GList *box_priors = NULL;
  gchar *box_row;

  g_return_val_if_fail (tflite_info != NULL, FALSE);
  g_return_val_if_fail (read_lines (tflite_info->box_prior_path, &box_priors),
      FALSE);

  for (int row = 0; row < BOX_SIZE; row++) {
    int column = 0;
    int i = 0, j = 0;
    char buff[11];

    memset (buff, 0, 11);
    box_row = (gchar *) g_list_nth_data (box_priors, row);

    while ((box_row[i] != '\n') && (box_row[i] != '\0')) {
      if (box_row[i] != ' ') {
        buff[j] = box_row[i];
        j++;
      } else {
        if (j != 0) {
          tflite_info->box_priors[row][column++] = atof (buff);
          memset (buff, 0, 11);
        }
        j = 0;
      }
      i++;
    }

    tflite_info->box_priors[row][column++] = atof (buff);
  }

  g_list_free_full (box_priors, g_free);
  return TRUE;
}

/**
 * @brief Load labels.
 */
static gboolean
tflite_load_labels (TFLiteModelInfo * tflite_info)
{
  g_return_val_if_fail (tflite_info != NULL, FALSE);

  return read_lines (tflite_info->label_path, &tflite_info->labels);
}

/**
 * @brief Check tflite model and load labels.
 */
static gboolean
tflite_init_info (TFLiteModelInfo * tflite_info, const gchar * path)
{
  const gchar tflite_model[] = "ssd_mobilenet_v2_coco.tflite";
  const gchar tflite_label[] = "coco_labels_list.txt";
  const gchar tflite_box_priors[] = "box_priors.txt";

  g_return_val_if_fail (tflite_info != NULL, FALSE);

  tflite_info->model_path = g_strdup_printf ("%s/%s", path, tflite_model);
  tflite_info->label_path = g_strdup_printf ("%s/%s", path, tflite_label);
  tflite_info->box_prior_path =
      g_strdup_printf ("%s/%s", path, tflite_box_priors);

  tflite_info->labels = NULL;

  g_return_val_if_fail (tflite_load_box_priors (tflite_info), FALSE);
  g_return_val_if_fail (tflite_load_labels (tflite_info), FALSE);

  return TRUE;
}

/**
 * @brief Free data in tflite info structure.
 */
static void
tflite_free_info (TFLiteModelInfo * tflite_info)
{
  g_return_if_fail (tflite_info != NULL);

  if (tflite_info->model_path) {
    g_free (tflite_info->model_path);
    tflite_info->model_path = NULL;
  }

  if (tflite_info->label_path) {
    g_free (tflite_info->label_path);
    tflite_info->label_path = NULL;
  }

  if (tflite_info->box_prior_path) {
    g_free (tflite_info->box_prior_path);
    tflite_info->box_prior_path = NULL;
  }

  if (tflite_info->labels) {
    g_list_free_full (tflite_info->labels, g_free);
    tflite_info->labels = NULL;
  }
}

/**
 * @brief Free resources in app data.
 */
static void
free_app_data (void)
{
  if (g_app.loop) {
    g_main_loop_unref (g_app.loop);
    g_app.loop = NULL;
  }

  if (g_app.bus) {
    gst_bus_remove_signal_watch (g_app.bus);
    gst_object_unref (g_app.bus);
    g_app.bus = NULL;
  }

  if (g_app.pipeline) {
    gst_object_unref (g_app.pipeline);
    g_app.pipeline = NULL;
  }

  g_app.detected_objects.clear ();

  tflite_free_info (&g_app.tflite_info);
  g_mutex_clear (&g_app.mutex);
}

/**
 * @brief Function to print error message.
 */
static void
parse_err_message (GstMessage * message)
{
  gchar *debug;
  GError *error;

  g_return_if_fail (message != NULL);

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:
      gst_message_parse_error (message, &error, &debug);
      break;

    case GST_MESSAGE_WARNING:
      gst_message_parse_warning (message, &error, &debug);
      break;

    default:
      return;
  }

  gst_object_default_error (GST_MESSAGE_SRC (message), error, debug);
  g_error_free (error);
  g_free (debug);
}

/**
 * @brief Function to print qos message.
 */
static void
parse_qos_message (GstMessage * message)
{
  GstFormat format;
  guint64 processed;
  guint64 dropped;

  gst_message_parse_qos_stats (message, &format, &processed, &dropped);
  _print_log ("format[%d] processed[%" G_GUINT64_FORMAT "] dropped[%"
      G_GUINT64_FORMAT "]", format, processed, dropped);
}

/**
 * @brief Callback for message.
 */
static void
bus_message_cb (GstBus * bus, GstMessage * message, gpointer user_data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      _print_log ("received eos message");
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_ERROR:
      _print_log ("received error message");
      parse_err_message (message);
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_WARNING:
      _print_log ("received warning message");
      parse_err_message (message);
      break;

    case GST_MESSAGE_STREAM_START:
      _print_log ("received start message");
      break;

    case GST_MESSAGE_QOS:
      parse_qos_message (message);
      break;

    default:
      break;
  }
}

/**
 * @brief Compare score of detected objects.
 */
static bool
compare_objs (DetectedObject & a, DetectedObject & b)
{
  return a.prob > b.prob;
}

/**
 * @brief Intersection of union
 */
static gfloat
iou (DetectedObject & A, DetectedObject & B)
{
  int x1 = std::max (A.x, B.x);
  int y1 = std::max (A.y, B.y);
  int x2 = std::min (A.x + A.width, B.x + B.width);
  int y2 = std::min (A.y + A.height, B.y + B.height);
  int w = std::max (0, (x2 - x1 + 1));
  int h = std::max (0, (y2 - y1 + 1));
  float inter = w * h;
  float areaA = A.width * A.height;
  float areaB = B.width * B.height;
  float o = inter / (areaA + areaB - inter);
  return (o >= 0) ? o : 0;
}

/**
 * @brief NMS (non-maximum suppression)
 */
static void
nms (std::vector < DetectedObject > &detected)
{
  const float threshold_iou = .5f;
  guint boxes_size;
  guint i, j;

  std::sort (detected.begin (), detected.end (), compare_objs);
  boxes_size = detected.size ();

  std::vector < bool > del (boxes_size, false);
  for (i = 0; i < boxes_size; i++) {
    if (!del[i]) {
      for (j = i + 1; j < boxes_size; j++) {
        if (iou (detected.at (i), detected.at (j)) > threshold_iou) {
          del[j] = true;
        }
      }
    }
  }

  /* update result */
  g_mutex_lock (&g_app.mutex);

  g_app.detected_objects.clear ();
  for (i = 0; i < boxes_size; i++) {
    if (!del[i]) {
      g_app.detected_objects.push_back (detected[i]);

      if (DBG) {
        _print_log ("==============================");
        _print_log ("Label           : %s",
            (gchar *) g_list_nth_data (g_app.tflite_info.labels,
                detected[i].class_id));
        _print_log ("x               : %d", detected[i].x);
        _print_log ("y               : %d", detected[i].y);
        _print_log ("width           : %d", detected[i].width);
        _print_log ("height          : %d", detected[i].height);
        _print_log ("Confidence Score: %f", detected[i].prob);
      }
    }
  }

  g_mutex_unlock (&g_app.mutex);
}

#define _expit(x) \
    (1.f / (1.f + expf (-x)))

/**
 * @brief Get detected objects.
 */
static void
get_detected_objects (gfloat * detections, gfloat * boxes)
{
  const float threshold_score = .5f;
  std::vector < DetectedObject > detected;

  for (int d = 0; d < DETECTION_MAX; d++) {
    float ycenter =
        boxes[0] / Y_SCALE * g_app.tflite_info.box_priors[2][d] +
        g_app.tflite_info.box_priors[0][d];
    float xcenter =
        boxes[1] / X_SCALE * g_app.tflite_info.box_priors[3][d] +
        g_app.tflite_info.box_priors[1][d];
    float h =
        (float) expf (boxes[2] / H_SCALE) * g_app.tflite_info.box_priors[2][d];
    float w =
        (float) expf (boxes[3] / W_SCALE) * g_app.tflite_info.box_priors[3][d];

    float ymin = ycenter - h / 2.f;
    float xmin = xcenter - w / 2.f;
    float ymax = ycenter + h / 2.f;
    float xmax = xcenter + w / 2.f;

    int x = xmin * MODEL_WIDTH;
    int y = ymin * MODEL_HEIGHT;
    int width = (xmax - xmin) * MODEL_WIDTH;
    int height = (ymax - ymin) * MODEL_HEIGHT;

    for (int c = 1; c < LABEL_SIZE; c++) {
      gfloat score = _expit (detections[c]);
      /**
       * This score cutoff is taken from Tensorflow's demo app.
       * There are quite a lot of nodes to be run to convert it to the useful possibility
       * scores. As a result of that, this cutoff will cause it to lose good detections in
       * some scenarios and generate too much noise in other scenario.
       */
      if (score < threshold_score)
        continue;

      DetectedObject object;

      object.class_id = c;
      object.x = x;
      object.y = y;
      object.width = width;
      object.height = height;
      object.prob = score;

      detected.push_back (object);
    }

    detections += LABEL_SIZE;
    boxes += BOX_SIZE;
  }

  nms (detected);
}

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  GstMemory *mem_boxes, *mem_detections;
  GstMapInfo info_boxes, info_detections;
  gfloat *boxes, *detections;

  g_return_if_fail (g_app.running);

  /**
   * tensor type is float32.
   * [0] dim of boxes > BOX_SIZE : 1 : DETECTION_MAX : 1
   * [1] dim of labels > LABEL_SIZE : DETECTION_MAX : 1 : 1
   */
  g_assert (gst_buffer_n_memory (buffer) == 2);

  /* boxes */
  mem_boxes = gst_buffer_get_memory (buffer, 0);
  g_assert (gst_memory_map (mem_boxes, &info_boxes, GST_MAP_READ));
  g_assert (info_boxes.size == BOX_SIZE * DETECTION_MAX * 4);
  boxes = (gfloat *) info_boxes.data;

  /* detections */
  mem_detections = gst_buffer_get_memory (buffer, 1);
  g_assert (gst_memory_map (mem_detections, &info_detections, GST_MAP_READ));
  g_assert (info_detections.size == LABEL_SIZE * DETECTION_MAX * 4);
  detections = (gfloat *) info_detections.data;

  get_detected_objects (detections, boxes);

  gst_memory_unmap (mem_boxes, &info_boxes);
  gst_memory_unmap (mem_detections, &info_detections);

  gst_memory_unref (mem_boxes);
  gst_memory_unref (mem_detections);
}

/**
 * @brief Set window title.
 * @param name GstXImageSink element name
 * @param title window title
 */
static void
set_window_title (const gchar * name, const gchar * title)
{
  GstTagList *tags;
  GstPad *sink_pad;
  GstElement *element;

  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), name);

  g_return_if_fail (element != NULL);

  sink_pad = gst_element_get_static_pad (element, "sink");

  if (sink_pad) {
    tags = gst_tag_list_new (GST_TAG_TITLE, title, NULL);
    gst_pad_send_event (sink_pad, gst_event_new_tag (tags));
    gst_object_unref (sink_pad);
  }

  gst_object_unref (element);
}

/**
 * @brief Store the information from the caps that we are interested in.
 */
static void
prepare_overlay_cb (GstElement * overlay, GstCaps * caps, gpointer user_data)
{
  CairoOverlayState *state = &g_app.overlay_state;

  state->valid = gst_video_info_from_caps (&state->vinfo, caps);
}

/**
 * @brief Callback to draw the overlay.
 */
static void
draw_overlay_cb (GstElement * overlay, cairo_t * cr, guint64 timestamp,
    guint64 duration, gpointer user_data)
{
  CairoOverlayState *state = &g_app.overlay_state;
  gfloat x, y, width, height;
  gchar *label;
  guint drawed = 0;

  g_return_if_fail (state->valid);
  g_return_if_fail (g_app.running);

  std::vector < DetectedObject > detected;
  std::vector < DetectedObject >::iterator iter;

  g_mutex_lock (&g_app.mutex);
  detected = g_app.detected_objects;
  g_mutex_unlock (&g_app.mutex);

  /* set font props */
  cairo_select_font_face (cr, "Sans", CAIRO_FONT_SLANT_NORMAL,
      CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size (cr, 20.0);

  for (iter = detected.begin (); iter != detected.end (); ++iter) {
    label =
        (gchar *) g_list_nth_data (g_app.tflite_info.labels, iter->class_id);

    x = iter->x * VIDEO_WIDTH / MODEL_WIDTH;
    y = iter->y * VIDEO_HEIGHT / MODEL_HEIGHT;
    width = iter->width * VIDEO_WIDTH / MODEL_WIDTH;
    height = iter->height * VIDEO_WIDTH / MODEL_HEIGHT;

    /* draw rectangle */
    cairo_rectangle (cr, x, y, width, height);
    cairo_set_source_rgb (cr, 1, 0, 0);
    cairo_set_line_width (cr, 1.5);
    cairo_stroke (cr);
    cairo_fill_preserve (cr);

    /* draw title */
    cairo_move_to (cr, x + 5, y + 25);
    cairo_text_path (cr, label);
    cairo_set_source_rgb (cr, 1, 0, 0);
    cairo_fill_preserve (cr);
    cairo_set_source_rgb (cr, 1, 1, 1);
    cairo_set_line_width (cr, .3);
    cairo_stroke (cr);
    cairo_fill_preserve (cr);

    if (++drawed >= MAX_OBJECT_DETECTION) {
      /* max objects drawed */
      break;
    }
  }
}

/**
 * @brief Main function.
 */
int
main (int argc, char ** argv)
{
  const gchar tflite_model_path[] = "./tflite_model_ssd";

  gchar *str_pipeline;
  GstElement *element;

  _print_log ("start app..");

  /* init app variable */
  g_app.running = FALSE;
  g_app.loop = NULL;
  g_app.bus = NULL;
  g_app.pipeline = NULL;
  g_app.detected_objects.clear ();
  g_mutex_init (&g_app.mutex);

  _check_cond_err (tflite_init_info (&g_app.tflite_info, tflite_model_path));

  /* init gstreamer */
  gst_init (&argc, &argv);

  /* main loop */
  g_app.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app.loop != NULL);

  /* init pipeline */
  str_pipeline =
      g_strdup_printf
      ("v4l2src name=src ! videoscale ! "
      "video/x-raw,width=%d,height=%d,format=RGB ! tee name=t_raw "
      "t_raw. ! queue ! videoconvert ! cairooverlay name=tensor_res ! ximagesink name=img_tensor "
      "t_raw. ! queue leaky=2 max-size-buffers=2 ! videoscale ! video/x-raw,width=%d,height=%d ! tensor_converter ! "
      "tensor_transform mode=arithmetic option=typecast:float32,add:-127,mul:0.007843 ! "
      "tensor_filter framework=tensorflow-lite model=%s ! "
      "tensor_sink name=tensor_sink",
      VIDEO_WIDTH, VIDEO_HEIGHT, MODEL_WIDTH, MODEL_HEIGHT,
      g_app.tflite_info.model_path);
  g_app.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app.pipeline != NULL);

  /* bus and message callback */
  g_app.bus = gst_element_get_bus (g_app.pipeline);
  _check_cond_err (g_app.bus != NULL);

  gst_bus_add_signal_watch (g_app.bus);
  g_signal_connect (g_app.bus, "message", G_CALLBACK (bus_message_cb), NULL);

  /* tensor sink signal : new data callback */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_sink");
  g_signal_connect (element, "new-data", G_CALLBACK (new_data_cb), NULL);
  gst_object_unref (element);

  /* cairo overlay */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_res");
  g_signal_connect (element, "draw", G_CALLBACK (draw_overlay_cb), NULL);
  g_signal_connect (element, "caps-changed", G_CALLBACK (prepare_overlay_cb),
      NULL);
  gst_object_unref (element);

  /* start pipeline */
  gst_element_set_state (g_app.pipeline, GST_STATE_PLAYING);
  g_app.running = TRUE;

  /* set window title */
  set_window_title ("img_tensor", "NNStreamer Example");

  /* run main loop */
  g_main_loop_run (g_app.loop);

  /* quit when received eos or error message */
  g_app.running = FALSE;

  /* cam source element */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "src");

  gst_element_set_state (element, GST_STATE_READY);
  gst_element_set_state (g_app.pipeline, GST_STATE_READY);

  g_usleep (200 * 1000);

  gst_element_set_state (element, GST_STATE_NULL);
  gst_element_set_state (g_app.pipeline, GST_STATE_NULL);

  g_usleep (200 * 1000);
  gst_object_unref (element);

error:
  _print_log ("close app..");

  free_app_data ();
  return 0;
}
