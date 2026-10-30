/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    hailort.h
 * @date    28 Oct 2026
 * @brief   Reduced stand-in of the HailoRT C API for the sub-plugin tests
 * @see     https://github.com/hailo-ai/hailort
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * Declares only the types, values and functions that tensor_filter_hailort.cc
 * uses, with the names and signatures of the public HailoRT 4.18 - 5.x
 * header. It is not ABI compatible with HailoRT: the structures are reduced,
 * and only the mock in hailort_mock.cc implements these functions.
 */
#ifndef __HAILORT_MOCK_HAILORT_H__
#define __HAILORT_MOCK_HAILORT_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HAILORTAPI __attribute__ ((visibility ("default")))

#define HAILO_MAX_NAME_SIZE (128)
#define HAILO_MAX_STREAM_NAME_SIZE (HAILO_MAX_NAME_SIZE)
#define HAILO_MAX_STREAMS_COUNT (40)
#define HAILO_MAX_NETWORK_GROUPS (8)
#define HAILO_MAX_NETWORK_GROUP_NAME_SIZE (HAILO_MAX_NAME_SIZE)
#define HAILO_MAX_NETWORK_NAME_SIZE (HAILO_MAX_NETWORK_GROUP_NAME_SIZE + 1 + HAILO_MAX_NAME_SIZE)
#define HAILO_UNIQUE_VDEVICE_GROUP_ID ("UNIQUE")
#define HAILO_DEFAULT_VDEVICE_GROUP_ID HAILO_UNIQUE_VDEVICE_GROUP_ID

/** @brief HailoRT return codes (a subset, with the values of HailoRT) */
typedef enum {
  HAILO_SUCCESS = 0,
  HAILO_INVALID_ARGUMENT = 2,
  HAILO_OUT_OF_HOST_MEMORY = 3,
  HAILO_TIMEOUT = 4,
  HAILO_INSUFFICIENT_BUFFER = 5,
  HAILO_INVALID_OPERATION = 6,
  HAILO_OPEN_FILE_FAILURE = 13,
  HAILO_INVALID_HEF = 26,
  HAILO_NOT_FOUND = 61,
  HAILO_OUT_OF_PHYSICAL_DEVICES = 74,
} hailo_status;

typedef struct _hailo_vdevice *hailo_vdevice;
typedef struct _hailo_hef *hailo_hef;
typedef struct _hailo_configured_network_group *hailo_configured_network_group;
typedef struct _hailo_input_vstream *hailo_input_vstream;
typedef struct _hailo_output_vstream *hailo_output_vstream;

/** @brief Scheduling algorithm of a virtual device */
typedef enum {
  HAILO_SCHEDULING_ALGORITHM_NONE = 0,
  HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN,
} hailo_scheduling_algorithm_t;

/** @brief Opaque in this mock */
typedef struct _hailo_device_id_t hailo_device_id_t;

/** @brief Virtual device parameters */
typedef struct {
  uint32_t device_count;
  hailo_device_id_t *device_ids;
  hailo_scheduling_algorithm_t scheduling_algorithm;
  const char *group_id;
  bool multi_process_service;
} hailo_vdevice_params_t;

/** @brief Data format types */
typedef enum {
  HAILO_FORMAT_TYPE_AUTO = 0,
  HAILO_FORMAT_TYPE_UINT8 = 1,
  HAILO_FORMAT_TYPE_UINT16 = 2,
  HAILO_FORMAT_TYPE_FLOAT32 = 3,
} hailo_format_type_t;

/** @brief Data format orders (a subset) */
typedef enum {
  HAILO_FORMAT_ORDER_AUTO = 0,
  HAILO_FORMAT_ORDER_NHWC = 1,
  HAILO_FORMAT_ORDER_NHW = 5,
  HAILO_FORMAT_ORDER_NC = 6,
  HAILO_FORMAT_ORDER_HAILO_NMS = 9,
  HAILO_FORMAT_ORDER_NCHW = 11,
} hailo_format_order_t;

/** @brief Data format flags */
typedef enum {
  HAILO_FORMAT_FLAGS_NONE = 0,
} hailo_format_flags_t;

/** @brief Hailo data format */
typedef struct {
  hailo_format_type_t type;
  hailo_format_order_t order;
  hailo_format_flags_t flags;
} hailo_format_t;

/** @brief Stream direction */
typedef enum {
  HAILO_H2D_STREAM = 0,
  HAILO_D2H_STREAM = 1,
} hailo_stream_direction_t;

/** @brief Image shape */
typedef struct {
  uint32_t height;
  uint32_t width;
  uint32_t features;
} hailo_3d_image_shape_t;

/** @brief Shape of an NMS result */
typedef struct {
  uint32_t number_of_classes;
  uint32_t max_bboxes_per_class;
  uint32_t max_bboxes_total;
  uint32_t max_accumulated_mask_size;
} hailo_nms_shape_t;

/** @brief Quantization parameters */
typedef struct {
  float qp_zp;
  float qp_scale;
  float limvals_min;
  float limvals_max;
} hailo_quant_info_t;

/** @brief Input or output virtual stream information */
typedef struct {
  char name[HAILO_MAX_STREAM_NAME_SIZE];
  char network_name[HAILO_MAX_NETWORK_NAME_SIZE];
  hailo_stream_direction_t direction;
  hailo_format_t format;
  union {
    hailo_3d_image_shape_t shape;
    hailo_nms_shape_t nms_shape;
  };
  hailo_quant_info_t quant_info;
} hailo_vstream_info_t;

/** @brief Virtual stream parameters (reduced) */
typedef struct {
  hailo_format_t user_buffer_format;
  uint32_t timeout_ms;
  uint32_t queue_size;
} hailo_vstream_params_t;

/** @brief Input virtual stream parameters */
typedef struct {
  char name[HAILO_MAX_STREAM_NAME_SIZE];
  hailo_vstream_params_t params;
} hailo_input_vstream_params_by_name_t;

/** @brief Output virtual stream parameters */
typedef struct {
  char name[HAILO_MAX_STREAM_NAME_SIZE];
  hailo_vstream_params_t params;
} hailo_output_vstream_params_by_name_t;

/** @brief Configure parameters per network group (reduced) */
typedef struct {
  char name[HAILO_MAX_NETWORK_GROUP_NAME_SIZE];
  uint16_t batch_size;
} hailo_configure_network_group_params_t;

/** @brief Configure parameters */
typedef struct {
  size_t network_group_params_count;
  hailo_configure_network_group_params_t network_group_params[HAILO_MAX_NETWORK_GROUPS];
} hailo_configure_params_t;

/** @brief Get the name of a status code. */
HAILORTAPI const char *hailo_get_status_message (hailo_status status);

/** @brief Fill virtual device parameters with the defaults. */
HAILORTAPI hailo_status hailo_init_vdevice_params (hailo_vdevice_params_t *params);
/** @brief Create a virtual device. */
HAILORTAPI hailo_status hailo_create_vdevice (hailo_vdevice_params_t *params, hailo_vdevice *vdevice);
/** @brief Configure a virtual device with a model. */
HAILORTAPI hailo_status hailo_configure_vdevice (hailo_vdevice vdevice, hailo_hef hef,
    hailo_configure_params_t *params, hailo_configured_network_group *network_groups,
    size_t *number_of_network_groups);
/** @brief Release a virtual device and its network groups. */
HAILORTAPI hailo_status hailo_release_vdevice (hailo_vdevice vdevice);

/** @brief Load a model file. */
HAILORTAPI hailo_status hailo_create_hef_file (hailo_hef *hef, const char *file_name);
/** @brief Release a model. */
HAILORTAPI hailo_status hailo_release_hef (hailo_hef hef);
/** @brief Fill configure parameters with the defaults of a model. */
HAILORTAPI hailo_status hailo_init_configure_params_by_vdevice (
    hailo_hef hef, hailo_vdevice vdevice, hailo_configure_params_t *params);

/** @brief Make the parameters of the input virtual streams. */
HAILORTAPI hailo_status hailo_make_input_vstream_params (hailo_configured_network_group network_group,
    bool unused, hailo_format_type_t format_type,
    hailo_input_vstream_params_by_name_t *input_params, size_t *input_params_count);
/** @brief Make the parameters of the output virtual streams. */
HAILORTAPI hailo_status hailo_make_output_vstream_params (hailo_configured_network_group network_group,
    bool unused, hailo_format_type_t format_type,
    hailo_output_vstream_params_by_name_t *output_params, size_t *output_params_count);
/** @brief Create input virtual streams. */
HAILORTAPI hailo_status hailo_create_input_vstreams (hailo_configured_network_group configured_network_group,
    const hailo_input_vstream_params_by_name_t *inputs_params, size_t inputs_count,
    hailo_input_vstream *input_vstreams);
/** @brief Create output virtual streams. */
HAILORTAPI hailo_status hailo_create_output_vstreams (hailo_configured_network_group configured_network_group,
    const hailo_output_vstream_params_by_name_t *outputs_params, size_t outputs_count,
    hailo_output_vstream *output_vstreams);
/** @brief Release input virtual streams. */
HAILORTAPI hailo_status hailo_release_input_vstreams (
    const hailo_input_vstream *input_vstreams, size_t inputs_count);
/** @brief Release output virtual streams. */
HAILORTAPI hailo_status hailo_release_output_vstreams (
    const hailo_output_vstream *output_vstreams, size_t outputs_count);

/** @brief Get the host-side frame size of an input virtual stream. */
HAILORTAPI hailo_status hailo_get_input_vstream_frame_size (
    hailo_input_vstream input_vstream, size_t *frame_size);
/** @brief Get the information of an input virtual stream. */
HAILORTAPI hailo_status hailo_get_input_vstream_info (
    hailo_input_vstream input_vstream, hailo_vstream_info_t *vstream_info);
/** @brief Get the host-side format of an input virtual stream. */
HAILORTAPI hailo_status hailo_get_input_vstream_user_format (
    hailo_input_vstream input_vstream, hailo_format_t *user_buffer_format);
/** @brief Get the host-side frame size of an output virtual stream. */
HAILORTAPI hailo_status hailo_get_output_vstream_frame_size (
    hailo_output_vstream output_vstream, size_t *frame_size);
/** @brief Get the information of an output virtual stream. */
HAILORTAPI hailo_status hailo_get_output_vstream_info (
    hailo_output_vstream output_vstream, hailo_vstream_info_t *vstream_info);
/** @brief Get the host-side format of an output virtual stream. */
HAILORTAPI hailo_status hailo_get_output_vstream_user_format (
    hailo_output_vstream output_vstream, hailo_format_t *user_buffer_format);

/** @brief Write a frame to an input virtual stream. */
HAILORTAPI hailo_status hailo_vstream_write_raw_buffer (
    hailo_input_vstream input_vstream, const void *buffer, size_t buffer_size);
/** @brief Read a frame from an output virtual stream. */
HAILORTAPI hailo_status hailo_vstream_read_raw_buffer (
    hailo_output_vstream output_vstream, void *buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif /* __HAILORT_MOCK_HAILORT_H__ */
