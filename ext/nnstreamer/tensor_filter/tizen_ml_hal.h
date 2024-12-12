/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        tizen_ml_hal.h
 * @date        12 Dec 2024
 * @brief       Tizen HAL support for tensor-filter subplugins
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see         http://github.com/nnstreamer/nnstreamer
 * @bug         This is not yet implemented fully.
 *
 * @todo        Driver discovery/load/unload infrastructure. Does Tizen HAL support it already?
 * @todo        Re-license to APACHE 2.0?
 *
 * This represents Approach A of https://github.com/nnstreamer/nnstreamer/issues/4660
 */
#include <stdint.h>

typedef int (*driver_callback) (void *private_data, int arg1, int arg2);

/**
 * @brief Tizen ML HAL. API Version 1.0-draft
 * @note Most of the callbacks may be kept NULL.
 *       The private_data can be freely defined by the driver author.
 *       It is recommended to allocate the memory for private_data by the caller (C++/platform side).
 * @note Use "main" in general. Use pre/post if you need to do something in C++/platform side
 *       between the calls.
 *
 */
typedef struct {
  char driver_name[256];
  char device_info[256];
  uint64_t driver_version;
  uint64_t tizen_ml_hal_api_version; /**< e.g., 0x0000 0001 0000 0000 = API 1 / Draft 0 */

  driver_callback constructor_pre;
  driver_callback constructor_main;
  driver_callback constructor_post;

  driver_callback destructor_pre;
  driver_callback destructor_main;
  driver_callback destructor_post;

  driver_callback getEmptyInstance_pre;
  driver_callback getEmptyInstance_main;
  driver_callback getEmptyInstance_post;

  driver_callback configure_instance_pre;
  driver_callback configure_instance_main;
  driver_callback configure_instance_post;

  driver_callback invoke_pre;
  driver_callback invoke_main;
  driver_callback invoke_post;

  driver_callback invoke_dynamic_pre;
  driver_callback invoke_dynamic_main;
  driver_callback invoke_dynamic_post;

  driver_callback getFrameworkInfo_pre;
  driver_callback getFrameworkInfo_main;
  driver_callback getFrameworkInfo_post;

  driver_callback getModelInfo_pre;
  driver_callback getModelInfo_main;
  driver_callback getModelInfo_post;

  driver_callback eventHandler_pre;
  driver_callback eventHandler_main;
  driver_callback eventHandler_post;

  driver_callback custom_functions[64];
} tizen_ml_hal_tensor_filter_driver;
