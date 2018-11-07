
/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file   tensordec_bounding_boxes_core.h
 * @author Jinhyuck Park <jinhyuck83.park@samsung.com>
 * @date   11/6/2018
 * @brief	 tensor decoder bounding boxes mode c++ apis.
 *
 * @bug     No known bugs.
 */


/**
 * @brief	the definition of functions to be used at C files.
 */

#ifdef __cplusplus
#include <glib.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include "tensordec.h"

extern "C"
{
#endif
  extern gboolean gst_tensordec_read_lines (const gchar * file_name,
      GList ** lines);
  extern void gst_tensordec_get_detected_objects (gfloat * detections,
      gfloat * boxes);
#ifdef __cplusplus
}
#endif
