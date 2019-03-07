/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file tizen-api-private.h
 * @date 07 March 2019
 * @brief Tizen NNStreamer/Pipeline(main) C-API Private Header.
 *        This file should NOT be exported to SDK or devel package.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_NNSTREAMER_API_PRIVATE_H__
#define __TIZEN_NNSTREAMER_API_PRIVATE_H___

#include <glib.h>
#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Internal private representation of pipeline handle.
 * @detail This should not be exposed to applications
 */
typedef struct _nns_pipeline {
  GstElement *element;
  GError *error;
  GMutex lock;
  /** @todo: add list of switches and valves for faster control */
  /** @todo: add list of appsrc / tensorsink for faster stream */
} nns_pipeline;

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /*__TIZEN_NNSTREAMER_API_PRIVATE_H___*/
