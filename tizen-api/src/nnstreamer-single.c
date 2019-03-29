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
 * @file nnstreamer-simple.c
 * @date 08 May 2019
 * @brief Tizen NNStreamer/Simple C-API Wrapper.
 *        This allows to invoke individual input frame with NNStreamer.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <nnstreamer.h>         /* Uses NNStreamer/Pipeline C-API */

typedef struct
{
  nns_pipeline_h pipe;
} ml_simpleshot_model;
/**
 * ml_simpleshot_model *model = g_new0 (ml_simpleshot_model, 1);
 * ml_simpleshot_model_h *model_h;
 * *model_h = model;
 */

/**
 * @brief Refer to nnstreamer-single.h
 */
int
ml_model_open (const char *model_path, ml_simpleshot_model_h * model,
    const nns_tensors_info_s * inputtype, ml_model_nnfw nnfw, ml_model_hw hw)
{
  ml_simpleshot_model_h *_model;
  int ret = NNS_ERROR_NONE;
  char *pipedesc;               /* pipeline description */

  /* 1. Determine nnfw */

  /* 2. Determine hw */

  /* 3. Determine input dimension ==> caps_filter string */

  /* 4. Construct a pipeline */
  _model = g_new (ml_simpleshot_model, 1);
  ret = nns_pipeline_construct (pipedesc, &_model->pipe);

  /* 5. Allocate */
  *model = _model;
}
