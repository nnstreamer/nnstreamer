/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        tizen_ml_hal_tensor_filter_driver_snpe.c
 * @date        12 Dec 2024
 * @brief       Tizen HAL support for snpe tensor-filter subplugins
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see         http://github.com/nnstreamer/nnstreamer
 * @bug         This is not yet implemented fully.
 *
 * @todo        Re-license to APACHE 2.0?
 *
 * This represents Approach A of https://github.com/nnstreamer/nnstreamer/issues/4660
 */
#include <stdint.h>
#include "tizen_ml_hal.h"

#include <DlContainer/DlContainer.h>
#include <DlSystem/DlEnums.h>
#include <DlSystem/DlError.h>
#include <DlSystem/DlVersion.h>
#include <DlSystem/IUserBuffer.h>
#include <DlSystem/RuntimeList.h>
#include <DlSystem/UserBufferMap.h>
#include <SNPE/SNPE.h>
#include <SNPE/SNPEBuilder.h>
#include <SNPE/SNPEUtil.h>

#if SNPE_VERSION_MAJOR != 2
#error "This code targets only SNPE 2.x"
#endif

typedef struct {
} shared_data;

typedef struct {
  Snpe_SNPE_Handle_t snpe_h;
  Snpe_UserBufferMap_Handle_t inputMap_h;
  Snpe_UserBufferMap_Handle_t outputMap_h;
} hidden_data;

typedef struct {
  /* platform accessible parts */
  union {
    shared_data shared;
    char platform_accessible_part_size[2048];
  };

  /* platform inaccessible parts */
  union {
    hidden_data hidden;
    char platform_inaccessible_part_size[2048];
  };
} private_data;

int custom_function_0_cleanup (void *pdata, int arg1, int arg2)
{
  /* pdata == private_data */
  hidden_data *data = &((private_data *) pdata)->hidden;

  if (data->inputMap_h)
    Snpe_UserBufferMap_Delete (data->inputMap_h);

  if (data->outputMap_h)
    Snpe_UserBufferMap_Delete (data->outputMap_h);

  if (data->snpe_h)
    Snpe_SNPE_Delete (data->snpe_h);

  data->snpe_h = NULL;
  data->inputMap_h = NULL;
  data->outputMap_h = NULL;
}

int custom_function_1_cleanup_userbuffer (void *pdata, int arg1, int arg2)
{
  /* pdata == Snpe_IUserBuffer_Handle_t */
  Snpe_IUserBuffer_Delete ((Snpe_IUserBuffer_Handle_t) pdata);
  return 0;
}
