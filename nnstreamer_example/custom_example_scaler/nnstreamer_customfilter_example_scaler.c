/**
 * NNStreamer Custom Filter Example 3. Scaler
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * LICENSE: LGPL-2.1
 *
 * @file  nnstreamer_customfilter_example_scaler.c
 * @date  22 Jun 2018
 * @brief  Custom NNStreamer Filter Example 3. "Scaler"
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This scales a tensor of [N][y][x][M] to [N][new-y][new-x][M]
 *
 * The custom property is to be given as, "custom=[new-x]x[new-y]", where new-x and new-y are unsigned integers.
 * E.g., custom=640x480
 *
 * Output[y'][x'] = Input[ y' * y / new-y ][ x' * x / new-x ]. Yeah This is Way too Simple. But this is just an example :D
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <tensor_filter_custom.h>

typedef struct _pt_data
{
  uint32_t id; /***< Just for testing */
  char *property; /***< The string given as "custom" property of tensor_filter element */
  uint32_t new_y;
  uint32_t new_x;
} pt_data;

static void *
pt_init (const GstTensor_Filter_Properties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));

  if (prop->customProperties && strlen (prop->customProperties) > 0)
    data->property = strdup (prop->customProperties);
  else
    data->property = NULL;
  data->new_x = 0;
  data->new_y = 0;

  /* Parse property and set new_x, new_y */
  if (data->property) {
    const char s[6] = "Xx:_ ";
    char *token;
    char *saveptr;

    token = strtok_r (data->property, s, &saveptr);

    if (token != NULL) {
      /* The first part */
      data->new_x = atoi (token);
      if (data->new_x < 0)
        data->new_x = 0;
      token = strtok_r (NULL, s, &saveptr);
      if (token != NULL) {
        /* The second part */
        data->new_y = atoi (token);
        if (data->new_y < 0)
          data->new_y = 0;
      }
    }
  }

  data->id = 0;
  return data;
}

static void
pt_exit (void *private_data, const GstTensor_Filter_Properties * prop)
{
  pt_data *data = private_data;
  assert (data);
  if (data->property)
    free (data->property);
  free (data);
}

static int
set_inputDim (void *private_data, const GstTensor_Filter_Properties * prop,
    const tensor_dim iDim, const tensor_type iType,
    tensor_dim oDim, tensor_type * oType)
{
  int i;
  pt_data *data = private_data;
  assert (data);

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    oDim[i] = iDim[i];

  /* Update [1] and [2] oDim with new-x, new-y */
  if (data->new_x > 0)
    oDim[1] = data->new_x;
  if (data->new_y > 0)
    oDim[2] = data->new_y;

  *oType = iType;
  return 0;
}

static int
pt_invoke (void *private_data, const GstTensor_Filter_Properties * prop,
    const uint8_t * inptr, uint8_t * outptr)
{
  pt_data *data = private_data;
  uint32_t ox, oy, x, y, z;
  uint32_t oidx0, oidx1, oidx2;
  uint32_t iidx0, iidx1, iidx2;

  assert (data);
  assert (inptr);
  assert (outptr);

  /* This assumes the limit is 4 */
  assert (NNS_TENSOR_RANK_LIMIT == 4);

  assert (prop->inputDimension[0] == prop->outputDimension[0]);
  assert (prop->inputDimension[3] == prop->outputDimension[3]);
  assert (prop->inputType == prop->outputType);

  ox = (data->new_x > 0) ? data->new_x : prop->outputDimension[1];
  oy = (data->new_y > 0) ? data->new_y : prop->outputDimension[2];

  oidx0 = prop->outputDimension[0];
  oidx1 = oidx0 * prop->outputDimension[1];
  oidx2 = oidx1 * prop->outputDimension[2];

  iidx0 = prop->inputDimension[0];
  iidx1 = iidx0 * prop->inputDimension[1];
  iidx2 = iidx1 * prop->inputDimension[2];

  for (z = 0; z < prop->inputDimension[3]; z++) {
    for (y = 0; y < oy; y++) {
      for (x = 0; x < ox; x++) {
        unsigned int c;
        for (c = 0; c < prop->inputDimension[0]; c++) {
          /* Output[y'][x'] = Input[ y' * y / new-y ][ x' * x / new-x ]. Yeah This is Way too Simple. But this is just an example :D */
          unsigned ix, iy;

          ix = x * prop->inputDimension[1] / ox;
          iy = y * prop->inputDimension[2] / oy;

          assert (ix >= 0 && iy >= 0 && ix < prop->inputDimension[1]
              && iy < prop->inputDimension[2]);

          /* outptr[z][y][x][c] = inptr[z][iy][ix][c]; */
          *(outptr + c + x * oidx0 + y * oidx1 + z * oidx2) =
              *(inptr + c + ix * iidx0 + iy * iidx1 + z * iidx2);
        }
      }
    }
  }

  assert (inptr != outptr);

  return 0;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = pt_init,
  .exitfunc = pt_exit,
  .setInputDim = set_inputDim,
  .invoke = pt_invoke,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
