/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   nnstreamer_edge_aitt.c
 * @date   28 Mar 2022
 * @brief  Common library to support communication among devices using aitt.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <aitt_c.h>
#include "nnstreamer_edge_common.h"

typedef void *nns_edge_aitt_h;
typedef void *nns_edge_aitt_msg_h;
typedef void *nns_edge_aitt_sub_h;

/**
 * @brief Data structure for aitt handle.
 * @todo Update AITT-related handle later. This is internal struct to manage edge-handle with AITT.
 */
typedef struct
{
  nns_edge_protocol_e protocol;
  struct
  {
    nns_edge_aitt_h aitt_h;
    nns_edge_aitt_msg_h msg_h;
    nns_edge_aitt_sub_h sub_h;
  };
} nns_edge_handle_s;
