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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "nnstreamer_edge.h"
#include "aitt_c.h"
#include <stdio.h>
#include <stdlib.h>

#define DEBUG	0
#define DEFAULT_QOS 0

#define debug_print(fmt, ...) \
        do { if (DEBUG) fprintf(stderr, "%s:%d:%s(): " fmt "\n", __FILE__, \
                                __LINE__, __func__, ##__VA_ARGS__); } while (0)

typedef void *nns_edge_aitt_h;
typedef void *nns_edge_aitt_msg_h;
typedef void *nns_edge_aitt_sub_h;

/**
 * @brief Data structure for aitt handle.
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

/**
 * @brief Structure for commnunication data.
 */
typedef struct
{
  char *topic;
  void *data;
  size_t data_len;
  nns_edge_data_type_e dtype;
} nns_edge_data_s;
