/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    ntputil.h
 * @date    28 Jul 2021
 * @brief   A header file of NTP utility functions
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <stdint.h>
#include <time.h>

/**
 * @brief Get NTP timestamps from the given or public NTP servers
 * @param[in] hnums A number of hostname and port pairs. If 0 is given,
 *                  the NTP server pool will be used.
 * @param[in] hnames A list of hostname
 * @param[in] ports A list of port
 * @return an Unix epoch time as microseconds on success,
 *         negative values on error
 */
int64_t
ntputil_get_epoch (uint32_t hnums, char **hnames, uint16_t * ports);
