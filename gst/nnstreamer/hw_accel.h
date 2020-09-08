/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer hardware accelerator availability checking
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 */
/**
 * @file	hw_accel.h
 * @date	8 September 2020
 * @brief	Common hardware acceleration availability header
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __G_HW_ACCEL__
#define __G_HW_ACCEL__

#include <glib.h>

/**
 * @brief Check if neon is supported
 * @retval 0 if supported, else -errno
 */
gint cpu_neon_accel_available (void);

#endif /* __G_HW_ACCEL__ */
