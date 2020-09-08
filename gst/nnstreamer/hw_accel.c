/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer hardware accelerator availability checking
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 */
/**
 * @file	hw_accel.c
 * @date	8 September 2020
 * @brief	Common hardware acceleration availability checks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <hw_accel.h>
#include <errno.h>

#if defined(__aarch64__) || defined(__arm__)
#if defined(__TIZEN__)
#include <asm-arm/hwcap.h>
#elif defined(__ANDROID__) || defined(__linux__)
#include <asm/hwcap.h>
#endif /* __TIZEN__ */
#endif /* __arch64__ || __arm__ */

#include <sys/auxv.h>

/**
 * @brief Check if neon is supported
 * @retval 0 if supported, else -errno
 */
gint
cpu_neon_accel_available (void)
{
  gint neon_available = 0;

#if defined(__aarch64__) || defined(__arm__)
  gulong hwcap_flag;

#if defined(__aarch64__)
  hwcap_flag = HWCAP_ASIMD;
#elif defined(__arm__)
  hwcap_flag = HWCAP_NEON;
#endif /* __arch64 __ */

  if (getauxval (AT_HWCAP) & hwcap_flag) {
    neon_available = 0;
  } else {
    neon_available = -EINVAL;
  }
#endif /* __arch64__ || __arm__ */

  return neon_available;
}
