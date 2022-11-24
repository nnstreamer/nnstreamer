/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file	nnstreamer_log.c
 * @date	31 Mar 2022
 * @brief	Internal log and error handling for NNStreamer plugins and core codes.
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef __ANDROID__
#ifndef _NO_EXECINFO_
#define _NO_EXECINFO_
#endif
#endif

#ifndef _NO_EXECINFO_
/* Android does not have execinfo.h. It has unwind.h instead. */
#include <execinfo.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <glib.h>

#include "nnstreamer_log.h"

/**
 * @brief stack trace as a string for error messages
 * @return a string of stacktrace result. caller should free it.
 * @todo The .c file location of this function might be not appropriate.
 */
char *
_backtrace_to_string (void)
{
  char *retstr = NULL;
#ifndef _NO_EXECINFO_
/* Android does not have execinfo.h. It has unwind.h instead. */
  void *array[20];
  char **strings;
  int size, i, len;
  int strsize = 0, strcursor = 0;

  size = backtrace (array, 20);
  strings = backtrace_symbols (array, size);
  if (strings != NULL) {
    for (i = 0; i < size; i++)
      strsize += strlen (strings[i]);

    retstr = malloc (sizeof (char) * (strsize + 1));
    if (retstr) {
      for (i = 0; i < size; i++) {
        len = strlen (strings[i]);
        memcpy (retstr + strcursor, strings[i], len);
        strcursor += len;
      }

      retstr[strsize] = '\0';
    }

    free (strings);
  }
#else
  retstr = strdup ("Android-nnstreamer does not support backtrace.\n");
#endif

  return retstr;
}

#define _NNSTREAMER_ERROR_LENGTH        (4096U)
static char errmsg[_NNSTREAMER_ERROR_LENGTH] = { 0 };

static int errmsg_reported = 0;
G_LOCK_DEFINE_STATIC (errlock);

/**
 * @brief return the last internal error string and clean it.
 * @return a string of error. Do not free the returned string.
 */
const char *
_nnstreamer_error (void)
{

  G_LOCK (errlock);
  if (errmsg_reported || errmsg[0] == '\0') {
    G_UNLOCK (errlock);
    return NULL;
  }
  G_UNLOCK (errlock);

  errmsg_reported = 1;
  return errmsg;
}

/**
 * @brief overwrites the error message buffer with the new message.
 */
__attribute__((__format__ (__printf__, 1, 2)))
     void _nnstreamer_error_write (const char *fmt, ...)
{
  /** The attribute is for clang workaround in macos:
      https://stackoverflow.com/questions/20167124/vsprintf-and-vsnprintf-wformat-nonliteral-warning-on-clang-5-0
   */
  va_list arg_ptr;
  G_LOCK (errlock);

  va_start (arg_ptr, fmt);
  vsnprintf (errmsg, _NNSTREAMER_ERROR_LENGTH, fmt, arg_ptr);
  va_end (arg_ptr);

  errmsg_reported = 0;

  G_UNLOCK (errlock);
}

/**
 * @brief cleans up the error message buffer.
 */
void
_nnstreamer_error_clean (void)
{
  G_LOCK (errlock);

  errmsg[0] = '\0';

  G_UNLOCK (errlock);
}
