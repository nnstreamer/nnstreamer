/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    snpe_mock.h
 * @date    22 Sep 2026
 * @brief   Observability and failure-injection API of the SNPE mock.
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * The mock emulates the small part of the Qualcomm Neural Processing SDK that
 * the two tensor_filter SNPE sub-plugins call, so that both sub-plugin sources
 * can be compiled and unit tested without the proprietary SDK. No header of
 * that SDK is copied into this repository; the declarations were written from
 * the vendor's public API reference. See mock/README.md.
 *
 * The counters and the ledger below are process wide, and a pipeline reaches
 * them from its streaming thread, so both are updated atomically or under a
 * lock. The injected fault and snpe_mock_reset() are not: a test sets those
 * before each case, while nothing else is running.
 */
#ifndef __NNS_SNPE_MOCK_H__
#define __NNS_SNPE_MOCK_H__

#include <glib.h>

G_BEGIN_DECLS

/** @brief Name of the single input tensor of the emulated model. */
#define SNPE_MOCK_INPUT_NAME "input"
/** @brief Name of the single output tensor of the emulated model. */
#define SNPE_MOCK_OUTPUT_NAME "output"
/** @brief The emulated model computes output = input + SNPE_MOCK_ADDEND. */
#define SNPE_MOCK_ADDEND 2
/** @brief Elements the emulated runtime writes for a resizable output. */
#define SNPE_MOCK_RESIZABLE_OUTPUT_ELEMENTS 8
/**
 * @brief Elements of the input tensor of an oversized model.
 *
 * Larger than the output count, so that an input that does not fit its memory
 * can be told apart from an output that does not fit its own.
 */
#define SNPE_MOCK_OVERSIZED_INPUT_ELEMENTS 16

/**
 * @brief Kinds of mock object whose live instances are counted.
 */
typedef enum {
  SNPE_MOCK_OBJ_CONTAINER = 0,
  SNPE_MOCK_OBJ_SNPE,
  SNPE_MOCK_OBJ_BUILDER,
  SNPE_MOCK_OBJ_STRING_LIST,
  SNPE_MOCK_OBJ_TENSOR_SHAPE,
  SNPE_MOCK_OBJ_BUFFER_ATTRIBUTES,
  SNPE_MOCK_OBJ_USER_BUFFER,
  SNPE_MOCK_OBJ_USER_BUFFER_MAP,
  SNPE_MOCK_OBJ_ENCODING,
  SNPE_MOCK_OBJ_RUNTIME_LIST,
  SNPE_MOCK_OBJ_VERSION,
  SNPE_MOCK_OBJ_TENSOR,
  SNPE_MOCK_OBJ_MAX
} snpe_mock_obj_type;

/**
 * @brief Faults the mock injects on demand.
 */
typedef enum {
  SNPE_MOCK_FAIL_NONE = 0,
  SNPE_MOCK_FAIL_STRING_LIST_APPEND, /**< every append reports an error */
  SNPE_MOCK_FAIL_BUILD, /**< building an SNPE instance reports an error */
  SNPE_MOCK_FAIL_EXECUTE_NO_OUTPUT /**< a run produces no output tensor */
} snpe_mock_failure;

/**
 * @brief Properties of the model the mock emulates for a given file.
 */
typedef struct {
  gboolean quantized; /**< the default encoding is 8 bit quantized */
  gboolean resizable; /**< the tensor shapes carry a resizable (zero) dim */
  gboolean unsupported_encoding; /**< the default encoding has no NNS type */
  gboolean oversized_input; /**< the input tensor is larger than its buffer */
  guint output_elements; /**< elements the emulated runtime writes */
} snpe_mock_model;

/** @brief Drop every injected fault and clear the counters. */
void snpe_mock_reset (void);
/** @brief Make the mock report the given fault from now on. */
void snpe_mock_set_failure (snpe_mock_failure failure);
/** @brief Get the fault the mock is currently injecting. */
snpe_mock_failure snpe_mock_get_failure (void);

/** @brief Get the number of live instances of the given object kind. */
guint snpe_mock_live_count (snpe_mock_obj_type type);
/** @brief Get the number of live instances of every object kind. */
guint snpe_mock_total_live_count (void);
/** @brief Get how often an object was released more often than it was created. */
guint snpe_mock_over_release_count (void);

/** @brief Count one more live instance of the given object kind. */
void snpe_mock_obj_created (snpe_mock_obj_type type);
/**
 * @brief Count one fewer live instance of the given object kind.
 *
 * A release the count cannot balance is recorded instead, and reported by
 * snpe_mock_over_release_count().
 */
void snpe_mock_obj_destroyed (snpe_mock_obj_type type);

/** @brief Describe the model the mock emulates for the given file. */
gboolean snpe_mock_model_load (const char *path, snpe_mock_model *model);

/**
 * @brief Tell whether the ledger sees what the sub-plugin allocates.
 *
 * A test that duplicates a string the compiler knows must not rely on it:
 * GLib inlines g_strdup() of such a string, so an optimised build never
 * reaches the interposer. Duplicate a string built at run time instead, the
 * way both sub-plugins do.
 */
gboolean snpe_mock_ledger_available (void);
/** @brief Forget every recorded block. */
void snpe_mock_ledger_reset (void);
/** @brief Get the number of recorded blocks that are still not released. */
guint snpe_mock_ledger_live_count (void);

G_END_DECLS

#endif /* __NNS_SNPE_MOCK_H__ */
