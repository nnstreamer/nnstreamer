/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    snpe_mock_common.cc
 * @date    22 Sep 2026
 * @brief   Bookkeeping shared by the SNPE v1 and v2 mocks.
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * Three things live here:
 * - a live-instance counter per mock object kind, so a test can assert that a
 *   sub-plugin released every handle it obtained;
 * - the description of the model the mock emulates, derived from the file name
 *   because the mock never parses a .dlc container;
 * - a ledger of the GLib string allocations made by the sub-plugin sources,
 *   built on the linker's --wrap option, so a test can assert that a failing
 *   configure leaks nothing.
 *
 * The ledger only sees calls made from objects linked into the test binary,
 * which is exactly the sub-plugin under test plus the test code itself. Calls
 * made inside libnnstreamer or libglib are not redirected.
 */

#include <string.h>

#include <gtest/gtest.h>
#include <glib.h>
#include <nnstreamer_plugin_api_util.h>

#include "snpe_mock.h"

namespace
{
/**
 * @brief Listener putting the mock back to its initial state around each case.
 *
 * The cases of unittest_filter_snpe.cc run in the same binary and know nothing
 * about the mock, so a case that injects a fault must not be able to leave it
 * behind for them. Which suite runs first is decided by the order the objects
 * initialise in, which differs between builds, so this cannot be left to the
 * cases to get right.
 */
class snpe_mock_reset_listener : public testing::EmptyTestEventListener
{
  public:
  /** @brief Put the mock back before a case runs. */
  void OnTestStart (const testing::TestInfo &) override
  {
    snpe_mock_reset ();
  }
  /** @brief Put the mock back after a case has run. */
  void OnTestEnd (const testing::TestInfo &) override
  {
    snpe_mock_reset ();
  }
};

/** @brief Adds the listener above before main() runs. */
struct snpe_mock_listener_registrar {
  /** @brief Append the listener to the listeners of gtest. */
  snpe_mock_listener_registrar ()
  {
    testing::UnitTest::GetInstance ()->listeners ().Append (new snpe_mock_reset_listener ());
  }
};

snpe_mock_listener_registrar snpe_mock_listener;
} /* namespace */

static gint live_counts[SNPE_MOCK_OBJ_MAX];
static gint over_releases;
static snpe_mock_failure injected_failure = SNPE_MOCK_FAIL_NONE;

/**
 * @brief Drop every injected fault and clear the counters.
 */
void
snpe_mock_reset (void)
{
  for (guint i = 0; i < SNPE_MOCK_OBJ_MAX; i++)
    g_atomic_int_set (&live_counts[i], 0);
  g_atomic_int_set (&over_releases, 0);
  injected_failure = SNPE_MOCK_FAIL_NONE;
  snpe_mock_ledger_reset ();
}

/**
 * @brief Make the mock report the given fault from now on.
 */
void
snpe_mock_set_failure (snpe_mock_failure failure)
{
  injected_failure = failure;
}

/**
 * @brief Get the fault the mock is currently injecting.
 */
snpe_mock_failure
snpe_mock_get_failure (void)
{
  return injected_failure;
}

/**
 * @brief Count one more live instance of the given object kind.
 */
void
snpe_mock_obj_created (snpe_mock_obj_type type)
{
  g_assert (type < SNPE_MOCK_OBJ_MAX);
  g_atomic_int_inc (&live_counts[type]);
}

/**
 * @brief Count one fewer live instance of the given object kind.
 *
 * Counting below zero is recorded rather than asserted, so that an object
 * created before the last snpe_mock_reset() does not abort the run; a test
 * that wants to see an unbalanced release asserts on
 * snpe_mock_over_release_count().
 */
void
snpe_mock_obj_destroyed (snpe_mock_obj_type type)
{
  g_assert (type < SNPE_MOCK_OBJ_MAX);

  while (TRUE) {
    gint live = g_atomic_int_get (&live_counts[type]);

    if (live <= 0) {
      g_atomic_int_inc (&over_releases);
      break;
    }
    if (g_atomic_int_compare_and_exchange (&live_counts[type], live, live - 1))
      break;
  }
}

/**
 * @brief Get the number of live instances of the given object kind.
 */
guint
snpe_mock_live_count (snpe_mock_obj_type type)
{
  g_assert (type < SNPE_MOCK_OBJ_MAX);
  return (guint) g_atomic_int_get (&live_counts[type]);
}

/**
 * @brief Get how often an object was released more often than it was created.
 */
guint
snpe_mock_over_release_count (void)
{
  return (guint) g_atomic_int_get (&over_releases);
}

/**
 * @brief Get the number of live instances of every object kind.
 */
guint
snpe_mock_total_live_count (void)
{
  guint total = 0;

  for (guint i = 0; i < SNPE_MOCK_OBJ_MAX; i++)
    total += (guint) g_atomic_int_get (&live_counts[i]);

  return total;
}

/**
 * @brief Describe the model the mock emulates for the given file.
 * @param[in] path path of the model file, whose contents are never read
 * @param[out] model the emulated model description
 * @return TRUE if the mock can emulate a model for this path
 *
 * The in-tree add2_*.dlc fixtures are emulated as a single input and a single
 * output of one element that computes output = input + 2, quantized when the
 * name says uint8. A test that needs a shape the fixtures do not have creates
 * an empty file whose name carries one of the other keywords below.
 */
gboolean
snpe_mock_model_load (const char *path, snpe_mock_model *model)
{
  g_autofree gchar *base = NULL;

  if (!path || !model || path[0] == '\0')
    return FALSE;

  base = g_path_get_basename (path);

  model->quantized = (strstr (base, "uint8") != NULL);
  model->oversized_input = (strstr (base, "oversized") != NULL);
  model->resizable = (strstr (base, "resizable") != NULL) || model->oversized_input;
  model->unsupported_encoding = (strstr (base, "badenc") != NULL);
  model->output_elements = model->resizable ? SNPE_MOCK_RESIZABLE_OUTPUT_ELEMENTS : 1;

  return TRUE;
}

#if defined(SNPE_MOCK_LEDGER)
static GHashTable *ledger;
static GMutex ledger_lock;

extern "C" {
gchar **__real_g_strsplit (const gchar *string, const gchar *delimiter, gint max_tokens);
gchar *__real_g_strjoinv (const gchar *separator, gchar **str_array);
gchar *__real_g_strdup (const gchar *str);
void __real_g_strfreev (gchar **str_array);
void __real_g_free (gpointer mem);
void __real_gst_tensors_info_free (GstTensorsInfo *info);

gchar **__wrap_g_strsplit (const gchar *string, const gchar *delimiter, gint max_tokens);
gchar *__wrap_g_strjoinv (const gchar *separator, gchar **str_array);
gchar *__wrap_g_strdup (const gchar *str);
void __wrap_g_strfreev (gchar **str_array);
void __wrap_g_free (gpointer mem);
void __wrap_gst_tensors_info_free (GstTensorsInfo *info);
}

/**
 * @brief Remember a block the sub-plugin has to release.
 */
static void
ledger_add (gpointer mem)
{
  if (!mem)
    return;

  g_mutex_lock (&ledger_lock);
  if (!ledger)
    ledger = g_hash_table_new (g_direct_hash, g_direct_equal);
  g_hash_table_add (ledger, mem);
  g_mutex_unlock (&ledger_lock);
}

/**
 * @brief Forget a block, ignoring one the ledger never held.
 */
static void
ledger_remove (gpointer mem)
{
  if (!mem)
    return;

  g_mutex_lock (&ledger_lock);
  if (ledger)
    g_hash_table_remove (ledger, mem);
  g_mutex_unlock (&ledger_lock);
}

/**
 * @brief g_strsplit() interposer that records the returned vector.
 */
gchar **
__wrap_g_strsplit (const gchar *string, const gchar *delimiter, gint max_tokens)
{
  gchar **result = __real_g_strsplit (string, delimiter, max_tokens);

  ledger_add (result);
  return result;
}

/**
 * @brief g_strjoinv() interposer that records the returned string.
 */
gchar *
__wrap_g_strjoinv (const gchar *separator, gchar **str_array)
{
  gchar *result = __real_g_strjoinv (separator, str_array);

  ledger_add (result);
  return result;
}

/**
 * @brief g_strdup() interposer that records the returned string.
 */
gchar *
__wrap_g_strdup (const gchar *str)
{
  gchar *result = __real_g_strdup (str);

  ledger_add (result);
  return result;
}

/**
 * @brief g_strfreev() interposer that clears the recorded vector.
 */
void
__wrap_g_strfreev (gchar **str_array)
{
  ledger_remove (str_array);
  __real_g_strfreev (str_array);
}

/**
 * @brief g_free() interposer that clears the recorded block.
 */
void
__wrap_g_free (gpointer mem)
{
  ledger_remove (mem);
  __real_g_free (mem);
}

/**
 * @brief gst_tensors_info_free() interposer that clears the recorded names.
 *
 * The names are released inside libnnstreamer, where the interposer does not
 * reach, so they are taken off the ledger here instead. Both arrays are walked
 * whole, exactly as the real function releases them, rather than up to the
 * tensor count, which the caller is free to have left behind. The extra array
 * is read directly rather than through gst_tensors_info_get_nth_info(), which
 * would allocate it on a structure that never had one.
 */
void
__wrap_gst_tensors_info_free (GstTensorsInfo *info)
{
  if (info) {
    for (guint i = 0; i < NNS_TENSOR_MEMORY_MAX; i++)
      ledger_remove (info->info[i].name);

    if (info->extra) {
      for (guint i = 0; i < NNS_TENSOR_SIZE_EXTRA_LIMIT; i++)
        ledger_remove (info->extra[i].name);
    }
  }

  __real_gst_tensors_info_free (info);
}

/**
 * @brief Tell whether the ledger sees what the sub-plugin allocates.
 *
 * Being compiled in is not enough: an interposer only catches a call that
 * reaches the symbol it wraps, and GLib inlines g_strdup() of a string the
 * compiler knows, which an optimised build then never calls. The first caller
 * therefore measures one duplication of a string built at run time, which is
 * the shape every duplication in the two sub-plugins has.
 */
gboolean
snpe_mock_ledger_available (void)
{
  static gint probed;

  if (g_atomic_int_get (&probed) == 0) {
    gchar probe_name[16];
    guint before;
    gchar *copy;

    g_snprintf (probe_name, sizeof (probe_name), "ledger probe");
    before = snpe_mock_ledger_live_count ();
    copy = g_strdup (probe_name);
    g_atomic_int_set (&probed, (snpe_mock_ledger_live_count () > before) ? 1 : 2);
    g_free (copy);
  }

  return g_atomic_int_get (&probed) == 1;
}

/**
 * @brief Forget every recorded block.
 */
void
snpe_mock_ledger_reset (void)
{
  g_mutex_lock (&ledger_lock);
  if (ledger)
    g_hash_table_remove_all (ledger);
  g_mutex_unlock (&ledger_lock);
}

/**
 * @brief Get the number of recorded blocks that are still not released.
 */
guint
snpe_mock_ledger_live_count (void)
{
  guint size;

  g_mutex_lock (&ledger_lock);
  size = ledger ? g_hash_table_size (ledger) : 0;
  g_mutex_unlock (&ledger_lock);

  return size;
}
#else
/**
 * @brief Tell whether the ledger is compiled in.
 */
gboolean
snpe_mock_ledger_available (void)
{
  return FALSE;
}

/**
 * @brief Do nothing; the ledger is not compiled in.
 */
void
snpe_mock_ledger_reset (void)
{
}

/**
 * @brief Report no recorded block; the ledger is not compiled in.
 */
guint
snpe_mock_ledger_live_count (void)
{
  return 0;
}

#endif /* SNPE_MOCK_LEDGER */
