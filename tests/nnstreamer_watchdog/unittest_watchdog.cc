/**
 * @file        unittest_watchdog.cc
 * @date        31 Oct 2024
 * @brief       Unit test for watchdog commonm uitil.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <dlfcn.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include "../gst/nnstreamer/nnstreamer_watchdog.h"

/**
 * The watchdog handle is opaque, so the tests steer its worker into a given
 * interleaving by interposing the GLib calls libnnstreamer makes. On ELF the
 * executable's definitions below take precedence over libglib's; each wrapper
 * counts its calls, and a test that needs one skips itself when bypassed.
 */
static std::atomic<bool> hook_loop_run_delay{ false };
static std::atomic<bool> hook_loop_run_entered{ false };
static std::atomic<bool> hook_thread_new_fail{ false };
static std::atomic<bool> hook_thread_new_wait_loop{ false };
static std::atomic<bool> hook_cond_signal_delay{ false };
static std::atomic<bool> hook_track_mutex{ false };
static std::atomic<bool> hook_dispatch_lock_delay{ false };
static std::atomic<bool> hook_dispatch_lock_paused{ false };
static std::atomic<int> hook_mutex_violations{ 0 };
static std::atomic<int> hook_loop_run_calls{ 0 };
static std::atomic<int> hook_thread_new_calls{ 0 };
static std::atomic<int> hook_cond_signal_calls{ 0 };
static std::atomic<int> hook_mutex_calls{ 0 };
static std::thread::id hook_test_thread;
static std::mutex hook_cleared_lock;
static std::unordered_set<void *> hook_cleared;

/**
 * @brief Look up the next definition of a symbol, i.e., the one in libglib.
 */
static void *
_hook_next (const char *name)
{
  void *sym = dlsym (RTLD_NEXT, name);

  if (!sym)
    abort ();
  return sym;
}

/**
 * @brief Count a use of a mutex that was cleared and not initialised again.
 */
static void
_hook_check_mutex (GMutex *mutex)
{
  if (!hook_track_mutex.load ())
    return;

  hook_mutex_calls++;
  std::lock_guard<std::mutex> guard (hook_cleared_lock);
  if (hook_cleared.count (mutex))
    hook_mutex_violations++;
}

/**
 * @brief Interposed g_main_loop_run(): optionally starts the loop 5.5 s late.
 */
extern "C" void
g_main_loop_run (GMainLoop *loop)
{
  static auto real = (void (*) (GMainLoop *)) _hook_next ("g_main_loop_run");

  hook_loop_run_calls++;
  if (hook_loop_run_delay.exchange (false))
    g_usleep (5500000);
  hook_loop_run_entered = true;
  real (loop);
}

/**
 * @brief Interposed g_thread_try_new(): optionally fails, or returns only
 *        after the new thread is inside g_main_loop_run().
 */
extern "C" GThread *
g_thread_try_new (const gchar *name, GThreadFunc func, gpointer data, GError **error)
{
  static auto real = (GThread * (*) (const gchar *, GThreadFunc, gpointer, GError **) )
      _hook_next ("g_thread_try_new");
  GThread *thread;
  int i;

  hook_thread_new_calls++;
  if (hook_thread_new_fail.exchange (false)) {
    g_set_error_literal (error, G_THREAD_ERROR, G_THREAD_ERROR_AGAIN, "hooked failure");
    return NULL;
  }

  thread = real (name, func, data, error);
  if (thread && hook_thread_new_wait_loop.exchange (false)) {
    for (i = 0; i < 5000 && !hook_loop_run_entered.load (); i++)
      g_usleep (1000);
    g_usleep (100000);
  }

  return thread;
}

/**
 * @brief Interposed g_cond_signal(): optionally delays the first signal sent
 *        from a thread other than the test thread.
 */
extern "C" void
g_cond_signal (GCond *cond)
{
  static auto real = (void (*) (GCond *)) _hook_next ("g_cond_signal");

  hook_cond_signal_calls++;
  if (std::this_thread::get_id () != hook_test_thread
      && hook_cond_signal_delay.exchange (false))
    g_usleep (100000);
  real (cond);
}

/**
 * @brief Interposed g_mutex_init(): forgets a re-initialised mutex.
 */
extern "C" void
g_mutex_init (GMutex *mutex)
{
  static auto real = (void (*) (GMutex *)) _hook_next ("g_mutex_init");

  real (mutex);
  if (hook_track_mutex.load ()) {
    std::lock_guard<std::mutex> guard (hook_cleared_lock);
    hook_cleared.erase (mutex);
  }
}

/**
 * @brief Interposed g_mutex_clear(): remembers the cleared mutex.
 */
extern "C" void
g_mutex_clear (GMutex *mutex)
{
  static auto real = (void (*) (GMutex *)) _hook_next ("g_mutex_clear");

  if (hook_track_mutex.load ()) {
    std::lock_guard<std::mutex> guard (hook_cleared_lock);
    hook_cleared.insert (mutex);
  }
  real (mutex);
}

/**
 * @brief Interposed g_mutex_lock(): counts locking a cleared mutex.
 */
extern "C" void
g_mutex_lock (GMutex *mutex)
{
  static auto real = (void (*) (GMutex *)) _hook_next ("g_mutex_lock");

  _hook_check_mutex (mutex);
  if (hook_dispatch_lock_delay.load () && std::this_thread::get_id () != hook_test_thread
      && g_main_current_source () && hook_dispatch_lock_delay.exchange (false)) {
    hook_dispatch_lock_paused = true;
    g_usleep (300000);
  }
  real (mutex);
}

/**
 * @brief Interposed g_mutex_unlock(): counts unlocking a cleared mutex.
 */
extern "C" void
g_mutex_unlock (GMutex *mutex)
{
  static auto real = (void (*) (GMutex *)) _hook_next ("g_mutex_unlock");

  _hook_check_mutex (mutex);
  real (mutex);
}

/**
 * @brief Start counting uses of cleared mutexes.
 */
static void
_hook_track_start (void)
{
  {
    std::lock_guard<std::mutex> guard (hook_cleared_lock);
    hook_cleared.clear ();
  }
  hook_mutex_violations = 0;
  hook_mutex_calls = 0;
  hook_track_mutex = true;
}

/**
 * @brief Stop counting uses of cleared mutexes.
 */
static void
_hook_track_stop (void)
{
  hook_track_mutex = false;
  std::lock_guard<std::mutex> guard (hook_cleared_lock);
  hook_cleared.clear ();
}

/**
 * @brief Test for watchdog creation.
 */
TEST (NnstWatchdog, create)
{
  nns_watchdog_h watchdog_h = NULL;
  gboolean ret = FALSE;

  ret = nnstreamer_watchdog_create (&watchdog_h);
  EXPECT_EQ (TRUE, ret);

  nnstreamer_watchdog_destroy (watchdog_h);
}

/**
 * @brief Called when watchdog is triggered.
 */
static gboolean
_watchdog_trigger (gpointer ptr)
{
  guint *received = (guint *) ptr;

  if (received)
    (*received)++;
  else
    return FALSE;

  /** Trigger 10 times */
  return (*received) != 10;
}

/**
 * @brief Test for feeding watchdog.
 */
TEST (NnstWatchdog, feed)
{
  nns_watchdog_h watchdog_h = nullptr;
  gboolean ret = FALSE;
  guint *received = (guint *) g_malloc0 (sizeof (guint));
  const guint interval_ms = 50;

  ASSERT_NE (nullptr, received);

  ret = nnstreamer_watchdog_create (&watchdog_h);
  EXPECT_EQ (TRUE, ret);

  ret = nnstreamer_watchdog_feed (watchdog_h, _watchdog_trigger, interval_ms, received);
  EXPECT_EQ (TRUE, ret);

  g_usleep (1000000);
  EXPECT_EQ (10U, *received);

  nnstreamer_watchdog_destroy (watchdog_h);
  g_free (received);
}

/**
 * @brief Test for watchdog creation with invalid param.
 */
TEST (NnstWatchdog, create_n)
{
  gboolean ret = FALSE;

  ret = nnstreamer_watchdog_create (NULL);
  EXPECT_EQ (FALSE, ret);
}

/**
 * @brief Test for deeding watchdog with invalid param.
 */
TEST (NnstWatchdog, feed_1_n)
{
  nns_watchdog_h watchdog_h = nullptr;
  gboolean ret = FALSE;
  const guint interval_ms = 50;

  ret = nnstreamer_watchdog_create (&watchdog_h);
  EXPECT_EQ (TRUE, ret);

  ret = nnstreamer_watchdog_feed (NULL, _watchdog_trigger, interval_ms, NULL);
  EXPECT_EQ (FALSE, ret);

  nnstreamer_watchdog_destroy (watchdog_h);
}

/**
 * @brief Test for feeding watchdog with invalid param.
 */
TEST (NnstWatchdog, feed_2_n)
{
  nns_watchdog_h watchdog_h = nullptr;
  gboolean ret = FALSE;
  const guint interval_ms = 50;

  ret = nnstreamer_watchdog_create (&watchdog_h);
  EXPECT_EQ (TRUE, ret);

  ret = nnstreamer_watchdog_feed (watchdog_h, NULL, interval_ms, NULL);
  EXPECT_EQ (FALSE, ret);

  nnstreamer_watchdog_destroy (watchdog_h);
}

/**
 * @brief Result of a create/destroy cycle run on a helper thread.
 */
typedef struct {
  std::mutex lock;
  std::condition_variable cond;
  bool done;
  gboolean created;
  nns_watchdog_h handle;
} create_result_s;

/**
 * @brief Run create (and destroy on success) on a helper thread so that a
 *        hang fails the test instead of blocking it. The helper shares
 *        ownership of @a res, so a detached helper may still finish late.
 * @return TRUE if the helper finished within @a timeout_s seconds.
 */
static gboolean
_create_destroy_bounded (std::shared_ptr<create_result_s> res, int timeout_s)
{
  std::unique_lock<std::mutex> guard (res->lock);

  res->done = false;
  std::thread helper ([res] () {
    nns_watchdog_h handle = (nns_watchdog_h) 0x1;
    gboolean created = nnstreamer_watchdog_create (&handle);

    if (created)
      nnstreamer_watchdog_destroy (handle);

    std::lock_guard<std::mutex> g (res->lock);
    res->created = created;
    res->handle = handle;
    res->done = true;
    res->cond.notify_one ();
  });

  if (!res->cond.wait_for (guard, std::chrono::seconds (timeout_s),
          [res] () { return res->done; })) {
    helper.detach ();
    return FALSE;
  }

  guard.unlock ();
  helper.join ();
  return TRUE;
}

/**
 * @brief Test that create waits for the worker's idle callback and keeps the
 *        mutex alive when the loop is already running at the first check.
 */
TEST (NnstWatchdog, createLoopAlreadyRunning)
{
  nns_watchdog_h watchdog_h = nullptr;
  gboolean ret;
  int violations;

  hook_loop_run_entered = false;
  hook_thread_new_calls = 0;
  hook_cond_signal_calls = 0;
  _hook_track_start ();
  hook_cond_signal_delay = true;
  hook_thread_new_wait_loop = true;

  ret = nnstreamer_watchdog_create (&watchdog_h);
  EXPECT_EQ (TRUE, ret);
  EXPECT_NE (nullptr, watchdog_h);

  /* Let a callback that outlived create() finish before counting. */
  g_usleep (200000);
  violations = hook_mutex_violations.load ();

  nnstreamer_watchdog_destroy (watchdog_h);
  _hook_track_stop ();
  hook_cond_signal_delay = false;
  hook_thread_new_wait_loop = false;

  if (hook_thread_new_calls.load () == 0 || hook_cond_signal_calls.load () == 0
      || hook_mutex_calls.load () == 0) {
    GTEST_SKIP () << "GLib calls from libnnstreamer are not interposable here.";
  }

  EXPECT_TRUE (hook_loop_run_entered.load ());
  EXPECT_EQ (0, violations);
}

/**
 * @brief Test repeated create/feed/destroy with delayed idle callbacks.
 */
TEST (NnstWatchdog, createDestroyStress)
{
  nns_watchdog_h watchdog_h;
  guint received = 0;
  int i;

  _hook_track_start ();
  for (i = 0; i < 100; i++) {
    watchdog_h = nullptr;
    hook_cond_signal_delay = (i % 10 == 0);
    if (!nnstreamer_watchdog_create (&watchdog_h)) {
      ADD_FAILURE () << "Failed to create watchdog at iteration " << i;
      break;
    }
    if (i % 2) {
      EXPECT_EQ (TRUE,
          nnstreamer_watchdog_feed (watchdog_h, _watchdog_trigger, 1, &received));
    }
    nnstreamer_watchdog_destroy (watchdog_h);
  }
  hook_cond_signal_delay = false;
  EXPECT_EQ (0, hook_mutex_violations.load ());
  _hook_track_stop ();
}

/**
 * @brief Test that create returns when the worker misses the start deadline.
 */
TEST (NnstWatchdog, createTimeout_n)
{
  auto res = std::make_shared<create_result_s> ();

  hook_loop_run_calls = 0;
  hook_loop_run_delay = true;

  ASSERT_TRUE (_create_destroy_bounded (res, 20))
      << "nnstreamer_watchdog_create() hung after a start timeout.";
  hook_loop_run_delay = false;

  if (hook_loop_run_calls.load () == 0) {
    GTEST_SKIP () << "GLib calls from libnnstreamer are not interposable here.";
  }

  EXPECT_EQ (FALSE, res->created);
  EXPECT_EQ (nullptr, res->handle);
}

/**
 * @brief Count GLib critical logs.
 */
static void
_count_critical_logs (const gchar *log_domain, GLogLevelFlags log_level,
    const gchar *message, gpointer user_data)
{
  guint *count = (guint *) user_data;

  (*count)++;
}

/**
 * @brief Test that create fails cleanly when the worker thread cannot start.
 */
TEST (NnstWatchdog, createThreadFail_n)
{
  nns_watchdog_h watchdog_h = (nns_watchdog_h) 0x1;
  guint critical_count = 0;
  guint handler;
  gboolean ret;

  hook_thread_new_calls = 0;
  hook_thread_new_fail = true;
  handler = g_log_set_handler (
      "GLib", G_LOG_LEVEL_CRITICAL, _count_critical_logs, &critical_count);

  ret = nnstreamer_watchdog_create (&watchdog_h);

  g_log_remove_handler ("GLib", handler);
  hook_thread_new_fail = false;

  if (hook_thread_new_calls.load () == 0) {
    GTEST_SKIP () << "GLib calls from libnnstreamer are not interposable here.";
  }

  EXPECT_EQ (FALSE, ret);
  EXPECT_EQ (nullptr, watchdog_h);
  EXPECT_EQ (0U, critical_count);
}

/**
 * @brief State shared with a watchdog callback that blocks for a while.
 */
typedef struct {
  std::atomic<int> entered;
  std::atomic<int> finished;
} slow_callback_s;

/**
 * @brief Called when watchdog is triggered; returns only after 300 ms.
 */
static gboolean
_watchdog_slow_trigger (gpointer ptr)
{
  slow_callback_s *state = (slow_callback_s *) ptr;

  state->entered++;
  g_usleep (300000);
  state->finished++;

  return FALSE;
}

/**
 * @brief Wait until @a counter becomes non-zero, for at most @a timeout_ms.
 */
static gboolean
_wait_nonzero (std::atomic<int> &counter, guint timeout_ms)
{
  guint i;

  for (i = 0; i < timeout_ms && counter.load () == 0; i++)
    g_usleep (1000);

  return counter.load () != 0;
}

/**
 * @brief Test that release waits for the callback running in the watchdog thread.
 */
TEST (NnstWatchdog, releaseWaitsRunningCallback)
{
  nns_watchdog_h watchdog_h = nullptr;
  slow_callback_s state;

  state.entered = 0;
  state.finished = 0;

  ASSERT_EQ (TRUE, nnstreamer_watchdog_create (&watchdog_h));
  ASSERT_EQ (TRUE,
      nnstreamer_watchdog_feed (watchdog_h, _watchdog_slow_trigger, 10, &state));
  ASSERT_TRUE (_wait_nonzero (state.entered, 5000));

  nnstreamer_watchdog_release (watchdog_h);
  EXPECT_EQ (1, state.finished.load ());

  nnstreamer_watchdog_destroy (watchdog_h);
  EXPECT_EQ (1, state.entered.load ());
  EXPECT_EQ (1, state.finished.load ());
}

/**
 * @brief Test that the callback of a source released while being dispatched
 *        does not run.
 */
TEST (NnstWatchdog, releaseDuringDispatch_n)
{
  nns_watchdog_h watchdog_h = nullptr;
  bool paused;
  guint received = 0;
  guint i;

  hook_dispatch_lock_paused = false;
  ASSERT_EQ (TRUE, nnstreamer_watchdog_create (&watchdog_h));

  hook_dispatch_lock_delay = true;
  ASSERT_EQ (TRUE, nnstreamer_watchdog_feed (watchdog_h, _watchdog_trigger, 10, &received));
  for (i = 0; i < 1000 && !hook_dispatch_lock_paused.load (); i++)
    g_usleep (1000);
  paused = hook_dispatch_lock_paused.load ();

  nnstreamer_watchdog_release (watchdog_h);
  g_usleep (500000);
  nnstreamer_watchdog_destroy (watchdog_h);
  hook_dispatch_lock_delay = false;

  if (!paused) {
    GTEST_SKIP () << "The dispatch did not lock from libnnstreamer.";
  }

  EXPECT_EQ (0U, received);
}

/**
 * @brief Test that a released source is not dispatched and release is idempotent.
 */
TEST (NnstWatchdog, releaseTwice)
{
  nns_watchdog_h watchdog_h = nullptr;
  guint received = 0;

  ASSERT_EQ (TRUE, nnstreamer_watchdog_create (&watchdog_h));
  nnstreamer_watchdog_release (watchdog_h);

  ASSERT_EQ (TRUE, nnstreamer_watchdog_feed (watchdog_h, _watchdog_trigger, 50, &received));
  nnstreamer_watchdog_release (watchdog_h);
  nnstreamer_watchdog_release (watchdog_h);
  g_usleep (200000);
  EXPECT_EQ (0U, received);

  ASSERT_EQ (TRUE, nnstreamer_watchdog_feed (watchdog_h, _watchdog_trigger, 10, &received));
  g_usleep (1000000);
  EXPECT_EQ (10U, received);

  nnstreamer_watchdog_destroy (watchdog_h);
}

/**
 * @brief Test that feeding again replaces the source that is still armed.
 */
TEST (NnstWatchdog, feedTwice)
{
  nns_watchdog_h watchdog_h = nullptr;
  guint first = 9, second = 9;

  ASSERT_EQ (TRUE, nnstreamer_watchdog_create (&watchdog_h));
  ASSERT_EQ (TRUE, nnstreamer_watchdog_feed (watchdog_h, _watchdog_trigger, 100, &first));
  ASSERT_EQ (TRUE, nnstreamer_watchdog_feed (watchdog_h, _watchdog_trigger, 100, &second));
  g_usleep (500000);
  nnstreamer_watchdog_destroy (watchdog_h);

  EXPECT_EQ (9U, first);
  EXPECT_EQ (10U, second);
}

/**
 * @brief Test for releasing an invalid watchdog handle.
 */
TEST (NnstWatchdog, releaseNull_n)
{
  nnstreamer_watchdog_release (NULL);
  nnstreamer_watchdog_destroy (NULL);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  hook_test_thread = std::this_thread::get_id ();

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
