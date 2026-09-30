/**
 * @file        unittest_query_server.cc
 * @date        30 Sep 2026
 * @brief       Unit test for the tensor_query server data table
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * The test binary interposes g_try_malloc0, g_try_malloc0_n, g_free,
 * g_cond_wait_until, g_mutex_lock, nns_edge_data_create and nns_edge_send for
 * libnnstreamer (the executable's definitions win symbol lookup) so that
 * another thread can be run exactly inside the window where the server table
 * used to hand out an unprotected pointer.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <dlfcn.h>
#include <glib.h>
#include <gst/gst.h>
#include <mutex>
#include <tensor_common.h>
#include <thread>
#include "../gst/nnstreamer/tensor_query/tensor_query_server.h"

/**
 * @brief Hook state shared by the interposers and the test cases.
 */
static struct {
  std::mutex lock;
  std::condition_variable cond;
  guint id; /**< server id the racing thread works on */
  bool race_add; /**< run the racing add_data inside the armed allocation */
  bool racer_go; /**< the racing add_data may start */
  bool racer_done; /**< the racing add_data returned */
  gboolean racer_ret;
  bool waiting; /**< the waiter reached g_cond_wait_until */
  std::atomic<gpointer> watched; /**< server data allocated by the test thread */
  std::atomic<bool> user_active; /**< the test thread is inside a server call */
  std::atomic<bool> foreign_free; /**< watched data was freed by another thread under the user */
  std::atomic<bool> leak_watched; /**< keep the watched data to survive a detected defect */
  std::atomic<bool> sent; /**< the watched data user called nns_edge_send */
  std::atomic<gpointer> sent_handle; /**< edge handle given to that nns_edge_send */
} hook;

static thread_local bool arm_alloc; /**< watch the server data allocated on this thread */
static thread_local bool arm_wait; /**< hook g_cond_wait_until on this thread */
static thread_local bool arm_send; /**< hook nns_edge_data_create on this thread */
static thread_local bool arm_wake; /**< replace the wait on this thread with configure-then-remove */
static thread_local bool arm_lock; /**< hook g_mutex_lock of the watched data on this thread */
static thread_local bool fail_alloc; /**< fail the next server data allocation on this thread */
static thread_local bool is_user; /**< this thread is the watched data user */

/**
 * @brief Look up the next definition of an interposed symbol.
 */
static gpointer
_real_symbol (std::atomic<gpointer> &cache, const char *name)
{
  gpointer func = cache.load ();

  if (!func) {
    func = dlsym (RTLD_NEXT, name);
    cache.store (func);
  }

  return func;
}

/**
 * @brief Reset the hook state before a test case.
 */
static void
_reset_hook (guint id)
{
  hook.id = id;
  hook.race_add = hook.racer_go = hook.racer_done = hook.waiting = false;
  hook.racer_ret = FALSE;
  hook.watched = NULL;
  hook.user_active = false;
  hook.foreign_free = false;
  hook.leak_watched = false;
  hook.sent = false;
  hook.sent_handle = NULL;
}

/**
 * @brief When armed, watches the server data and optionally lets the racing
 *        add_data run while it is being allocated.
 */
static gpointer
_allocated (gpointer mem, gsize n_bytes)
{
  if (fail_alloc && n_bytes == sizeof (GstTensorQueryServer)) {
    fail_alloc = false;
    g_free (mem);
    return NULL;
  }

  if (arm_alloc && n_bytes == sizeof (GstTensorQueryServer)) {
    arm_alloc = false;
    hook.watched = mem;
    if (!hook.race_add)
      return mem;

    std::unique_lock<std::mutex> lk (hook.lock);
    hook.racer_go = true;
    hook.cond.notify_all ();
    /* Returns early only if the racer is not held off by the table lock. */
    hook.cond.wait_for (
        lk, std::chrono::milliseconds (500), [] { return hook.racer_done; });
    if (hook.racer_done)
      hook.leak_watched = true;
  }

  return mem;
}

/**
 * @brief Interposed g_try_malloc0 (g_try_new0 with optimization).
 */
extern "C" gpointer (g_try_malloc0) (gsize n_bytes)
{
  static std::atomic<gpointer> real;
  gpointer (*func) (gsize) = (gpointer (*) (gsize)) _real_symbol (real, "g_try_malloc0");

  return _allocated (func (n_bytes), n_bytes);
}

/**
 * @brief Interposed g_try_malloc0_n (g_try_new0 without optimization).
 */
extern "C" gpointer (g_try_malloc0_n) (gsize n_blocks, gsize n_block_bytes)
{
  static std::atomic<gpointer> real;
  gpointer (*func) (gsize, gsize)
      = (gpointer (*) (gsize, gsize)) _real_symbol (real, "g_try_malloc0_n");

  return _allocated (func (n_blocks, n_block_bytes), n_blocks * n_block_bytes);
}

/**
 * @brief Interposed g_free: records a free of the watched server data done by
 *        another thread while the test thread is still using it.
 */
extern "C" void (g_free) (gpointer mem)
{
  static std::atomic<gpointer> real;
  void (*func) (gpointer) = (void (*) (gpointer)) _real_symbol (real, "g_free");

  if (mem && mem == hook.watched.load ()) {
    if (!is_user && hook.user_active)
      hook.foreign_free = hook.leak_watched = true;
    if (hook.leak_watched)
      return;
    hook.watched = NULL;
  }

  func (mem);
}

/**
 * @brief Interposed g_cond_wait_until: tells the test the waiter is waiting,
 *        or stands in for a wait during which the data is configured and
 *        then removed before the waiter gets its lock back.
 */
extern "C" gboolean
g_cond_wait_until (GCond *cond, GMutex *mutex, gint64 end_time)
{
  static std::atomic<gpointer> real;
  gboolean (*func) (GCond *, GMutex *, gint64)
      = (gboolean (*) (GCond *, GMutex *, gint64)) _real_symbol (real, "g_cond_wait_until");

  if (arm_wake) {
    guint id = hook.id;

    arm_wake = false;
    g_mutex_unlock (mutex);
    std::thread peer ([id] {
      gst_tensor_query_server_set_configured (id);
      gst_tensor_query_server_remove_data (id);
    });
    peer.join ();
    g_mutex_lock (mutex);
    return TRUE;
  }

  if (arm_wait) {
    std::lock_guard<std::mutex> lk (hook.lock);

    arm_wait = false;
    hook.waiting = true;
    hook.cond.notify_all ();
  }

  return func (cond, mutex, end_time);
}

/**
 * @brief Interposed nns_edge_data_create: removes the server data from another
 *        thread while send_buffer holds it.
 */
extern "C" int
nns_edge_data_create (nns_edge_data_h *data_h)
{
  static std::atomic<gpointer> real;
  int (*func) (nns_edge_data_h *)
      = (int (*) (nns_edge_data_h *)) _real_symbol (real, "nns_edge_data_create");

  if (arm_send) {
    guint id = hook.id;

    arm_send = false;
    std::thread remover ([id] { gst_tensor_query_server_remove_data (id); });
    remover.join ();
  }

  return func (data_h);
}

/**
 * @brief Interposed nns_edge_send: records the edge handle the user sends with.
 */
extern "C" int
nns_edge_send (nns_edge_h edge_h, nns_edge_data_h data_h)
{
  static std::atomic<gpointer> real;
  int (*func) (nns_edge_h, nns_edge_data_h)
      = (int (*) (nns_edge_h, nns_edge_data_h)) _real_symbol (real, "nns_edge_send");

  if (is_user) {
    hook.sent_handle = edge_h;
    hook.sent = true;
  }

  return func (edge_h, data_h);
}

/**
 * @brief Interposed g_mutex_lock: removes the watched server data from another
 *        thread right before the user locks it.
 */
extern "C" void
g_mutex_lock (GMutex *mutex)
{
  static std::atomic<gpointer> real;
  void (*func) (GMutex *) = (void (*) (GMutex *)) _real_symbol (real, "g_mutex_lock");
  GstTensorQueryServer *data;

  if (arm_lock && (data = (GstTensorQueryServer *) hook.watched.load ())
      && mutex == &data->lock) {
    guint id = hook.id;

    arm_lock = false;
    std::thread remover ([id] { gst_tensor_query_server_remove_data (id); });
    remover.join ();
  }

  func (mutex);
}

/**
 * @brief Run a server call on the calling thread as the watched data user.
 */
template <typename F>
static auto
_as_user (F call) -> decltype (call ())
{
  is_user = true;
  hook.user_active = true;
  auto ret = call ();
  hook.user_active = false;
  is_user = false;

  return ret;
}

/**
 * @brief Add server data of the id and watch its allocation.
 */
static bool
_add_watched (guint id)
{
  gboolean added;

  _reset_hook (id);
  arm_alloc = true;
  added = gst_tensor_query_server_add_data (id);
  arm_alloc = false;

  return added && hook.watched.load () != nullptr;
}

/**
 * @brief Adding, configuring and removing server data. Every call gives its
 *        reference back, so the removal frees the data.
 */
TEST (tensorQueryServer, dataLifecycle)
{
  const guint id = 5100U;

  ASSERT_TRUE (_add_watched (id));
  EXPECT_TRUE (gst_tensor_query_server_add_data (id));

  gst_tensor_query_server_set_configured (id);
  EXPECT_TRUE (gst_tensor_query_server_wait_sink (id));
  EXPECT_TRUE (gst_tensor_query_server_prepare (id, NNS_EDGE_CONNECT_TYPE_TCP, NULL));
  gst_tensor_query_server_set_caps (id, "caps");
  gst_tensor_query_server_release_edge_handle (id);

  gst_tensor_query_server_remove_data (id);
  EXPECT_EQ (hook.watched.load (), nullptr);
  EXPECT_FALSE (gst_tensor_query_server_wait_sink (id));
  gst_tensor_query_server_remove_data (id);
}

/**
 * @brief A waiter is woken up when the sink configures the server data.
 */
TEST (tensorQueryServer, waitSinkConfiguredLater)
{
  const guint id = 5101U;
  gboolean ret = FALSE;

  ASSERT_TRUE (_add_watched (id));

  std::thread waiter ([&ret, id] {
    arm_wait = true;
    ret = gst_tensor_query_server_wait_sink (id);
    arm_wait = false;
  });

  {
    std::unique_lock<std::mutex> lk (hook.lock);
    EXPECT_TRUE (hook.cond.wait_for (
        lk, std::chrono::seconds (5), [] { return hook.waiting; }));
  }
  gst_tensor_query_server_set_configured (id);
  waiter.join ();

  EXPECT_TRUE (ret);
  gst_tensor_query_server_remove_data (id);
  EXPECT_EQ (hook.watched.load (), nullptr);
}

/**
 * @brief Calls on an id that has no server data fail without side effects.
 */
TEST (tensorQueryServer, unknownId_n)
{
  const guint id = 5102U;
  GstBuffer *buffer = gst_buffer_new ();

  gst_buffer_add_meta_query (buffer);

  EXPECT_FALSE (gst_tensor_query_server_wait_sink (id));
  EXPECT_FALSE (gst_tensor_query_server_send_buffer (id, buffer));
  EXPECT_FALSE (gst_tensor_query_server_prepare (id, NNS_EDGE_CONNECT_TYPE_TCP, NULL));
  gst_tensor_query_server_set_configured (id);
  gst_tensor_query_server_set_caps (id, "caps");
  gst_tensor_query_server_release_edge_handle (id);
  gst_tensor_query_server_remove_data (id);

  /* The calls above must not have created the data. */
  EXPECT_FALSE (gst_tensor_query_server_wait_sink (id));

  gst_buffer_unref (buffer);
}

/**
 * @brief send_buffer refuses a buffer without query meta or without an edge handle.
 */
TEST (tensorQueryServer, sendBufferInvalid_n)
{
  const guint id = 5103U;
  GstBuffer *buffer = gst_buffer_new_allocate (NULL, 4, NULL);

  ASSERT_TRUE (_add_watched (id));

  EXPECT_FALSE (gst_tensor_query_server_send_buffer (id, buffer));
  gst_buffer_add_meta_query (buffer);
  EXPECT_FALSE (gst_tensor_query_server_send_buffer (id, buffer));

  gst_tensor_query_server_remove_data (id);
  EXPECT_EQ (hook.watched.load (), nullptr);
  gst_buffer_unref (buffer);
}

/**
 * @brief A failed allocation fails add_data and leaves the table usable.
 */
TEST (tensorQueryServer, addDataAllocFail_n)
{
  const guint id = 5107U;

  fail_alloc = true;
  EXPECT_FALSE (gst_tensor_query_server_add_data (id));
  EXPECT_FALSE (fail_alloc);
  EXPECT_FALSE (gst_tensor_query_server_wait_sink (id));

  ASSERT_TRUE (_add_watched (id));
  gst_tensor_query_server_remove_data (id);
  EXPECT_EQ (hook.watched.load (), nullptr);
  _reset_hook (0);
}

/**
 * @brief prepare must not create an edge handle on server data that was
 *        removed while the caller held it.
 */
TEST (tensorQueryServer, prepareAfterRemove_n)
{
  const guint id = 5108U;
  gboolean ret;

  ASSERT_TRUE (_add_watched (id));

  arm_lock = true;
  ret = _as_user ([id] {
    return gst_tensor_query_server_prepare (id, NNS_EDGE_CONNECT_TYPE_TCP, NULL);
  });
  EXPECT_FALSE (arm_lock);
  arm_lock = false;

  EXPECT_FALSE (ret);
  EXPECT_FALSE (hook.foreign_free);
  EXPECT_EQ (hook.watched.load (), nullptr);
  _reset_hook (0);
}

/**
 * @brief Two add_data calls of one id racing each other must both succeed and
 *        leave a single live entry (check-then-insert used to free the table's
 *        value and fail one of them).
 */
TEST (tensorQueryServer, concurrentAddData)
{
  const guint id = 5104U;
  gboolean ret;

  _reset_hook (id);
  hook.race_add = true;

  std::thread racer ([] {
    std::unique_lock<std::mutex> lk (hook.lock);
    if (!hook.cond.wait_for (
            lk, std::chrono::seconds (5), [] { return hook.racer_go; }))
      return;
    lk.unlock ();

    gboolean r = gst_tensor_query_server_add_data (hook.id);

    lk.lock ();
    hook.racer_ret = r;
    hook.racer_done = true;
    hook.cond.notify_all ();
  });

  arm_alloc = true;
  ret = gst_tensor_query_server_add_data (id);
  arm_alloc = false;
  racer.join ();

  EXPECT_TRUE (ret);
  EXPECT_TRUE (hook.racer_ret);
  /* The racer must have been held off until the insert was done. */
  EXPECT_FALSE (hook.leak_watched);

  gst_tensor_query_server_set_configured (id);
  EXPECT_TRUE (gst_tensor_query_server_wait_sink (id));
  gst_tensor_query_server_remove_data (id);
  EXPECT_EQ (hook.watched.load (), nullptr);

  _reset_hook (0);
}

/**
 * @brief Removing the server data while a waiter holds it must wake the waiter
 *        up and must not free the data under it.
 */
TEST (tensorQueryServer, removeDuringWaitSink_n)
{
  const guint id = 5105U;
  gboolean ret = TRUE;
  gint64 elapsed = 0;

  ASSERT_TRUE (_add_watched (id));

  std::thread waiter ([&ret, &elapsed, id] {
    gint64 start = g_get_monotonic_time ();

    arm_wait = true;
    ret = _as_user ([id] { return gst_tensor_query_server_wait_sink (id); });
    arm_wait = false;
    elapsed = g_get_monotonic_time () - start;
  });

  {
    std::unique_lock<std::mutex> lk (hook.lock);
    EXPECT_TRUE (hook.cond.wait_for (
        lk, std::chrono::seconds (5), [] { return hook.waiting; }));
  }
  gst_tensor_query_server_remove_data (id);
  waiter.join ();

  EXPECT_FALSE (ret);
  EXPECT_FALSE (hook.foreign_free);
  EXPECT_EQ (hook.watched.load (), nullptr);
  EXPECT_LT (elapsed, DEFAULT_QUERY_INFO_TIMEOUT * G_TIME_SPAN_SECOND);

  _reset_hook (0);
}

/**
 * @brief A waiter woken up by the configuration must still fail if the server
 *        data was removed before it got its lock back.
 */
TEST (tensorQueryServer, removeAfterConfigureDuringWaitSink_n)
{
  const guint id = 5109U;
  gboolean ret;

  ASSERT_TRUE (_add_watched (id));

  arm_wake = true;
  ret = _as_user ([id] { return gst_tensor_query_server_wait_sink (id); });
  EXPECT_FALSE (arm_wake);
  arm_wake = false;

  EXPECT_FALSE (ret);
  EXPECT_FALSE (hook.foreign_free);
  EXPECT_EQ (hook.watched.load (), nullptr);
  _reset_hook (0);
}

/**
 * @brief Removing the server data while send_buffer holds it must release the
 *        edge handle at once, so the sender cannot send with it, and must not
 *        free the data under the sender.
 */
TEST (tensorQueryServer, removeDuringSendBuffer_n)
{
  const guint id = 5106U;
  GstBuffer *buffer = gst_buffer_new_allocate (NULL, 4, NULL);
  gboolean ret;

  gst_buffer_add_meta_query (buffer);
  ASSERT_TRUE (_add_watched (id));
  ASSERT_TRUE (gst_tensor_query_server_prepare (id, NNS_EDGE_CONNECT_TYPE_TCP, NULL));

  arm_send = true;
  ret = _as_user (
      [id, buffer] { return gst_tensor_query_server_send_buffer (id, buffer); });
  arm_send = false;

  EXPECT_FALSE (ret);
  EXPECT_TRUE (hook.sent);
  EXPECT_EQ (hook.sent_handle.load (), nullptr);
  EXPECT_FALSE (hook.foreign_free);
  /* The sender dropped the last reference. */
  EXPECT_EQ (hook.watched.load (), nullptr);
  EXPECT_FALSE (gst_tensor_query_server_wait_sink (id));

  gst_buffer_unref (buffer);
  _reset_hook (0);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
