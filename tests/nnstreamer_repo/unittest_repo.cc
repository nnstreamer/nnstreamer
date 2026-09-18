/**
 * @file        unittest_repo.cc
 * @date        15 Sep 2026
 * @brief       Unit test for the tensor repo shared by tensor_reposink / tensor_reposrc
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <string.h>
#include <unittest_util.h>
#include "../gst/nnstreamer/elements/gsttensor_repo.h"
#include "../gst/nnstreamer/elements/gsttensor_reposrc.h"

/**
 * @brief Caps used to push/pull a single 4-byte uint8 tensor through the repo.
 */
#define TEST_REPO_CAPS \
  "other/tensors,format=static,num_tensors=1,dimensions=4,types=uint8,framerate=0/1"

/**
 * @brief Deadline for a worker thread that may be blocked in the repo to finish.
 */
#define REPO_WAIT_TIMEOUT_US (5 * G_USEC_PER_SEC)

/**
 * @brief Rendezvous used to let a worker's blocking call finish (or time out)
 *        without hanging the test binary.
 */
typedef struct {
  GMutex lock;
  GCond cond;
  gboolean finished;
  gint started; /* accessed via g_atomic_int_* only */
} WaitCtx;

/**
 * @brief Initialize a WaitCtx before starting its worker thread.
 */
static void
wait_ctx_init (WaitCtx *ctx)
{
  g_mutex_init (&ctx->lock);
  g_cond_init (&ctx->cond);
  ctx->finished = FALSE;
  g_atomic_int_set (&ctx->started, 0);
}

/**
 * @brief Release a WaitCtx's glib resources. Only safe once the worker joined.
 */
static void
wait_ctx_clear (WaitCtx *ctx)
{
  g_mutex_clear (&ctx->lock);
  g_cond_clear (&ctx->cond);
}

/**
 * @brief Mark, from the worker thread, that it is about to make the call that
 *        may block inside the repo.
 */
static void
wait_ctx_mark_started (WaitCtx *ctx)
{
  g_atomic_int_set (&ctx->started, 1);
}

/**
 * @brief Block the calling (main) thread until the worker has marked started.
 * @return TRUE if it started within the deadline.
 */
static gboolean
wait_ctx_wait_started (WaitCtx *ctx, gint64 timeout_us)
{
  gint64 end_time = g_get_monotonic_time () + timeout_us;

  while (!g_atomic_int_get (&ctx->started)) {
    if (g_get_monotonic_time () >= end_time)
      return FALSE;
    g_usleep (1000);
  }

  return TRUE;
}

/**
 * @brief Mark, from the worker thread, that its call has returned.
 */
static void
wait_ctx_mark_finished (WaitCtx *ctx)
{
  g_mutex_lock (&ctx->lock);
  ctx->finished = TRUE;
  g_cond_signal (&ctx->cond);
  g_mutex_unlock (&ctx->lock);
}

/**
 * @brief Wait for the worker to finish within a deadline.
 * @return TRUE if it finished in time, FALSE on timeout (the worker is then
 *         presumed stuck and must not be joined).
 */
static gboolean
wait_ctx_join_or_timeout (WaitCtx *ctx, gint64 timeout_us)
{
  gint64 end_time = g_get_monotonic_time () + timeout_us;
  gboolean finished;

  g_mutex_lock (&ctx->lock);
  while (!ctx->finished) {
    if (!g_cond_wait_until (&ctx->cond, &ctx->lock, end_time))
      break;
  }
  finished = ctx->finished;
  g_mutex_unlock (&ctx->lock);

  return finished;
}

/**
 * @brief Which repo call a RepoJob's worker thread should make.
 */
typedef enum { OP_SET_BUFFER, OP_GET_BUFFER } RepoOp;

/**
 * @brief One gst_tensor_repo_set_buffer() / gst_tensor_repo_get_buffer() call
 *        run on a worker thread, with its arguments and result.
 */
typedef struct {
  WaitCtx ctx;
  RepoOp op;
  guint slot;

  /* OP_SET_BUFFER */
  GstBuffer *in_buffer;
  GstCaps *in_caps;
  gboolean set_ret;

  /* OP_GET_BUFFER */
  GstBuffer *out_buffer;
  gboolean eos;
  guint newid;
  GstCaps *out_caps;
} RepoJob;

/**
 * @brief Worker thread function: run the RepoJob's configured repo call.
 */
static gpointer
repo_job_thread (gpointer data)
{
  RepoJob *job = (RepoJob *) data;

  wait_ctx_mark_started (&job->ctx);

  if (job->op == OP_SET_BUFFER) {
    job->set_ret = gst_tensor_repo_set_buffer (job->slot, job->in_buffer, job->in_caps);
  } else {
    job->eos = FALSE;
    job->newid = 0;
    job->out_caps = NULL;
    job->out_buffer = gst_tensor_repo_get_buffer (
        job->slot, &job->eos, &job->newid, &job->out_caps);
  }

  wait_ctx_mark_finished (&job->ctx);
  return NULL;
}

/**
 * @brief Allocate a job to run on a worker thread.
 * @details A worker that is still blocked when its case gives up keeps writing
 *          to the job, so the job may not live in the case's stack frame. Such
 *          a job is left allocated on purpose; a joined one is freed.
 */
static RepoJob *
repo_job_new (RepoOp op, guint slot)
{
  RepoJob *job = g_new0 (RepoJob, 1);

  wait_ctx_init (&job->ctx);
  job->op = op;
  job->slot = slot;

  return job;
}

/**
 * @brief Release a job. Only safe once its worker thread has been joined.
 */
static void
repo_job_free (RepoJob *job)
{
  wait_ctx_clear (&job->ctx);
  g_free (job);
}

/**
 * @brief Pad probe counting the buffers that reach an element.
 */
static GstPadProbeReturn
count_buffer_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  (void) pad;
  (void) info;
  g_atomic_int_inc ((guint *) user_data);

  return GST_PAD_PROBE_OK;
}

/**
 * @brief Log handler counting what set_buffer() logs for a missing slot.
 * @details A case that means to exercise a blocked set_buffer() must not take
 *          that early return instead.
 */
static void
count_missing_slot_log (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer user_data)
{
  if (message && strstr (message, "gst_tensor_repo_set_buffer") != NULL)
    g_atomic_int_inc ((guint *) user_data);

  g_log_default_handler (domain, level, message, NULL);
}

/**
 * @brief Barrier releasing several threads together, to maximize the chance
 *        of them racing on the same repo call.
 */
typedef struct {
  GMutex lock;
  GCond cond;
  gboolean go;
} StartBarrier;

/**
 * @brief Block until the barrier is released.
 */
static void
start_barrier_wait (StartBarrier *b)
{
  g_mutex_lock (&b->lock);
  while (!b->go)
    g_cond_wait (&b->cond, &b->lock);
  g_mutex_unlock (&b->lock);
}

/**
 * @brief Release every thread currently blocked on the barrier.
 */
static void
start_barrier_release (StartBarrier *b)
{
  g_mutex_lock (&b->lock);
  b->go = TRUE;
  g_cond_broadcast (&b->cond);
  g_mutex_unlock (&b->lock);
}

/**
 * @brief Add the same range of slots from one thread, used by
 *        addSameSlotConcurrently to reproduce C25.
 */
typedef struct {
  StartBarrier *barrier;
  guint base;
  guint count;
  gboolean is_sink;
  gint *failures; /* shared, updated via g_atomic_int_inc */
} AddJob;

/**
 * @brief Worker thread function for addSameSlotConcurrently.
 */
static gpointer
add_concurrent_thread (gpointer data)
{
  AddJob *job = (AddJob *) data;
  guint i;

  start_barrier_wait (job->barrier);

  for (i = 0; i < job->count; i++) {
    if (!gst_tensor_repo_add_repodata (job->base + i, job->is_sink))
      g_atomic_int_inc (job->failures);
  }

  return NULL;
}

/**
 * @brief Add a slot as both a sink and a src, set/check eos, then remove it.
 */
TEST (tensorRepo, addAndRemoveSlot)
{
  const guint slot = 1000;

  gst_tensor_repo_init ();

  EXPECT_TRUE (gst_tensor_repo_add_repodata (slot, TRUE));
  EXPECT_TRUE (gst_tensor_repo_add_repodata (slot, FALSE));
  EXPECT_TRUE (gst_tensor_repo_set_eos (slot));
  EXPECT_TRUE (gst_tensor_repo_check_eos (slot));
  EXPECT_TRUE (gst_tensor_repo_remove_repodata (slot));
}

/**
 * @brief A buffer pushed with set_buffer() comes back byte-identical (but a
 *        deep copy) from get_buffer(), with matching caps.
 */
TEST (tensorRepo, bufferRoundTrip)
{
  const guint slot = 1001;
  const guint8 bytes[4] = { 1, 2, 3, 4 };
  GstCaps *caps, *out_caps = NULL;
  GstBuffer *in_buf, *out_buf;
  GstMapInfo info;
  gboolean eos = FALSE;
  guint newid = 0;

  gst_tensor_repo_init ();
  ASSERT_TRUE (gst_tensor_repo_add_repodata (slot, TRUE));

  caps = gst_caps_from_string (TEST_REPO_CAPS);
  ASSERT_NE (caps, nullptr);
  in_buf = gst_buffer_new_allocate (NULL, sizeof (bytes), NULL);
  gst_buffer_fill (in_buf, 0, bytes, sizeof (bytes));

  ASSERT_TRUE (gst_tensor_repo_set_buffer (slot, in_buf, caps));

  out_buf = gst_tensor_repo_get_buffer (slot, &eos, &newid, &out_caps);
  ASSERT_NE (out_buf, nullptr);
  EXPECT_NE (out_buf, in_buf);
  EXPECT_FALSE (eos);
  ASSERT_NE (out_caps, nullptr);
  EXPECT_TRUE (gst_caps_is_equal (out_caps, caps));

  ASSERT_TRUE (gst_buffer_map (out_buf, &info, GST_MAP_READ));
  EXPECT_EQ (sizeof (bytes), info.size);
  EXPECT_EQ (0, memcmp (info.data, bytes, sizeof (bytes)));
  gst_buffer_unmap (out_buf, &info);

  gst_buffer_unref (out_buf);
  gst_caps_unref (out_caps);
  gst_buffer_unref (in_buf);
  gst_caps_unref (caps);
  EXPECT_TRUE (gst_tensor_repo_remove_repodata (slot));
}

/**
 * @brief A get_buffer() blocked on an empty slot returns NULL with the new id
 *        once gst_tensor_repo_set_changed() marks the slot's src changed.
 */
TEST (tensorRepo, changedSlotEndsWait)
{
  const guint slot = 1002;
  const guint newslot = 1003;
  RepoJob *job;
  GThread *thread;
  guint id = 0;

  gst_tensor_repo_init ();
  ASSERT_TRUE (gst_tensor_repo_add_repodata (slot, FALSE));

  job = repo_job_new (OP_GET_BUFFER, slot);

  thread = g_thread_new ("changedSlotEndsWait", repo_job_thread, job);
  ASSERT_TRUE (wait_ctx_wait_started (&job->ctx, REPO_WAIT_TIMEOUT_US));
  g_usleep (200 * 1000);

  ASSERT_TRUE (gst_tensor_repo_set_changed (slot, newslot, FALSE));

  if (!wait_ctx_join_or_timeout (&job->ctx, REPO_WAIT_TIMEOUT_US))
    FAIL () << "get_buffer() did not return after set_changed()";
  g_thread_join (thread);

  EXPECT_EQ (job->out_buffer, nullptr);
  EXPECT_EQ (job->newid, newslot);

  id = 0;
  EXPECT_TRUE (gst_tensor_repo_check_changed (slot, &id, FALSE));
  EXPECT_EQ (id, newslot);
  EXPECT_FALSE (gst_tensor_repo_check_changed (slot, &id, TRUE));

  ASSERT_TRUE (gst_tensor_repo_set_changed (slot, newslot, TRUE));
  id = 0;
  EXPECT_TRUE (gst_tensor_repo_check_changed (slot, &id, TRUE));
  EXPECT_EQ (id, newslot);

  repo_job_free (job);
  EXPECT_TRUE (gst_tensor_repo_remove_repodata (slot));
}

/**
 * @brief C26: removing a slot while a sink is blocked pushing a second buffer
 *        wakes it with FALSE instead of leaving it stuck on freed memory.
 */
TEST (tensorRepo, removeWakesBlockedSink)
{
  const guint slot = 1004;
  GstCaps *caps;
  GstBuffer *buf1, *buf2;
  RepoJob *job;
  GThread *thread;

  gst_tensor_repo_init ();
  ASSERT_TRUE (gst_tensor_repo_add_repodata (slot, TRUE));

  caps = gst_caps_from_string (TEST_REPO_CAPS);
  ASSERT_NE (caps, nullptr);
  buf1 = gst_buffer_new_allocate (NULL, 4, NULL);
  buf2 = gst_buffer_new_allocate (NULL, 4, NULL);

  ASSERT_TRUE (gst_tensor_repo_set_buffer (slot, buf1, caps));

  job = repo_job_new (OP_SET_BUFFER, slot);
  job->in_buffer = buf2;
  job->in_caps = caps;

  thread = g_thread_new ("removeWakesBlockedSink", repo_job_thread, job);
  ASSERT_TRUE (wait_ctx_wait_started (&job->ctx, REPO_WAIT_TIMEOUT_US));
  g_usleep (200 * 1000);

  ASSERT_TRUE (gst_tensor_repo_remove_repodata (slot));

  if (!wait_ctx_join_or_timeout (&job->ctx, REPO_WAIT_TIMEOUT_US))
    FAIL () << "set_buffer() did not return after remove_repodata()";
  g_thread_join (thread);

  EXPECT_FALSE (job->set_ret);

  repo_job_free (job);
  gst_buffer_unref (buf1);
  gst_buffer_unref (buf2);
  gst_caps_unref (caps);
}

/**
 * @brief C26: removing a slot while a src is blocked pulling from an empty
 *        slot wakes it with NULL and eos set.
 */
TEST (tensorRepo, removeWakesBlockedSrc)
{
  const guint slot = 1005;
  RepoJob *job;
  GThread *thread;

  gst_tensor_repo_init ();
  ASSERT_TRUE (gst_tensor_repo_add_repodata (slot, FALSE));

  job = repo_job_new (OP_GET_BUFFER, slot);

  thread = g_thread_new ("removeWakesBlockedSrc", repo_job_thread, job);
  ASSERT_TRUE (wait_ctx_wait_started (&job->ctx, REPO_WAIT_TIMEOUT_US));
  g_usleep (200 * 1000);

  ASSERT_TRUE (gst_tensor_repo_remove_repodata (slot));

  if (!wait_ctx_join_or_timeout (&job->ctx, REPO_WAIT_TIMEOUT_US))
    FAIL () << "get_buffer() did not return after remove_repodata()";
  g_thread_join (thread);

  EXPECT_EQ (job->out_buffer, nullptr);
  EXPECT_TRUE (job->eos);

  repo_job_free (job);
}

/**
 * @brief C26: removing a full slot wakes every waiter (broadcast, not
 *        signal) blocked pushing into it.
 */
TEST (tensorRepo, removeWakesAllWaiters)
{
  const guint slot = 1006;
  GstCaps *caps;
  GstBuffer *buf0, *buf1, *buf2;
  RepoJob *job1, *job2;
  GThread *t1, *t2;

  gst_tensor_repo_init ();
  ASSERT_TRUE (gst_tensor_repo_add_repodata (slot, TRUE));

  caps = gst_caps_from_string (TEST_REPO_CAPS);
  ASSERT_NE (caps, nullptr);
  buf0 = gst_buffer_new_allocate (NULL, 4, NULL);
  buf1 = gst_buffer_new_allocate (NULL, 4, NULL);
  buf2 = gst_buffer_new_allocate (NULL, 4, NULL);

  ASSERT_TRUE (gst_tensor_repo_set_buffer (slot, buf0, caps));

  job1 = repo_job_new (OP_SET_BUFFER, slot);
  job1->in_buffer = buf1;
  job1->in_caps = caps;

  job2 = repo_job_new (OP_SET_BUFFER, slot);
  job2->in_buffer = buf2;
  job2->in_caps = caps;

  t1 = g_thread_new ("removeWakesAllWaiters1", repo_job_thread, job1);
  t2 = g_thread_new ("removeWakesAllWaiters2", repo_job_thread, job2);

  ASSERT_TRUE (wait_ctx_wait_started (&job1->ctx, REPO_WAIT_TIMEOUT_US));
  ASSERT_TRUE (wait_ctx_wait_started (&job2->ctx, REPO_WAIT_TIMEOUT_US));
  g_usleep (200 * 1000);

  ASSERT_TRUE (gst_tensor_repo_remove_repodata (slot));

  gboolean ok1 = wait_ctx_join_or_timeout (&job1->ctx, REPO_WAIT_TIMEOUT_US);
  gboolean ok2 = wait_ctx_join_or_timeout (&job2->ctx, REPO_WAIT_TIMEOUT_US);

  if (ok1)
    g_thread_join (t1);
  if (ok2)
    g_thread_join (t2);

  ASSERT_TRUE (ok1) << "first set_buffer() did not return after remove_repodata()";
  ASSERT_TRUE (ok2) << "second set_buffer() did not return after remove_repodata()";

  EXPECT_FALSE (job1->set_ret);
  EXPECT_FALSE (job2->set_ret);

  repo_job_free (job1);
  repo_job_free (job2);
  gst_buffer_unref (buf0);
  gst_buffer_unref (buf1);
  gst_buffer_unref (buf2);
  gst_caps_unref (caps);
}

/**
 * @brief Re-adding a slot after it was removed starts a fresh, non-eos slot.
 */
TEST (tensorRepo, readdAfterRemove)
{
  const guint slot = 1007;
  GstCaps *caps, *out_caps = NULL;
  GstBuffer *buf1, *buf2, *out_buf;
  gboolean eos = FALSE;
  guint newid = 0;

  gst_tensor_repo_init ();
  ASSERT_TRUE (gst_tensor_repo_add_repodata (slot, TRUE));

  caps = gst_caps_from_string (TEST_REPO_CAPS);
  ASSERT_NE (caps, nullptr);
  buf1 = gst_buffer_new_allocate (NULL, 4, NULL);
  ASSERT_TRUE (gst_tensor_repo_set_buffer (slot, buf1, caps));
  ASSERT_TRUE (gst_tensor_repo_remove_repodata (slot));

  ASSERT_TRUE (gst_tensor_repo_add_repodata (slot, TRUE));
  EXPECT_FALSE (gst_tensor_repo_check_eos (slot));

  buf2 = gst_buffer_new_allocate (NULL, 4, NULL);
  ASSERT_TRUE (gst_tensor_repo_set_buffer (slot, buf2, caps));

  out_buf = gst_tensor_repo_get_buffer (slot, &eos, &newid, &out_caps);
  ASSERT_NE (out_buf, nullptr);
  EXPECT_NE (out_buf, buf2);
  EXPECT_FALSE (eos);

  gst_buffer_unref (out_buf);
  gst_caps_unref (out_caps);
  gst_buffer_unref (buf1);
  gst_buffer_unref (buf2);
  gst_caps_unref (caps);
  EXPECT_TRUE (gst_tensor_repo_remove_repodata (slot));
}

/**
 * @brief C25: 8 threads racing to add_repodata() the same fresh slots must
 *        never fail (the old code could double-free and abort instead).
 */
TEST (tensorRepo, addSameSlotConcurrently)
{
  const guint base = 5000;
  const guint count = 2000;
  const guint num_threads = 8;
  StartBarrier barrier = {};
  gint failures = 0;
  AddJob jobs[num_threads];
  GThread *threads[num_threads];
  guint t, i;

  gst_tensor_repo_init ();
  g_mutex_init (&barrier.lock);
  g_cond_init (&barrier.cond);

  for (t = 0; t < num_threads; t++) {
    jobs[t].barrier = &barrier;
    jobs[t].base = base;
    jobs[t].count = count;
    jobs[t].is_sink = (t % 2) == 0;
    jobs[t].failures = &failures;
    threads[t]
        = g_thread_new ("addSameSlotConcurrently", add_concurrent_thread, &jobs[t]);
  }

  /* give every thread a chance to reach the barrier before releasing them */
  g_usleep (50 * 1000);
  start_barrier_release (&barrier);

  for (t = 0; t < num_threads; t++)
    g_thread_join (threads[t]);

  EXPECT_EQ (0, g_atomic_int_get (&failures));

  for (i = 0; i < count; i++) {
    EXPECT_TRUE (gst_tensor_repo_remove_repodata (base + i));
    EXPECT_FALSE (gst_tensor_repo_remove_repodata (base + i));
  }

  g_mutex_clear (&barrier.lock);
  g_cond_clear (&barrier.cond);
}

/**
 * @brief C26 through the elements: destroying a tensor_reposrc that nobody
 *        ever pulled from must error the blocked tensor_reposink instead of
 *        hanging it forever, and must not deadlock the following state change.
 */
TEST (tensorRepo, pipelineSinkErrorsWhenSlotRemoved)
{
  const guint slot = 1008;
  gchar *str_pipeline;
  GstElement *pipeline, *reposrc, *reposink;
  GstPad *sinkpad;
  GstBus *bus;
  GstMessage *msg;
  gulong probe_id;
  guint received = 0;
  guint missing_slot = 0;

  str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=10 ! video/x-raw,format=RGB,width=4,height=4,framerate=30/1 ! "
      "tensor_converter ! tensor_reposink name=rsink slot-index=%u",
      slot);
  pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);
  g_free (str_pipeline);

  reposrc = gst_element_factory_make ("tensor_reposrc", NULL);
  ASSERT_NE (reposrc, nullptr);
  gst_object_ref_sink (reposrc);
  g_object_set (reposrc, "slot-index", slot, NULL);

  reposink = gst_bin_get_by_name (GST_BIN (pipeline), "rsink");
  ASSERT_NE (reposink, nullptr);
  sinkpad = gst_element_get_static_pad (reposink, "sink");
  ASSERT_NE (sinkpad, nullptr);
  probe_id = gst_pad_add_probe (
      sinkpad, GST_PAD_PROBE_TYPE_BUFFER, count_buffer_probe, &received, NULL);

  bus = gst_element_get_bus (pipeline);
  g_log_set_default_handler (count_missing_slot_log, &missing_slot);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  /**
   * Nobody pulls from the slot, so the sink blocks in set_buffer() on the
   * second buffer. The probe runs before render(), hence the settle time.
   */
  if (wait_pipeline_process_buffers (&received, 2U, 5000U))
    g_usleep (200 * 1000);
  else
    ADD_FAILURE () << "the sink did not receive two buffers";

  gst_object_unref (reposrc);

  msg = gst_bus_timed_pop_filtered (bus, 5 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  /**
   * The probe and the log handler point into this frame, so they are dropped
   * here, before any check that could leave the case early.
   */
  g_log_set_default_handler (g_log_default_handler, NULL);
  gst_pad_remove_probe (sinkpad, probe_id);
  gst_object_unref (sinkpad);
  gst_object_unref (reposink);
  gst_object_unref (bus);

  EXPECT_EQ (0U, (guint) g_atomic_int_get (&missing_slot));
  EXPECT_NE (msg, nullptr);

  if (msg != NULL) {
    EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_ERROR);

    if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR) {
      GError *error = NULL;

      EXPECT_STREQ (GST_OBJECT_NAME (GST_MESSAGE_SRC (msg)), "rsink");
      gst_message_parse_error (msg, &error, NULL);
      EXPECT_NE (error, nullptr);

      if (error != NULL) {
        EXPECT_EQ (GST_RESOURCE_ERROR, error->domain);
        EXPECT_EQ ((gint) GST_RESOURCE_ERROR_WRITE, error->code);
      }
      g_clear_error (&error);
    }
    gst_message_unref (msg);
  }

  /* the render() call already returned an error, so this must not hang */
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (pipeline);
}

/**
 * @brief Removing a slot that was never added fails, both times.
 */
TEST (tensorRepo, removeUnknownSlot_n)
{
  const guint slot = 1100;

  gst_tensor_repo_init ();
  EXPECT_FALSE (gst_tensor_repo_remove_repodata (slot));
  EXPECT_FALSE (gst_tensor_repo_remove_repodata (slot));
}

/**
 * @brief set_buffer() on a slot that was never added fails.
 */
TEST (tensorRepo, setBufferUnknownSlot_n)
{
  const guint slot = 1101;
  GstCaps *caps;
  GstBuffer *buf;

  gst_tensor_repo_init ();
  caps = gst_caps_from_string (TEST_REPO_CAPS);
  ASSERT_NE (caps, nullptr);
  buf = gst_buffer_new_allocate (NULL, 4, NULL);

  EXPECT_FALSE (gst_tensor_repo_set_buffer (slot, buf, caps));

  gst_buffer_unref (buf);
  gst_caps_unref (caps);
}

/**
 * @brief get_buffer() on a slot that was never added returns NULL.
 */
TEST (tensorRepo, getBufferUnknownSlot_n)
{
  const guint slot = 1102;
  gboolean eos = FALSE;
  guint newid = 0;
  GstCaps *caps = NULL;

  gst_tensor_repo_init ();
  EXPECT_EQ (gst_tensor_repo_get_buffer (slot, &eos, &newid, &caps), nullptr);
}

/**
 * @brief set_eos() on a slot that was never added fails.
 */
TEST (tensorRepo, setEosUnknownSlot_n)
{
  const guint slot = 1103;

  gst_tensor_repo_init ();
  EXPECT_FALSE (gst_tensor_repo_set_eos (slot));
}

/**
 * @brief check_changed() on a slot that was never added fails.
 */
TEST (tensorRepo, checkChangedUnknownSlot_n)
{
  const guint slot = 1104;
  guint newid = 0;

  gst_tensor_repo_init ();
  EXPECT_FALSE (gst_tensor_repo_check_changed (slot, &newid, TRUE));
}

/**
 * @brief set_buffer() fails once the slot reached eos, and get_buffer() then
 *        reports eos without ever returning a buffer.
 */
TEST (tensorRepo, setBufferAfterEos_n)
{
  const guint slot = 1105;
  GstCaps *caps;
  GstBuffer *buf, *out_buf;
  gboolean eos = FALSE;
  guint newid = 0;
  GstCaps *out_caps = NULL;

  gst_tensor_repo_init ();
  ASSERT_TRUE (gst_tensor_repo_add_repodata (slot, TRUE));
  ASSERT_TRUE (gst_tensor_repo_set_eos (slot));

  caps = gst_caps_from_string (TEST_REPO_CAPS);
  ASSERT_NE (caps, nullptr);
  buf = gst_buffer_new_allocate (NULL, 4, NULL);

  EXPECT_FALSE (gst_tensor_repo_set_buffer (slot, buf, caps));

  out_buf = gst_tensor_repo_get_buffer (slot, &eos, &newid, &out_caps);
  EXPECT_EQ (out_buf, nullptr);
  EXPECT_TRUE (eos);

  gst_buffer_unref (buf);
  gst_caps_unref (caps);
  EXPECT_TRUE (gst_tensor_repo_remove_repodata (slot));
}

/**
 * @brief Number of tensors a stream carries to reach GstTensorsInfo::extra.
 */
#define EXTRA_NUM_TENSORS ((guint) (NNS_TENSOR_MEMORY_MAX + 4))

/**
 * @brief Query the caps of tensor_reposrc repeatedly with more tensors than
 *        NNS_TENSOR_MEMORY_MAX, which reparses the configuration every time.
 */
TEST (tensorRepoSrc, extraTensorsCapsQuery)
{
  GstElement *reposrc;
  GstPad *srcpad;
  GstCaps *caps;
  guint i;

  reposrc = gst_element_factory_make ("tensor_reposrc", NULL);
  ASSERT_NE (reposrc, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  g_object_set (reposrc, "caps", caps, NULL);
  gst_caps_unref (caps);

  srcpad = gst_element_get_static_pad (reposrc, "src");
  ASSERT_NE (srcpad, nullptr);

  for (i = 0; i < 3; i++) {
    GstStructure *structure;
    gint num_tensors = 0;

    caps = gst_pad_query_caps (srcpad, NULL);
    ASSERT_NE (caps, nullptr);
    ASSERT_EQ (gst_caps_get_size (caps), 1U);

    structure = gst_caps_get_structure (caps, 0);
    EXPECT_TRUE (gst_structure_get_int (structure, "num_tensors", &num_tensors));
    EXPECT_EQ (num_tensors, (gint) EXTRA_NUM_TENSORS);
    gst_caps_unref (caps);
  }

  EXPECT_EQ (GST_TENSOR_REPOSRC (reposrc)->config.info.num_tensors, EXTRA_NUM_TENSORS);
  EXPECT_NE (GST_TENSOR_REPOSRC (reposrc)->config.info.extra, nullptr);

  /* the filtered query intersects the filter with the caps of the element */
  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  gst_caps_take (&caps, gst_pad_query_caps (srcpad, caps));
  ASSERT_NE (caps, nullptr);
  EXPECT_TRUE (gst_caps_is_fixed (caps));
  EXPECT_EQ (GST_TENSOR_REPOSRC (reposrc)->config.info.num_tensors, EXTRA_NUM_TENSORS);
  gst_caps_unref (caps);

  gst_object_unref (srcpad);
  gst_object_unref (reposrc);
}

/**
 * @brief Query the caps of tensor_reposrc which has no caps of its own, where
 *        the filter of the query describes the stream.
 */
TEST (tensorRepoSrc, extraTensorsFilteredCapsQuery)
{
  GstElement *reposrc;
  GstPad *srcpad;
  GstCaps *caps;
  GstStructure *structure;
  gint num_tensors = 0;

  reposrc = gst_element_factory_make ("tensor_reposrc", NULL);
  ASSERT_NE (reposrc, nullptr);

  srcpad = gst_element_get_static_pad (reposrc, "src");
  ASSERT_NE (srcpad, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  gst_caps_take (&caps, gst_pad_query_caps (srcpad, caps));
  ASSERT_NE (caps, nullptr);
  ASSERT_EQ (gst_caps_get_size (caps), 1U);

  structure = gst_caps_get_structure (caps, 0);
  EXPECT_TRUE (gst_structure_get_int (structure, "num_tensors", &num_tensors));
  EXPECT_EQ (num_tensors, (gint) EXTRA_NUM_TENSORS);
  EXPECT_EQ (GST_TENSOR_REPOSRC (reposrc)->config.info.num_tensors, EXTRA_NUM_TENSORS);
  gst_caps_unref (caps);

  gst_object_unref (srcpad);
  gst_object_unref (reposrc);
}

/**
 * @brief Query the caps of tensor_reposrc which cannot describe a tensor stream.
 */
TEST (tensorRepoSrc, extraTensorsCapsQuery_n)
{
  GstElement *reposrc;
  GstPad *srcpad;
  GstCaps *caps;

  reposrc = gst_element_factory_make ("tensor_reposrc", NULL);
  ASSERT_NE (reposrc, nullptr);

  srcpad = gst_element_get_static_pad (reposrc, "src");
  ASSERT_NE (srcpad, nullptr);

  /* neither the element nor the query describes a single tensor stream */
  caps = gst_pad_query_caps (srcpad, NULL);
  ASSERT_NE (caps, nullptr);
  EXPECT_TRUE (gst_caps_is_any (caps));
  EXPECT_EQ (GST_TENSOR_REPOSRC (reposrc)->config.info.num_tensors, 0U);
  gst_caps_unref (caps);

  caps = gst_caps_from_string ("video/x-raw,format=RGB,width=4,height=4");
  g_object_set (reposrc, "caps", caps, NULL);
  gst_caps_unref (caps);

  caps = gst_pad_query_caps (srcpad, NULL);
  ASSERT_NE (caps, nullptr);
  EXPECT_TRUE (gst_caps_is_any (caps));
  EXPECT_EQ (GST_TENSOR_REPOSRC (reposrc)->config.info.num_tensors, 0U);
  gst_caps_unref (caps);

  gst_object_unref (srcpad);
  gst_object_unref (reposrc);
}

/**
 * @brief Number of tensors the buffer tensor_reposrc generated carried.
 */
static guint dummy_num_tensors = 0;

/**
 * @brief Record the number of tensors of the buffer tensor_sink received.
 */
static void
record_num_tensors (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  (void) element;
  (void) user_data;

  dummy_num_tensors = gst_tensor_buffer_get_count (buffer);
}

/**
 * @brief Let tensor_reposrc generate the buffer it starts a stream of more
 *        tensors than NNS_TENSOR_MEMORY_MAX with.
 */
TEST (tensorRepoSrc, extraTensorsDummyBuffer)
{
  const guint slot = 1100;
  guint received = 0;
  GstElement *pipeline, *reposrc, *sink;
  GstCaps *caps;

  dummy_num_tensors = 0;
  gst_tensor_repo_init ();

  pipeline = gst_parse_launch ("tensor_reposrc name=srcx ! tensor_sink name=sinkx", NULL);
  ASSERT_NE (pipeline, nullptr);

  reposrc = gst_bin_get_by_name (GST_BIN (pipeline), "srcx");
  ASSERT_NE (reposrc, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  g_object_set (reposrc, "caps", caps, "slot-index", slot, NULL);
  gst_caps_unref (caps);
  gst_object_unref (reposrc);

  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink, nullptr);
  g_signal_connect (sink, "new-data", G_CALLBACK (record_num_tensors), NULL);
  g_signal_connect (sink, "new-data", G_CALLBACK (count_output), &received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_pipeline_process_buffers (&received, 1U, TEST_TIMEOUT_LIMIT_MS));
  EXPECT_EQ (dummy_num_tensors, EXTRA_NUM_TENSORS);

  /* the element waits for a buffer of the slot, let it reach the end instead */
  gst_tensor_repo_set_eos (slot);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (sink);
  gst_object_unref (pipeline);
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
