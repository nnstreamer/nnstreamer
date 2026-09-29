/**
 * @file        unittest_subplugin.cc
 * @date        17 Sep 2026
 * @brief       Unit tests for the sub-plugin registry
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>
#include <dlfcn.h>
#include <glib.h>
#include <nnstreamer_subplugin.h>
#include <stdarg.h>
#include <string.h>

#define RACE_THREADS (16U)
#define RACE_ROUNDS (500U)
#define UNLOCK_WAIT_US (100 * G_TIME_SPAN_MILLISECOND)

/**
 * @brief A registry entry to unregister from inside an interposed GLib call.
 */
typedef struct {
  GThread *owner; /**< the thread whose call triggers the unregister */
  subpluginType type; /**< type of the entry */
  const char *name; /**< name of the entry */
  gboolean triggered; /**< TRUE once the interposed call has fired */
  gboolean unregistered; /**< result of unregister_subplugin () */
  gboolean finished; /**< TRUE when unregister_subplugin () returned */
  gboolean finished_in_call; /**< finished, sampled inside the interposed call */
  GMutex lock; /**< protects finished and finished_in_call */
  GCond cond; /**< signals finished */
} interpose_s;

static interpose_s *join_hook = NULL;
static interpose_s *clear_hook = NULL;

/**
 * @brief Take the hook if the calling thread armed it.
 */
static interpose_s *
_take_hook (interpose_s **hook)
{
  interpose_s *h = *hook;

  if (h == NULL || h->owner != g_thread_self ())
    return NULL;

  *hook = NULL;
  h->triggered = TRUE;
  return h;
}

static gsize real_strjoinv = 0;
static gsize real_datalist_clear = 0;

/**
 * @brief Resolve the GLib function an interposer wraps, once.
 */
static gpointer
_real_symbol (gsize *cache, const char *symbol)
{
  if (g_once_init_enter (cache)) {
    gpointer real = dlsym (RTLD_NEXT, symbol);

    g_assert (real != NULL);
    g_once_init_leave (cache, (gsize) real);
  }

  return (gpointer) *cache;
}

/**
 * @brief Interposes g_strjoinv (): drops the armed entry before the join,
 *        as a concurrent unregister_subplugin () would.
 */
extern "C" gchar *
g_strjoinv (const gchar *separator, gchar **str_array)
{
  typedef gchar *(*strjoinv_f) (const gchar *, gchar **);
  strjoinv_f real_join = (strjoinv_f) _real_symbol (&real_strjoinv, "g_strjoinv");
  interpose_s *h = _take_hook (&join_hook);

  if (h != NULL)
    h->unregistered = unregister_subplugin (h->type, h->name);

  return real_join (separator, str_array);
}

/**
 * @brief Thread body: unregister the entry of the hook.
 */
static gpointer
_unregister_worker (gpointer user_data)
{
  interpose_s *h = (interpose_s *) user_data;
  gboolean ret = unregister_subplugin (h->type, h->name);

  g_mutex_lock (&h->lock);
  h->unregistered = ret;
  h->finished = TRUE;
  g_cond_signal (&h->cond);
  g_mutex_unlock (&h->lock);

  return NULL;
}

/**
 * @brief Interposes g_datalist_clear (): starts a concurrent
 *        unregister_subplugin () and gives it time to run before the clear.
 */
extern "C" void
g_datalist_clear (GData **datalist)
{
  typedef void (*datalist_clear_f) (GData **);
  datalist_clear_f real_clear
      = (datalist_clear_f) _real_symbol (&real_datalist_clear, "g_datalist_clear");
  interpose_s *h = _take_hook (&clear_hook);

  if (h != NULL) {
    gint64 end = g_get_monotonic_time () + UNLOCK_WAIT_US;
    GThread *t = g_thread_new ("sp-unregister", _unregister_worker, h);

    g_thread_unref (t);
    g_mutex_lock (&h->lock);
    while (!h->finished && g_cond_wait_until (&h->cond, &h->lock, end))
      ;
    h->finished_in_call = h->finished;
    g_mutex_unlock (&h->lock);

    /* The entry holding *datalist is freed if the unregister got through. */
    if (h->finished_in_call)
      return;
  }

  real_clear (datalist);
}

/**
 * @brief Calls subplugin_set_custom_property_desc () with variable arguments.
 */
static void
_set_desc (subpluginType type, const char *name, const gchar *prop, ...)
{
  va_list varargs;

  va_start (varargs, prop);
  subplugin_set_custom_property_desc (type, name, prop, varargs);
  va_end (varargs);
}

/**
 * @brief State shared by the threads of the concurrent registration test.
 */
typedef struct {
  GMutex lock; /**< protects every field below */
  GCond cond; /**< signals a new round or a finished registration */
  const char *name; /**< the name every thread registers */
  guint round; /**< current round, 0 before the first one */
  gboolean quit; /**< TRUE when the threads should exit */
  guint done; /**< threads that finished the current round */
  guint registered; /**< threads that registered the name in this round */
  const void *winner; /**< data of the thread that registered the name */
  gint slots[RACE_THREADS]; /**< distinct data pointer for each thread */
} race_state_s;

/**
 * @brief Argument of a registering thread.
 */
typedef struct {
  race_state_s *state; /**< the shared state */
  guint idx; /**< index of the thread's data slot */
} race_arg_s;

/**
 * @brief Thread body: register the name once per round, all threads at once.
 */
static gpointer
_register_worker (gpointer user_data)
{
  race_arg_s *arg = (race_arg_s *) user_data;
  race_state_s *state = arg->state;
  const void *data = &state->slots[arg->idx];
  guint last = 0;
  gboolean ret;

  g_mutex_lock (&state->lock);
  while (TRUE) {
    while (!state->quit && state->round == last)
      g_cond_wait (&state->cond, &state->lock);
    if (state->quit)
      break;
    last = state->round;
    g_mutex_unlock (&state->lock);

    ret = register_subplugin (NNS_IF_CUSTOM, state->name, data);

    g_mutex_lock (&state->lock);
    if (ret) {
      state->registered++;
      state->winner = data;
    }
    state->done++;
    g_cond_broadcast (&state->cond);
  }
  g_mutex_unlock (&state->lock);

  return NULL;
}

/**
 * @brief Log handler that drops the messages of the registration race.
 */
static void
_drop_log (const gchar *log_domain, GLogLevelFlags log_level,
    const gchar *message, gpointer user_data)
{
}

/**
 * @brief Register one name from many threads at once; exactly one must win
 *        and the registry must keep the winner's data.
 */
TEST (nnstreamerSubplugin, registerSameNameConcurrently)
{
  const char *name = "unittest_subplugin_race";
  race_state_s state;
  race_arg_s args[RACE_THREADS];
  GThread *threads[RACE_THREADS];
  guint r, i, failed_rounds = 0;
  guint handler;

  memset (&state, 0, sizeof (state));
  g_mutex_init (&state.lock);
  g_cond_init (&state.cond);
  state.name = name;
  handler = g_log_set_handler (NULL, G_LOG_LEVEL_WARNING, _drop_log, NULL);

  for (i = 0; i < RACE_THREADS; i++) {
    args[i].state = &state;
    args[i].idx = i;
    threads[i] = g_thread_new ("sp-register", _register_worker, &args[i]);
  }

  for (r = 1; r <= RACE_ROUNDS; r++) {
    g_mutex_lock (&state.lock);
    state.done = 0;
    state.registered = 0;
    state.winner = NULL;
    state.round = r;
    g_cond_broadcast (&state.cond);
    while (state.done < RACE_THREADS)
      g_cond_wait (&state.cond, &state.lock);
    g_mutex_unlock (&state.lock);

    if (state.registered != 1U || get_subplugin (NNS_IF_CUSTOM, name) != state.winner)
      failed_rounds++;

    EXPECT_TRUE (unregister_subplugin (NNS_IF_CUSTOM, name));
  }

  g_mutex_lock (&state.lock);
  state.quit = TRUE;
  g_cond_broadcast (&state.cond);
  g_mutex_unlock (&state.lock);

  for (i = 0; i < RACE_THREADS; i++)
    g_thread_join (threads[i]);

  g_log_remove_handler (NULL, handler);

  EXPECT_EQ (failed_rounds, 0U);
  EXPECT_EQ (get_subplugin (NNS_IF_CUSTOM, name), nullptr);

  g_cond_clear (&state.cond);
  g_mutex_clear (&state.lock);
}

/**
 * @brief A registered name can be looked up, listed, released and registered again.
 */
TEST (nnstreamerSubplugin, registerUnregisterCycle)
{
  const char *name = "unittest_subplugin_cycle";
  gint first = 1, second = 2;
  gchar **names;

  EXPECT_TRUE (register_subplugin (NNS_CUSTOM_DECODER, name, &first));
  EXPECT_EQ (get_subplugin (NNS_CUSTOM_DECODER, name), &first);

  names = get_all_subplugins (NNS_CUSTOM_DECODER);
  ASSERT_NE (names, nullptr);
  EXPECT_TRUE (g_strv_contains ((const gchar *const *) names, name));
  g_strfreev (names);

  EXPECT_TRUE (unregister_subplugin (NNS_CUSTOM_DECODER, name));
  EXPECT_EQ (get_subplugin (NNS_CUSTOM_DECODER, name), nullptr);

  EXPECT_TRUE (register_subplugin (NNS_CUSTOM_DECODER, name, &second));
  EXPECT_EQ (get_subplugin (NNS_CUSTOM_DECODER, name), &second);
  EXPECT_TRUE (unregister_subplugin (NNS_CUSTOM_DECODER, name));
}

/**
 * @brief Registering a name twice is refused and keeps the first data.
 */
TEST (nnstreamerSubplugin, registerDuplicate_n)
{
  const char *name = "unittest_subplugin_dup";
  gint first = 1, second = 2;

  ASSERT_TRUE (register_subplugin (NNS_CUSTOM_CONVERTER, name, &first));
  EXPECT_FALSE (register_subplugin (NNS_CUSTOM_CONVERTER, name, &second));
  EXPECT_FALSE (register_subplugin (NNS_CUSTOM_CONVERTER, name, &first));
  EXPECT_EQ (get_subplugin (NNS_CUSTOM_CONVERTER, name), &first);

  EXPECT_TRUE (unregister_subplugin (NNS_CUSTOM_CONVERTER, name));
  EXPECT_EQ (get_subplugin (NNS_CUSTOM_CONVERTER, name), nullptr);
}

/**
 * @brief The same name may exist once per sub-plugin type.
 */
TEST (nnstreamerSubplugin, registerSameNameOtherType)
{
  const char *name = "unittest_subplugin_types";
  gint first = 1, second = 2;

  EXPECT_TRUE (register_subplugin (NNS_CUSTOM_DECODER, name, &first));
  EXPECT_TRUE (register_subplugin (NNS_IF_CUSTOM, name, &second));
  EXPECT_EQ (get_subplugin (NNS_CUSTOM_DECODER, name), &first);
  EXPECT_EQ (get_subplugin (NNS_IF_CUSTOM, name), &second);

  EXPECT_TRUE (unregister_subplugin (NNS_CUSTOM_DECODER, name));
  EXPECT_TRUE (unregister_subplugin (NNS_IF_CUSTOM, name));
}

/**
 * @brief Reserved names, an unknown type and NULL arguments are refused.
 */
TEST (nnstreamerSubplugin, registerInvalidArgs_n)
{
  gint data = 1;

  EXPECT_FALSE (register_subplugin (NNS_IF_CUSTOM, "any", &data));
  EXPECT_FALSE (register_subplugin (NNS_IF_CUSTOM, "AUTO", &data));
  EXPECT_FALSE (register_subplugin (NNS_SUBPLUGIN_END, "unittest_subplugin_type", &data));
  EXPECT_FALSE (register_subplugin (NNS_IF_CUSTOM, NULL, &data));
  EXPECT_FALSE (register_subplugin (NNS_IF_CUSTOM, "unittest_subplugin_null", NULL));
  EXPECT_EQ (get_subplugin (NNS_IF_CUSTOM, "any"), nullptr);
  EXPECT_EQ (get_subplugin (NNS_IF_CUSTOM, "unittest_subplugin_null"), nullptr);
}

/**
 * @brief Unregistering a name that is not registered fails.
 */
TEST (nnstreamerSubplugin, unregisterUnknown_n)
{
  gint data = 1;

  ASSERT_TRUE (register_subplugin (NNS_CUSTOM_DECODER, "unittest_subplugin_once", &data));
  EXPECT_FALSE (unregister_subplugin (NNS_CUSTOM_DECODER, "unittest_subplugin_none"));
  EXPECT_TRUE (unregister_subplugin (NNS_CUSTOM_DECODER, "unittest_subplugin_once"));
  EXPECT_FALSE (unregister_subplugin (NNS_CUSTOM_DECODER, "unittest_subplugin_once"));
}

/**
 * @brief get_all_subplugins () keeps the names it listed when an entry is
 *        unregistered before they are joined.
 */
TEST (nnstreamerSubplugin, listNamesUnregisteredWhileJoining)
{
  const char *name = "unittest_subplugin_list_unregistered";
  const char *other = "unittest_subplugin_list_kept";
  interpose_s hook;
  gint data = 1;
  gchar **names;

  memset (&hook, 0, sizeof (hook));
  hook.owner = g_thread_self ();
  hook.type = NNS_CUSTOM_CONVERTER;
  hook.name = name;

  ASSERT_TRUE (register_subplugin (NNS_CUSTOM_CONVERTER, name, &data));
  ASSERT_TRUE (register_subplugin (NNS_CUSTOM_CONVERTER, other, &data));

  join_hook = &hook;
  names = get_all_subplugins (NNS_CUSTOM_CONVERTER);
  join_hook = NULL;

  EXPECT_TRUE (hook.triggered);
  EXPECT_TRUE (hook.unregistered);
  ASSERT_NE (names, nullptr);
  EXPECT_TRUE (g_strv_contains ((const gchar *const *) names, name));
  EXPECT_TRUE (g_strv_contains ((const gchar *const *) names, other));
  g_strfreev (names);

  names = get_all_subplugins (NNS_CUSTOM_CONVERTER);
  ASSERT_NE (names, nullptr);
  EXPECT_FALSE (g_strv_contains ((const gchar *const *) names, name));
  EXPECT_TRUE (g_strv_contains ((const gchar *const *) names, other));
  g_strfreev (names);

  EXPECT_EQ (get_subplugin (NNS_CUSTOM_CONVERTER, name), nullptr);
  EXPECT_TRUE (unregister_subplugin (NNS_CUSTOM_CONVERTER, other));
}

/**
 * @brief A concurrent unregister_subplugin () waits until
 *        subplugin_set_custom_property_desc () is done with the entry.
 */
TEST (nnstreamerSubplugin, setDescBlocksUnregister)
{
  const char *name = "unittest_subplugin_desc_unregister";
  interpose_s hook;
  gint data = 1;

  memset (&hook, 0, sizeof (hook));
  g_mutex_init (&hook.lock);
  g_cond_init (&hook.cond);
  hook.owner = g_thread_self ();
  hook.type = NNS_IF_CUSTOM;
  hook.name = name;

  ASSERT_TRUE (register_subplugin (NNS_IF_CUSTOM, name, &data));

  clear_hook = &hook;
  _set_desc (NNS_IF_CUSTOM, name, NULL);
  clear_hook = NULL;

  g_mutex_lock (&hook.lock);
  while (hook.triggered && !hook.finished)
    g_cond_wait (&hook.cond, &hook.lock);
  g_mutex_unlock (&hook.lock);

  EXPECT_TRUE (hook.triggered);
  EXPECT_FALSE (hook.finished_in_call);
  EXPECT_TRUE (hook.unregistered);
  EXPECT_EQ (get_subplugin (NNS_IF_CUSTOM, name), nullptr);
  EXPECT_EQ (subplugin_get_custom_property_desc (NNS_IF_CUSTOM, name), nullptr);

  g_cond_clear (&hook.cond);
  g_mutex_clear (&hook.lock);
}

/**
 * @brief Custom property descriptions are stored, replaced and dropped with
 *        the entry.
 */
TEST (nnstreamerSubplugin, customPropertyDesc)
{
  const char *name = "unittest_subplugin_desc";
  gint data = 1;
  GData *dlist;

  ASSERT_TRUE (register_subplugin (NNS_CUSTOM_DECODER, name, &data));
  EXPECT_EQ (subplugin_get_custom_property_desc (NNS_CUSTOM_DECODER, name), nullptr);

  _set_desc (NNS_CUSTOM_DECODER, name, "alpha", "first", "beta", "second", NULL);
  dlist = subplugin_get_custom_property_desc (NNS_CUSTOM_DECODER, name);
  EXPECT_STREQ ((const gchar *) g_datalist_get_data (&dlist, "alpha"), "first");
  EXPECT_STREQ ((const gchar *) g_datalist_get_data (&dlist, "beta"), "second");

  _set_desc (NNS_CUSTOM_DECODER, name, "gamma", "third", NULL);
  dlist = subplugin_get_custom_property_desc (NNS_CUSTOM_DECODER, name);
  EXPECT_EQ (g_datalist_get_data (&dlist, "alpha"), nullptr);
  EXPECT_STREQ ((const gchar *) g_datalist_get_data (&dlist, "gamma"), "third");

  EXPECT_TRUE (unregister_subplugin (NNS_CUSTOM_DECODER, name));
  EXPECT_EQ (subplugin_get_custom_property_desc (NNS_CUSTOM_DECODER, name), nullptr);
}

/**
 * @brief A property without a description stops the list and keeps the
 *        ones before it; an unknown name stores nothing.
 */
TEST (nnstreamerSubplugin, customPropertyDescInvalid_n)
{
  const char *name = "unittest_subplugin_desc_invalid";
  gint data = 1;
  GData *dlist;
  guint handler;

  handler = g_log_set_handler (NULL,
      (GLogLevelFlags) (G_LOG_LEVEL_WARNING | G_LOG_LEVEL_CRITICAL), _drop_log, NULL);

  ASSERT_TRUE (register_subplugin (NNS_CUSTOM_DECODER, name, &data));
  _set_desc (NNS_CUSTOM_DECODER, name, "alpha", "first", "beta", NULL);
  dlist = subplugin_get_custom_property_desc (NNS_CUSTOM_DECODER, name);
  EXPECT_STREQ ((const gchar *) g_datalist_get_data (&dlist, "alpha"), "first");
  EXPECT_EQ (g_datalist_get_data (&dlist, "beta"), nullptr);

  _set_desc (NNS_CUSTOM_DECODER, "unittest_subplugin_desc_none", "alpha", "first", NULL);
  EXPECT_EQ (subplugin_get_custom_property_desc (NNS_CUSTOM_DECODER, "unittest_subplugin_desc_none"),
      nullptr);

  EXPECT_TRUE (unregister_subplugin (NNS_CUSTOM_DECODER, name));
  g_log_remove_handler (NULL, handler);
}

/**
 * @brief State shared by the lookup-versus-unregister stress test.
 */
typedef struct {
  const char *name; /**< the name the writer registers and unregisters */
  const void *data; /**< the data registered under name */
  gint quit; /**< nonzero when the readers should exit */
  gint started; /**< readers that finished their first lookup round */
  gint bad; /**< lookups that returned something other than data or NULL */
} lookup_state_s;

/**
 * @brief Thread body: look the name up until told to stop.
 */
static gpointer
_lookup_worker (gpointer user_data)
{
  lookup_state_s *state = (lookup_state_s *) user_data;
  const void *found;
  gchar **names;
  gboolean started = FALSE;

  while (!g_atomic_int_get (&state->quit)) {
    found = get_subplugin (NNS_CUSTOM_DECODER, state->name);
    if (found != NULL && found != state->data)
      g_atomic_int_inc (&state->bad);

    subplugin_get_custom_property_desc (NNS_CUSTOM_DECODER, state->name);

    names = get_all_subplugins (NNS_CUSTOM_DECODER);
    g_strfreev (names);

    if (!started) {
      started = TRUE;
      g_atomic_int_inc (&state->started);
    }

    /* let the writer run under valgrind's serialized thread scheduling */
    g_usleep (10);
  }

  return NULL;
}

/**
 * @brief Lookups racing register/unregister of the same name only ever see
 *        the registered data or nothing.
 */
TEST (nnstreamerSubplugin, lookupWhileUnregistering)
{
  lookup_state_s state;
  GThread *threads[4];
  gint data = 1;
  guint r, i;
  guint handler;

  memset (&state, 0, sizeof (state));
  state.name = "unittest_subplugin_lookup_race";
  state.data = &data;

  /* load the configuration before the readers race on it (#4960 B4) */
  handler = g_log_set_handler (NULL, G_LOG_LEVEL_CRITICAL, _drop_log, NULL);
  g_strfreev (get_all_subplugins (NNS_CUSTOM_DECODER));

  for (i = 0; i < G_N_ELEMENTS (threads); i++)
    threads[i] = g_thread_new ("sp-lookup", _lookup_worker, &state);
  while (g_atomic_int_get (&state.started) < (gint) G_N_ELEMENTS (threads))
    g_thread_yield ();

  for (r = 0; r < RACE_ROUNDS; r++) {
    EXPECT_TRUE (register_subplugin (NNS_CUSTOM_DECODER, state.name, &data));
    _set_desc (NNS_CUSTOM_DECODER, state.name, "alpha", "first", NULL);
    EXPECT_TRUE (unregister_subplugin (NNS_CUSTOM_DECODER, state.name));
  }

  g_atomic_int_set (&state.quit, 1);
  for (i = 0; i < G_N_ELEMENTS (threads); i++)
    g_thread_join (threads[i]);
  g_log_remove_handler (NULL, handler);

  EXPECT_EQ (g_atomic_int_get (&state.bad), 0);
}

/**
 * @brief Main gtest
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

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
