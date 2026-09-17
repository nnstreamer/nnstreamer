/**
 * @file        unittest_subplugin.cc
 * @date        17 Sep 2026
 * @brief       Unit tests for the sub-plugin registry
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <nnstreamer_subplugin.h>
#include <string.h>

#define RACE_THREADS (16U)
#define RACE_ROUNDS (500U)

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
