/**
 * @file        unittest_conf.cc
 * @date        30 Sep 2026
 * @brief       Unit tests for concurrent use of the configuration manager
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <dlfcn.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <mutex>
#include <nnstreamer_conf.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_subplugin.h>
#include <string.h>
#include <string>
#include <thread>
#include <vector>

#define STRESS_THREADS (8U)
#define STRESS_KEYS (64U)

/**
 * The configuration globals are static, so the tests observe them through the
 * GLib calls libnnstreamer makes while it touches them. On ELF the executable's
 * definitions below take precedence over libglib's. A thread that armed a hook
 * waits inside the hooked call for a second armed thread: two threads inside at
 * once means the two calls are not serialized. A test that needs a hook skips
 * itself when the hook was bypassed.
 */
#define HOOK_KEYFILE (1U << 0)
#define HOOK_LOOKUP (1U << 1)
#define HOOK_INSERT (1U << 2)
#define HOOK_LOOKUP_HOLD (1U << 3)
#define HOOK_STRFREEV_HOLD (1U << 4)
static thread_local unsigned int hook_mask = 0;
static std::mutex region_lock;
static std::condition_variable region_cond;
static int region_calls = 0;
static int region_inside = 0;
static int region_max = 0;

/**
 * A thread that armed a *_HOLD hook stops inside the hooked call at its gate
 * until the test opens the gate (or 5 s pass).
 */
#define GATE_LOOKUP (0)
#define GATE_STRFREEV (1)
static bool gate_arrived[2];
static bool gate_open[2];

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
 * @brief Stay in the region until a second armed thread joins or 500 ms pass.
 */
static void
_region_enter (void)
{
  std::unique_lock<std::mutex> lk (region_lock);

  region_calls++;
  region_inside++;
  region_max = std::max (region_max, region_inside);
  region_cond.notify_all ();
  region_cond.wait_for (
      lk, std::chrono::milliseconds (500), [] { return region_inside >= 2; });
  region_inside--;
}

/**
 * @brief Reset the region counters.
 */
static void
_region_reset (void)
{
  std::lock_guard<std::mutex> lk (region_lock);

  region_calls = region_inside = region_max = 0;
}

/**
 * @brief Wait until an armed thread entered the region.
 */
static bool
_region_wait_entered (void)
{
  std::unique_lock<std::mutex> lk (region_lock);

  return region_cond.wait_for (
      lk, std::chrono::seconds (10), [] { return region_calls >= 1; });
}

/**
 * @brief Reset both gates.
 */
static void
_gate_reset (void)
{
  std::lock_guard<std::mutex> lk (region_lock);

  gate_arrived[GATE_LOOKUP] = gate_arrived[GATE_STRFREEV] = false;
  gate_open[GATE_LOOKUP] = gate_open[GATE_STRFREEV] = false;
}

/**
 * @brief Stop at a gate until it is opened.
 */
static void
_gate_hold (int gate)
{
  std::unique_lock<std::mutex> lk (region_lock);

  gate_arrived[gate] = true;
  region_cond.notify_all ();
  region_cond.wait_for (
      lk, std::chrono::seconds (5), [gate] { return gate_open[gate]; });
}

/**
 * @brief Wait until a thread stopped at a gate.
 */
static bool
_gate_wait_arrived (int gate)
{
  std::unique_lock<std::mutex> lk (region_lock);

  return region_cond.wait_for (
      lk, std::chrono::seconds (10), [gate] { return gate_arrived[gate]; });
}

/**
 * @brief Open a gate.
 */
static void
_gate_open (int gate)
{
  std::lock_guard<std::mutex> lk (region_lock);

  gate_open[gate] = true;
  region_cond.notify_all ();
}

/**
 * @brief Interposed g_key_file_load_from_file(), the ini read of nnsconf_loadconf().
 */
extern "C" gboolean
g_key_file_load_from_file (
    GKeyFile *key_file, const gchar *file, GKeyFileFlags flags, GError **error)
{
  static auto real = (gboolean (*) (GKeyFile *, const gchar *, GKeyFileFlags,
      GError **)) _hook_next ("g_key_file_load_from_file");

  if (hook_mask & HOOK_KEYFILE)
    _region_enter ();
  return real (key_file, file, flags, error);
}

/**
 * @brief Interposed g_hash_table_lookup(), the custom value cache lookup.
 */
extern "C" gpointer
g_hash_table_lookup (GHashTable *hash_table, gconstpointer key)
{
  static auto real = (gpointer (*) (GHashTable *, gconstpointer)) _hook_next (
      "g_hash_table_lookup");

  if (hook_mask & HOOK_LOOKUP)
    _region_enter ();
  if (hook_mask & HOOK_LOOKUP_HOLD)
    _gate_hold (GATE_LOOKUP);
  return real (hash_table, key);
}

/**
 * @brief Interposed g_hash_table_insert(), the custom value cache insertion.
 */
extern "C" gboolean
g_hash_table_insert (GHashTable *hash_table, gpointer key, gpointer value)
{
  static auto real = (gboolean (*) (GHashTable *, gpointer, gpointer)) _hook_next (
      "g_hash_table_insert");

  if (hook_mask & HOOK_INSERT)
    _region_enter ();
  return real (hash_table, key, value);
}

/**
 * @brief Interposed g_strfreev(), first called by a forced reload after it
 *        has freed the path of the ini file.
 */
extern "C" void
g_strfreev (gchar **str_array)
{
  static auto real = (void (*) (gchar **)) _hook_next ("g_strfreev");

  if (hook_mask & HOOK_STRFREEV_HOLD)
    _gate_hold (GATE_STRFREEV);
  real (str_array);
}

/**
 * @brief Fixture providing an ini file with a filter directory and custom values.
 */
class NNSConfConcurrency : public ::testing::Test
{
  protected:
  gchar *dir = NULL;
  gchar *filterdir = NULL;
  gchar *filter = NULL;
  gchar *decoderdir = NULL;
  gchar *decoder = NULL;
  gchar *ini = NULL;
  gchar *saved_env = NULL;
  gchar *reload_group = NULL;

  /**
   * @brief Write the ini, point NNSTREAMER_CONF at it and reload.
   */
  void SetUp () override
  {
    static guint setup_serial = 0;
    GString *content = g_string_new (NULL);
    guint i;

    reload_group = g_strdup_printf ("b4reload%u", setup_serial++);
    dir = g_dir_make_tmp ("nns-conf-XXXXXX", NULL);
    ASSERT_TRUE (dir != NULL);
    filterdir = g_build_filename (dir, "filters", NULL);
    filter = g_build_filename (filterdir,
        "libnnstreamer_filter_b4dummy" NNSTREAMER_SO_FILE_EXTENSION, NULL);
    ini = g_build_filename (dir, "nnstreamer.ini", NULL);
    ASSERT_EQ (g_mkdir (filterdir, 0755), 0);
    ASSERT_TRUE (g_file_set_contents (filter, "", 0, NULL));
    decoderdir = g_build_filename (dir, "decoders", NULL);
    decoder = g_build_filename (decoderdir,
        "libnnstreamer_decoder_b4reg" NNSTREAMER_SO_FILE_EXTENSION, NULL);
    ASSERT_EQ (g_mkdir (decoderdir, 0755), 0);
    ASSERT_TRUE (g_file_set_contents (decoder, "", 0, NULL));

    g_string_append_printf (content, "[common]\nenable_envvar=True\n");
    g_string_append_printf (content, "[filter]\nfilters=%s\n", filterdir);
    g_string_append_printf (content, "[decoder]\ndecoders=%s\n", decoderdir);
    g_string_append_printf (content, "[b4]\nkey=value\n[b4stress]\n");
    for (i = 0; i < STRESS_KEYS; i++)
      g_string_append_printf (content, "k%u=v%u\n", i, i);
    g_string_append_printf (content, "[%s]\nkey=reloaded\n", reload_group);
    ASSERT_TRUE (g_file_set_contents (ini, content->str, -1, NULL));
    g_string_free (content, TRUE);

    saved_env = g_strdup (g_getenv ("NNSTREAMER_CONF"));
    g_setenv ("NNSTREAMER_CONF", ini, TRUE);
    ASSERT_TRUE (nnsconf_loadconf (TRUE));

    if (g_strcmp0 (nnsconf_get_fullpath ("b4dummy", NNSCONF_PATH_FILTERS), filter) != 0)
      GTEST_SKIP () << "The ini given by NNSTREAMER_CONF is not used.";
  }

  /**
   * @brief Restore NNSTREAMER_CONF, reload and remove the files.
   */
  void TearDown () override
  {
    hook_mask = 0;
    if (saved_env)
      g_setenv ("NNSTREAMER_CONF", saved_env, TRUE);
    else
      g_unsetenv ("NNSTREAMER_CONF");
    nnsconf_loadconf (TRUE);

    if (ini)
      g_remove (ini);
    if (filter)
      g_remove (filter);
    if (filterdir)
      g_rmdir (filterdir);
    if (decoder)
      g_remove (decoder);
    if (decoderdir)
      g_rmdir (decoderdir);
    if (dir)
      g_rmdir (dir);
    g_free (saved_env);
    g_free (reload_group);
    g_free (ini);
    g_free (filter);
    g_free (filterdir);
    g_free (decoder);
    g_free (decoderdir);
    g_free (dir);
  }
};

/**
 * @brief A caller arriving during a load waits for it instead of loading again.
 */
TEST_F (NNSConfConcurrency, loadWaitsForLoad)
{
  const gchar *path = NULL;

  _region_reset ();

  std::thread loader ([] {
    hook_mask = HOOK_KEYFILE;
    EXPECT_TRUE (nnsconf_loadconf (TRUE));
  });

  if (!_region_wait_entered ()) {
    loader.join ();
    GTEST_SKIP () << "g_key_file_load_from_file() is not interposed.";
  }

  std::thread reader ([&path] {
    hook_mask = HOOK_KEYFILE;
    path = nnsconf_get_fullpath ("b4dummy", NNSCONF_PATH_FILTERS);
  });

  loader.join ();
  reader.join ();

  EXPECT_EQ (region_max, 1);
  EXPECT_EQ (region_calls, 1);
  EXPECT_STREQ (path, filter);
}

/**
 * @brief A lookup does not access the cache while a new value is inserted.
 */
TEST_F (NNSConfConcurrency, customInsertExcludesLookup)
{
  static guint serial = 0;
  gchar *key = g_strdup_printf ("envkey%u", serial++);
  gchar *envname = g_strdup_printf ("NNSTREAMER_b4_%s", key);
  gchar *inserted = NULL;
  gchar *looked_up = NULL;

  gchar *primed = nnsconf_get_custom_value_string ("b4", "key");
  ASSERT_STREQ (primed, "value");
  g_free (primed);
  g_setenv (envname, "envvalue", TRUE);

  _region_reset ();
  std::thread writer ([&inserted, key] {
    hook_mask = HOOK_INSERT;
    inserted = nnsconf_get_custom_value_string ("b4", key);
  });

  if (!_region_wait_entered ()) {
    writer.join ();
    g_unsetenv (envname);
    g_free (inserted);
    g_free (envname);
    g_free (key);
    GTEST_SKIP () << "g_hash_table_insert() is not interposed.";
  }

  std::thread reader ([&looked_up] {
    hook_mask = HOOK_LOOKUP;
    looked_up = nnsconf_get_custom_value_string ("b4", "key");
  });

  writer.join ();
  reader.join ();
  g_unsetenv (envname);

  EXPECT_EQ (region_max, 1);
  EXPECT_EQ (region_calls, 2);
  EXPECT_STREQ (inserted, "envvalue");
  EXPECT_STREQ (looked_up, "value");
  g_free (inserted);
  g_free (looked_up);
  g_free (envname);
  g_free (key);
}

/**
 * @brief A cache miss that started before a forced reload waits for the reload
 *        to finish before it reads the ini file.
 */
TEST_F (NNSConfConcurrency, customMissWaitsForReload)
{
  std::atomic<bool> done{ false };
  gchar *value = NULL;
  bool done_during_reload;

  gchar *primed = nnsconf_get_custom_value_string ("b4", "key");
  ASSERT_STREQ (primed, "value");
  g_free (primed);

  _gate_reset ();
  std::thread reader ([this, &value, &done] {
    hook_mask = HOOK_LOOKUP_HOLD;
    value = nnsconf_get_custom_value_string (reload_group, "key");
    done = true;
  });

  if (!_gate_wait_arrived (GATE_LOOKUP)) {
    reader.join ();
    g_free (value);
    GTEST_SKIP () << "g_hash_table_lookup() is not interposed.";
  }

  std::thread reloader ([] {
    hook_mask = HOOK_STRFREEV_HOLD;
    EXPECT_TRUE (nnsconf_loadconf (TRUE));
  });

  if (!_gate_wait_arrived (GATE_STRFREEV)) {
    _gate_open (GATE_LOOKUP);
    reloader.join ();
    reader.join ();
    g_free (value);
    GTEST_SKIP () << "g_strfreev() is not interposed.";
  }

  _gate_open (GATE_LOOKUP);
  g_usleep (200000);
  done_during_reload = done;
  _gate_open (GATE_STRFREEV);
  reloader.join ();
  reader.join ();

  EXPECT_FALSE (done_during_reload);
  EXPECT_STREQ (value, "reloaded");
  g_free (value);
}

/**
 * @brief Loads arriving while a forced reload is resetting the configuration
 *        wait for the reload and then see the complete configuration.
 */
TEST_F (NNSConfConcurrency, loadWaitsForReloadReset)
{
  std::atomic<int> done{ 0 };
  const gchar *path = NULL;
  gboolean valid = FALSE;
  int done_during_reload;

  _gate_reset ();
  std::thread reloader ([] {
    hook_mask = HOOK_STRFREEV_HOLD;
    EXPECT_TRUE (nnsconf_loadconf (TRUE));
  });

  if (!_gate_wait_arrived (GATE_STRFREEV)) {
    reloader.join ();
    GTEST_SKIP () << "g_strfreev() is not interposed.";
  }

  std::thread finder ([&path, &done] {
    path = nnsconf_get_fullpath ("b4dummy", NNSCONF_PATH_FILTERS);
    done++;
  });
  std::thread validator ([this, &valid, &done] {
    valid = nnsconf_validate_file (NNSCONF_PATH_FILTERS, filter);
    done++;
  });

  g_usleep (200000);
  done_during_reload = done;
  _gate_open (GATE_STRFREEV);
  reloader.join ();
  finder.join ();
  validator.join ();

  EXPECT_EQ (done_during_reload, 0);
  EXPECT_STREQ (path, filter);
  EXPECT_TRUE (valid);
}

/**
 * @brief Concurrent first lookups of the same keys fill the cache consistently.
 */
TEST_F (NNSConfConcurrency, customLookupStress)
{
  std::vector<std::thread> threads;
  std::mutex fail_lock;
  guint failures = 0;
  guint t, i;

  for (t = 0; t < STRESS_THREADS; t++) {
    threads.emplace_back ([&fail_lock, &failures, t] {
      guint n;

      for (n = 0; n < STRESS_KEYS; n++) {
        guint k = (n + t * 7) % STRESS_KEYS;
        std::string key = "k" + std::to_string (k);
        std::string expected = "v" + std::to_string (k);
        gchar *v = nnsconf_get_custom_value_string ("b4stress", key.c_str ());

        if (g_strcmp0 (v, expected.c_str ()) != 0) {
          std::lock_guard<std::mutex> lk (fail_lock);
          failures++;
        }
        g_free (v);
      }
    });
  }
  for (auto &th : threads)
    th.join ();

  EXPECT_EQ (failures, 0U);

  for (i = 0; i < STRESS_KEYS; i++) {
    std::string key = "k" + std::to_string (i);
    std::string expected = "v" + std::to_string (i);
    gchar *v = nnsconf_get_custom_value_string ("b4stress", key.c_str ());

    EXPECT_STREQ (v, expected.c_str ());
    g_free (v);
  }
}

/**
 * @brief Concurrent forced reloads leave a complete configuration.
 */
TEST_F (NNSConfConcurrency, forceReloadStress)
{
  std::vector<std::thread> threads;
  guint t;

  for (t = 0; t < STRESS_THREADS; t++) {
    threads.emplace_back ([] {
      guint n;

      for (n = 0; n < 16; n++)
        EXPECT_TRUE (nnsconf_loadconf (TRUE));
    });
  }
  for (auto &th : threads)
    th.join ();

  EXPECT_STREQ (nnsconf_get_fullpath ("b4dummy", NNSCONF_PATH_FILTERS), filter);
}

/**
 * @brief Concurrent lookups of an absent key return nothing and cache nothing.
 */
TEST_F (NNSConfConcurrency, customLookupAbsent_n)
{
  std::vector<std::thread> threads;
  std::mutex fail_lock;
  guint failures = 0;
  guint t;

  for (t = 0; t < STRESS_THREADS; t++) {
    threads.emplace_back ([&fail_lock, &failures] {
      guint n;

      for (n = 0; n < 32; n++) {
        gchar *v = nnsconf_get_custom_value_string ("b4", "absent");

        if (v != NULL || nnsconf_get_custom_value_bool ("b4", "absent", TRUE) != TRUE
            || nnsconf_get_custom_value_bool ("b4", "absent", FALSE) != FALSE) {
          std::lock_guard<std::mutex> lk (fail_lock);
          failures++;
        }
        g_free (v);
      }
    });
  }
  for (auto &th : threads)
    th.join ();

  EXPECT_EQ (failures, 0U);
}

/**
 * @brief An unknown sub-plugin type yields no sub-plugins.
 */
TEST_F (NNSConfConcurrency, subpluginInfoInvalidType_n)
{
  subplugin_info_s info;

  EXPECT_EQ (nnsconf_get_subplugin_info (NNSCONF_PATH_END, &info), 0U);
  EXPECT_TRUE (info.names == NULL);
  EXPECT_TRUE (info.paths == NULL);
  EXPECT_TRUE (nnsconf_get_fullpath ("b4dummy", NNSCONF_PATH_END) == NULL);
}

/**
 * @brief Fixture for the sub-plugin dump. The sub-plugin paths from the
 *        environment are dropped, and the cases skip when the hard-coded
 *        directories hold installed sub-plugins, so no real sub-plugin is
 *        loaded. The ini's
 *        filter directory holds only the unloadable b4dummy, and its decoder
 *        directory holds b4reg and b4prop, which are registered without
 *        loading them; b4prop has a custom property.
 */
class NNSConfDump : public NNSConfConcurrency
{
  protected:
  const gchar *path_envs[5] = { "NNSTREAMER_FILTERS", "NNSTREAMER_DECODERS",
    "NNSTREAMER_CUSTOMFILTERS", "NNSTREAMER_CONVERTERS", "NNSTREAMER_TRAINERS" };
  gchar *saved_paths[5] = { NULL };
  gchar *propdecoder = NULL;
  bool reg_registered = false;
  bool prop_registered = false;

  /**
   * @brief Drop the sub-plugin path variables, then set up the ini, b4reg
   *        and b4prop.
   */
  void SetUp () override
  {
    static int b4reg_data;
    guint i;

    for (i = 0; i < 5; i++) {
      saved_paths[i] = g_strdup (g_getenv (path_envs[i]));
      g_unsetenv (path_envs[i]);
    }

    NNSConfConcurrency::SetUp ();
    if (IsSkipped () || HasFatalFailure ())
      return;

    propdecoder = g_build_filename (decoderdir,
        "libnnstreamer_decoder_b4prop" NNSTREAMER_SO_FILE_EXTENSION, NULL);
    ASSERT_TRUE (g_file_set_contents (propdecoder, "", 0, NULL));
    ASSERT_TRUE (nnsconf_loadconf (TRUE));

    for (nnsconf_type_path type :
        { NNSCONF_PATH_DECODERS, NNSCONF_PATH_CONVERTERS, NNSCONF_PATH_TRAINERS }) {
      subplugin_info_s info;
      guint total = nnsconf_get_subplugin_info (type, &info);

      for (i = 0; i < total; i++) {
        if (g_strcmp0 (info.names[i], "b4reg") != 0
            && g_strcmp0 (info.names[i], "b4prop") != 0)
          GTEST_SKIP () << "An installed sub-plugin would be loaded: " << info.paths[i];
      }
    }

    reg_registered = register_subplugin (NNS_SUBPLUGIN_DECODER, "b4reg", &b4reg_data);
    ASSERT_TRUE (reg_registered);
    prop_registered = register_subplugin (NNS_SUBPLUGIN_DECODER, "b4prop", &b4reg_data);
    ASSERT_TRUE (prop_registered);
    nnstreamer_decoder_set_custom_property_desc ("b4prop", "option1", "b4 description", NULL);
  }

  /**
   * @brief Unregister b4reg and restore the sub-plugin path variables.
   */
  void TearDown () override
  {
    guint i;

    if (reg_registered)
      unregister_subplugin (NNS_SUBPLUGIN_DECODER, "b4reg");
    if (prop_registered)
      unregister_subplugin (NNS_SUBPLUGIN_DECODER, "b4prop");
    if (propdecoder)
      g_remove (propdecoder);
    g_free (propdecoder);

    for (i = 0; i < 5; i++) {
      if (saved_paths[i])
        g_setenv (path_envs[i], saved_paths[i], TRUE);
      g_free (saved_paths[i]);
    }

    NNSConfConcurrency::TearDown ();
  }
};

/**
 * @brief The sub-plugin dump lists every section and the loadable sub-plugin.
 */
TEST_F (NNSConfDump, subpluginDump)
{
  static const gchar *headers[]
      = { "\n[Filter]\n", "\n[Decoder]\n", "\n[Conterver]\n", "\n[Trainer]\n" };
  const gsize size = 65536;
  gchar *dump = (gchar *) g_malloc0 (size);
  const gchar *prev = dump;

  nnsconf_subplugin_dump (dump, size);

  for (const gchar *header : headers) {
    const gchar *pos = strstr (prev, header);

    EXPECT_TRUE (pos != NULL) << header;
    if (pos)
      prev = pos;
  }
  EXPECT_TRUE (strstr (dump, "\n  b4reg\n    - No custom property found\n") != NULL);
  EXPECT_TRUE (strstr (dump, "\n  b4prop\n    - option1: b4 description\n") != NULL);
  g_free (dump);
}

/**
 * @brief The sub-plugin dump does not list a sub-plugin that cannot be loaded.
 */
TEST_F (NNSConfDump, subpluginDumpUnloadable_n)
{
  const gsize size = 65536;
  gchar *dump = (gchar *) g_malloc0 (size);

  nnsconf_subplugin_dump (dump, size);

  EXPECT_TRUE (strstr (dump, "\n[Filter]\n\n[Decoder]\n") != NULL);
  EXPECT_TRUE (strstr (dump, "b4dummy") == NULL);
  g_free (dump);
}

/**
 * @brief A too small buffer gets a prefix of the sub-plugin dump, whatever
 *        line the dump stops in, and nothing past its end is written.
 */
TEST_F (NNSConfDump, subpluginDumpTruncated_n)
{
  const gsize size = 65536;
  gchar *full = (gchar *) g_malloc0 (size);
  gchar *dump;
  gsize len, n, i;
  guint failures = 0;

  nnsconf_subplugin_dump (full, size);
  len = strlen (full);
  ASSERT_TRUE (strstr (full, "  b4reg\n") != NULL);
  ASSERT_TRUE (strstr (full, "  b4prop\n") != NULL);
  dump = (gchar *) g_malloc (len + 16);

  for (n = 1; n <= len; n++) {
    memset (dump, 'x', len + 16);
    nnsconf_subplugin_dump (dump, n);

    if (strncmp (dump, full, n - 1) != 0 || dump[n - 1] != '\0')
      failures++;
    for (i = n; i < len + 16; i++) {
      if (dump[i] != 'x') {
        failures++;
        break;
      }
    }
  }

  EXPECT_EQ (failures, 0U);
  g_free (dump);
  g_free (full);
}

/**
 * @brief Fixture with two ini files, A (enable_envvar=true) and B
 *        (enable_envvar=false), each with its own filter directory and custom
 *        values, and a third filter directory given by NNSTREAMER_FILTERS.
 *        The custom value group is new in every case because the custom value
 *        cache outlives a forced reload.
 */
class NNSConfLookup : public ::testing::Test
{
  protected:
  gchar *dir = NULL;
  gchar *dirs[3] = { NULL };
  gchar *filters[3] = { NULL };
  gchar *ini_a = NULL;
  gchar *ini_b = NULL;
  gchar *group = NULL;
  gchar *envkey = NULL;
  gchar *saved_conf = NULL;
  gchar *saved_filters = NULL;

  /**
   * @brief Write both ini files and the filters, then load ini A.
   */
  void SetUp () override
  {
    static guint setup_serial = 0;
    static const gchar *names[3] = { "j6a", "j6b", "j6env" };
    gchar *content;
    guint i;

    group = g_strdup_printf ("j6grp%u", setup_serial++);
    envkey = g_strdup_printf ("NNSTREAMER_%s_key", group);
    dir = g_dir_make_tmp ("nns-conf-XXXXXX", NULL);
    ASSERT_TRUE (dir != NULL);

    for (i = 0; i < 3; i++) {
      gchar *file = g_strdup_printf (
          "libnnstreamer_filter_%s" NNSTREAMER_SO_FILE_EXTENSION, names[i]);

      dirs[i] = g_build_filename (dir, names[i], NULL);
      filters[i] = g_build_filename (dirs[i], file, NULL);
      g_free (file);
      ASSERT_EQ (g_mkdir (dirs[i], 0755), 0);
      ASSERT_TRUE (g_file_set_contents (filters[i], "", 0, NULL));
    }

    ini_a = g_build_filename (dir, "a.ini", NULL);
    content = g_strdup_printf ("[common]\nenable_envvar=True\n"
                               "[filter]\nfilters=%s\n"
                               "[%s]\nkey=ini-a\nother=ini-a\n",
        dirs[0], group);
    ASSERT_TRUE (g_file_set_contents (ini_a, content, -1, NULL));
    g_free (content);

    ini_b = g_build_filename (dir, "b.ini", NULL);
    content = g_strdup_printf ("[common]\nenable_envvar=False\n"
                               "[filter]\nfilters=%s\n"
                               "[%s]\nkey=ini-b\n",
        dirs[1], group);
    ASSERT_TRUE (g_file_set_contents (ini_b, content, -1, NULL));
    g_free (content);

    saved_conf = g_strdup (g_getenv ("NNSTREAMER_CONF"));
    saved_filters = g_strdup (g_getenv ("NNSTREAMER_FILTERS"));
    g_setenv ("NNSTREAMER_FILTERS", dirs[2], TRUE);

    ASSERT_NO_FATAL_FAILURE (UseConf (ini_a));
#ifdef __TIZEN__
    /* Tizen reads NNSTREAMER_CONF only when no system ini exists. */
    if (g_strcmp0 (nnsconf_get_fullpath ("j6a", NNSCONF_PATH_FILTERS), filters[0]) != 0)
      GTEST_SKIP () << "The ini given by NNSTREAMER_CONF is not used.";
#endif
  }

  /**
   * @brief Restore the environment, reload and remove the files.
   */
  void TearDown () override
  {
    guint i;

    if (envkey)
      g_unsetenv (envkey);
    if (saved_filters)
      g_setenv ("NNSTREAMER_FILTERS", saved_filters, TRUE);
    else
      g_unsetenv ("NNSTREAMER_FILTERS");
    if (saved_conf)
      g_setenv ("NNSTREAMER_CONF", saved_conf, TRUE);
    else
      g_unsetenv ("NNSTREAMER_CONF");
    nnsconf_loadconf (TRUE);

    if (ini_a)
      g_remove (ini_a);
    if (ini_b)
      g_remove (ini_b);
    for (i = 0; i < 3; i++) {
      if (filters[i])
        g_remove (filters[i]);
      if (dirs[i])
        g_rmdir (dirs[i]);
      g_free (filters[i]);
      g_free (dirs[i]);
    }
    if (dir)
      g_rmdir (dir);
    g_free (ini_a);
    g_free (ini_b);
    g_free (group);
    g_free (envkey);
    g_free (saved_conf);
    g_free (saved_filters);
    g_free (dir);
  }

  /**
   * @brief Point NNSTREAMER_CONF at path and reload.
   */
  void UseConf (const gchar *path)
  {
    g_setenv ("NNSTREAMER_CONF", path, TRUE);
    ASSERT_TRUE (nnsconf_loadconf (TRUE));
  }

  /**
   * @brief Get the rest of the first line of nnsconf_dump() containing label.
   */
  std::string DumpValue (const gchar *label)
  {
    gchar dump[8192] = { 0 };
    const gchar *start, *end;

    nnsconf_dump (dump, sizeof (dump));
    start = strstr (dump, label);
    if (!start)
      return std::string ("<missing>");
    start += strlen (label);
    end = strchr (start, '\n');
    return end ? std::string (start, end - start) : std::string (start);
  }
};

/**
 * @brief NNSTREAMER_CONF selects the ini file over the build-time one, and a
 *        reload follows a new NNSTREAMER_CONF.
 */
TEST_F (NNSConfLookup, envConfSelectsFile)
{
  EXPECT_EQ (DumpValue ("Configuration file path: "), std::string (ini_a));
  EXPECT_STREQ (nnsconf_get_fullpath ("j6a", NNSCONF_PATH_FILTERS), filters[0]);
  EXPECT_TRUE (nnsconf_get_fullpath ("j6b", NNSCONF_PATH_FILTERS) == NULL);

  ASSERT_NO_FATAL_FAILURE (UseConf (ini_b));
  EXPECT_EQ (DumpValue ("Configuration file path: "), std::string (ini_b));
  EXPECT_STREQ (nnsconf_get_fullpath ("j6b", NNSCONF_PATH_FILTERS), filters[1]);
  EXPECT_TRUE (nnsconf_get_fullpath ("j6a", NNSCONF_PATH_FILTERS) == NULL);
}

/**
 * @brief NNSTREAMER_FILTERS adds its directory only when the ini enables
 *        environment variables.
 */
TEST_F (NNSConfLookup, envvarDisabledIgnoresPathEnv_n)
{
  EXPECT_EQ (DumpValue ("Enable envvar: "), std::string ("TRUE"));
  EXPECT_EQ (DumpValue ("from envvar: "), std::string (dirs[2]));
  EXPECT_STREQ (nnsconf_get_fullpath ("j6env", NNSCONF_PATH_FILTERS), filters[2]);

  ASSERT_NO_FATAL_FAILURE (UseConf (ini_b));
  EXPECT_EQ (DumpValue ("Enable envvar: "), std::string ("FALSE"));
  EXPECT_EQ (DumpValue ("from envvar: "), std::string ("<disabled>"));
  EXPECT_TRUE (nnsconf_get_fullpath ("j6env", NNSCONF_PATH_FILTERS) == NULL);
  EXPECT_STREQ (nnsconf_get_fullpath ("j6b", NNSCONF_PATH_FILTERS), filters[1]);
}

/**
 * @brief A custom value from the environment beats the ini value, and the
 *        value read first stays cached after the variable is unset.
 */
TEST_F (NNSConfLookup, customEnvBeatsIni)
{
  gchar *value;

  g_setenv (envkey, "env", TRUE);
  value = nnsconf_get_custom_value_string (group, "key");
  EXPECT_STREQ (value, "env");
  g_free (value);

  value = nnsconf_get_custom_value_string (group, "other");
  EXPECT_STREQ (value, "ini-a");
  g_free (value);

  g_unsetenv (envkey);
  value = nnsconf_get_custom_value_string (group, "key");
  EXPECT_STREQ (value, "env");
  g_free (value);
}

/**
 * @brief A custom value from the environment is ignored when the ini does not
 *        enable environment variables.
 */
TEST_F (NNSConfLookup, customEnvDisabled_n)
{
  gchar *envonly = g_strdup_printf ("NNSTREAMER_%s_envonly", group);
  gchar *value;

  ASSERT_NO_FATAL_FAILURE (UseConf (ini_b));
  g_setenv (envkey, "env", TRUE);
  g_setenv (envonly, "env", TRUE);

  value = nnsconf_get_custom_value_string (group, "key");
  EXPECT_STREQ (value, "ini-b");
  g_free (value);
  EXPECT_TRUE (nnsconf_get_custom_value_string (group, "envonly") == NULL);
  EXPECT_FALSE (nnsconf_get_custom_value_bool (group, "envonly", FALSE));

  g_unsetenv (envonly);
  g_free (envonly);
}

/**
 * @brief An NNSTREAMER_CONF naming no file falls back to the build-time ini,
 *        then to the default one, and nothing of the previous ini remains.
 */
TEST_F (NNSConfLookup, missingEnvConfFallsBack_n)
{
  gchar *absent = g_build_filename (dir, "absent.ini", NULL);
  std::string build_conf = DumpValue ("build-config: ");
  /* The first "hard-coded: " is the default ini, not the filter path. */
  std::string default_conf = DumpValue ("hard-coded: ");
  std::string path;

  ASSERT_NO_FATAL_FAILURE (UseConf (absent));
  path = DumpValue ("Configuration file path: ");

  EXPECT_NE (path, std::string (ini_a));
  if (g_file_test (build_conf.c_str (), G_FILE_TEST_IS_REGULAR)) {
    EXPECT_EQ (path, build_conf);
  } else if (g_file_test (default_conf.c_str (), G_FILE_TEST_IS_REGULAR)) {
    EXPECT_EQ (path, default_conf);
  }

  EXPECT_TRUE (nnsconf_get_fullpath ("j6a", NNSCONF_PATH_FILTERS) == NULL);
  EXPECT_TRUE (nnsconf_get_custom_value_string (group, "key") == NULL);
  g_free (absent);
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

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
