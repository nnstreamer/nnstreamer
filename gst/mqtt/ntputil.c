/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    ntputil.c
 * @date    16 Jul 2021
 * @brief   NTP utility functions
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 * @todo    Need to support caching and polling timer mechanism
 */

#include <errno.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "ntputil.h"

/**
 *******************************************************************
 * NTP Timestamp Format (https://www.ietf.org/rfc/rfc5905.txt p.12)
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                            Seconds                            |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                            Fraction                           |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *******************************************************************
 */
/**
 * @brief A custom data type to represent NTP timestamp format
 */
typedef struct _ntp_timestamp_t
{
  uint32_t sec;
  uint32_t frac;
} ntp_timestamp_t;

/**
 *******************************************************************
 * NTP Packet Header Format (https://www.ietf.org/rfc/rfc5905.txt p.18)
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |LI | VN  |Mode |    Stratum     |     Poll      |  Precision   |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                         Root Delay                            |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                         Root Dispersion                       |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                          Reference ID                         |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                               |
 * +                     Reference Timestamp (64)                  +
 * |                                                               |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                               |
 * +                      Origin Timestamp (64)                    +
 * |                                                               |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                               |
 * +                      Receive Timestamp (64)                   +
 * |                                                               |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                               |
 * +                      Transmit Timestamp (64)                  +
 * |                                                               |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                               |
 * .                                                               .
 * .                    Extension Field 1 (variable)               .
 * .                                                               .
 * |                                                               |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                               |
 * .                                                               .
 * .                    Extension Field 2 (variable)               .
 * .                                                               .
 * |                                                               |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                          Key Identifier                       |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                               |
 * |                            dgst (128)                         |
 * |                                                               |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *******************************************************************
 */

/**
 * @brief A custom data type to represent NTP packet header format
 */
typedef struct _ntp_packet_t
{
  uint8_t li_vn_mode;
  uint8_t stratum;
  uint8_t poll;
  uint8_t precision;
  uint32_t root_delay;
  uint32_t root_dispersion;
  uint32_t ref_id;
  ntp_timestamp_t ref_ts;
  ntp_timestamp_t org_ts;
  ntp_timestamp_t recv_ts;
  ntp_timestamp_t xmit_ts;
} ntp_packet_t;

const uint64_t NTPUTIL_TIMESTAMP_DELTA = 2208988800ULL;
const double NTPUTIL_MAX_FRAC_DOUBLE = 4294967295.0L;
const int64_t NTPUTIL_SEC_TO_USEC_MULTIPLIER = 1000000;
const char NTPUTIL_DEFAULT_HNAME[] = "pool.ntp.org";
const uint16_t NTPUTIL_DEFAULT_PORT = 123;
const time_t NTPUTIL_RECV_TIMEOUT_SEC = 1;
const int64_t NTPUTIL_RETRY_HOLD_OFF_USEC = 10 * 1000000;
const uint32_t NTPUTIL_FAILURES_BEFORE_HOLD_OFF = 3;

#define NTPUTIL_CACHE_ENTRIES 8

/**
 * @brief The server address resolved for a host list, so that a healthy
 *        server is not looked up again on every query, and the failure
 *        streak and hold-off deadline of that list.
 */
typedef struct _ntputil_cache_entry_t
{
  int keyed;
  int resolved;
  uint32_t failures;            /* atomic store under ntputil_cache_lock, atomic load outside */
  int64_t retry_after;
  uint64_t last_used;
  uint32_t hnums;
  char **hnames;
  uint16_t *ports;
  struct sockaddr_in addr;
} ntputil_cache_entry_t;

/**
 * @brief One entry per host list in use, shared by all callers in the
 *        process; the least recently used entry is replaced when all are
 *        taken.
 */
static ntputil_cache_entry_t ntputil_cache[NTPUTIL_CACHE_ENTRIES];
static uint64_t ntputil_cache_tick;
static pthread_mutex_t ntputil_cache_lock = PTHREAD_MUTEX_INITIALIZER;

/**
 * @brief Serializes gethostbyname (), which returns a static buffer, without
 *        holding ntputil_cache_lock while a lookup waits for DNS.
 */
static pthread_mutex_t ntputil_resolve_lock = PTHREAD_MUTEX_INITIALIZER;

/**
 * @brief Wrapper function of ntohl.
 */
uint32_t
_convert_to_host_byte_order (uint32_t in)
{
  return ntohl (in);
}

/**
 * @brief Get the monotonic time in microseconds.
 */
int64_t
_get_monotonic_time_us (void)
{
  struct timespec ts;

  clock_gettime (CLOCK_MONOTONIC, &ts);
  return (int64_t) ts.tv_sec * NTPUTIL_SEC_TO_USEC_MULTIPLIER +
      ts.tv_nsec / 1000;
}

/**
 * @brief Set the failure streak of a cache entry. It is stored atomically so
 *        that a successful query can check it without taking the lock.
 * @note The caller must hold ntputil_cache_lock.
 */
static void
_cache_set_failures (ntputil_cache_entry_t * e, uint32_t failures)
{
  __atomic_store_n (&e->failures, failures, __ATOMIC_RELAXED);
}

/**
 * @brief Find the cache entry for the given hosts.
 * @note The caller must hold ntputil_cache_lock.
 */
static ntputil_cache_entry_t *
_cache_find (uint32_t hnums, char **hnames, uint16_t * ports)
{
  uint32_t i, j;

  for (i = 0; i < NTPUTIL_CACHE_ENTRIES; ++i) {
    ntputil_cache_entry_t *e = &ntputil_cache[i];

    if (!e->keyed || e->hnums != hnums)
      continue;

    for (j = 0; j < hnums; ++j) {
      if (e->ports[j] != ports[j] || strcmp (e->hnames[j], hnames[j]) != 0)
        break;
    }

    if (j == hnums)
      return e;
  }

  return NULL;
}

/**
 * @brief Drop a cache entry.
 * @note The caller must hold ntputil_cache_lock.
 */
static void
_cache_clear (ntputil_cache_entry_t * e)
{
  uint32_t i;

  _cache_set_failures (e, 0);
  for (i = 0; i < e->hnums; ++i)
    free (e->hnames[i]);
  free (e->hnames);
  free (e->ports);

  e->keyed = 0;
  e->resolved = 0;
  e->retry_after = 0;
  e->last_used = 0;
  e->hnums = 0;
  e->hnames = NULL;
  e->ports = NULL;
  memset (&e->addr, 0, sizeof (e->addr));
}

/**
 * @brief Start a cache entry for the given hosts in a free or the least
 *        recently used slot.
 * @return the entry, or NULL if memory runs out; the next query then resolves
 *         again
 * @note The caller must hold ntputil_cache_lock.
 */
static ntputil_cache_entry_t *
_cache_new (uint32_t hnums, char **hnames, uint16_t * ports)
{
  ntputil_cache_entry_t *e = &ntputil_cache[0];
  uint32_t i;

  for (i = 0; i < NTPUTIL_CACHE_ENTRIES && e->keyed; ++i) {
    if (!ntputil_cache[i].keyed || ntputil_cache[i].last_used < e->last_used)
      e = &ntputil_cache[i];
  }

  _cache_clear (e);

  if (hnums > 0) {
    e->hnames = calloc (hnums, sizeof (char *));
    e->ports = calloc (hnums, sizeof (uint16_t));
    if (!e->hnames || !e->ports) {
      _cache_clear (e);
      return NULL;
    }

    e->hnums = hnums;
    for (i = 0; i < hnums; ++i) {
      e->hnames[i] = strdup (hnames[i]);
      if (!e->hnames[i]) {
        _cache_clear (e);
        return NULL;
      }
      e->ports[i] = ports[i];
    }
  }

  e->keyed = 1;
  return e;
}

/**
 * @brief Count a failure of the hosts of a cache entry, and hold them off
 *        once they have failed NTPUTIL_FAILURES_BEFORE_HOLD_OFF times in a row.
 * @note The caller must hold ntputil_cache_lock.
 */
static void
_cache_count_failure (ntputil_cache_entry_t * e)
{
  _cache_set_failures (e, e->failures + 1);
  if (e->failures >= NTPUTIL_FAILURES_BEFORE_HOLD_OFF)
    e->retry_after = _get_monotonic_time_us () + NTPUTIL_RETRY_HOLD_OFF_USEC;
}

/**
 * @brief Look up the given hosts in the cache.
 * @param[out] entry set to their cache entry, or NULL
 * @return 1 with serv_addr set if their address is cached, -EAGAIN while they
 *         are held off, 0 if they need to be resolved
 * @note The caller must hold ntputil_cache_lock.
 */
static int64_t
_cache_lookup (uint32_t hnums, char **hnames, uint16_t * ports,
    struct sockaddr_in *serv_addr, ntputil_cache_entry_t ** entry)
{
  ntputil_cache_entry_t *e = _cache_find (hnums, hnames, ports);

  *entry = e;
  if (!e)
    return 0;

  e->last_used = ++ntputil_cache_tick;
  if (e->retry_after != 0) {
    if (_get_monotonic_time_us () < e->retry_after)
      return -EAGAIN;
    e->retry_after = 0;
  }

  if (e->resolved) {
    *serv_addr = e->addr;
    return 1;
  }

  return 0;
}

/**
 * @brief Resolve the first resolvable host, or the NTP server pool if none is.
 * @param[out] entry set to the cache entry of the hosts, or NULL; it is only a
 *             hint, since the entry may be reused once the lock is released
 * @return 0 on success, a negative value on error or while the hosts are held
 *         off after repeated failures
 */
static int64_t
_resolve_server (uint32_t hnums, char **hnames, uint16_t * ports,
    struct sockaddr_in *serv_addr, ntputil_cache_entry_t ** entry)
{
  ntputil_cache_entry_t *e;
  struct hostent *srv = NULL;
  uint16_t port = 0;
  uint32_t i;
  int64_t ret = 0;

  pthread_mutex_lock (&ntputil_cache_lock);
  ret = _cache_lookup (hnums, hnames, ports, serv_addr, &e);
  pthread_mutex_unlock (&ntputil_cache_lock);
  if (ret != 0)
    goto out;

  pthread_mutex_lock (&ntputil_resolve_lock);

  /* Another caller may have resolved or failed these hosts meanwhile */
  pthread_mutex_lock (&ntputil_cache_lock);
  ret = _cache_lookup (hnums, hnames, ports, serv_addr, &e);
  pthread_mutex_unlock (&ntputil_cache_lock);
  if (ret != 0) {
    pthread_mutex_unlock (&ntputil_resolve_lock);
    goto out;
  }

  for (i = 0; i < hnums; ++i) {
    srv = gethostbyname (hnames[i]);
    if (srv != NULL) {
      port = ports[i];
      break;
    }
  }

  if (srv == NULL) {
    srv = gethostbyname (NTPUTIL_DEFAULT_HNAME);
    if (srv == NULL)
      ret = (h_errno > 0) ? -h_errno : -1;
    else
      port = NTPUTIL_DEFAULT_PORT;
  }

  if (srv != NULL) {
    memset (serv_addr, 0, sizeof (*serv_addr));
    serv_addr->sin_family = AF_INET;
    memcpy ((uint8_t *) & serv_addr->sin_addr.s_addr,
        (uint8_t *) srv->h_addr_list[0], (size_t) srv->h_length);
    serv_addr->sin_port = htons (port);
  }

  /* Store before releasing the resolve lock so that its waiters find it */
  pthread_mutex_lock (&ntputil_cache_lock);
  e = _cache_find (hnums, hnames, ports);
  if (!e)
    e = _cache_new (hnums, hnames, ports);
  if (e) {
    e->last_used = ++ntputil_cache_tick;
    if (ret < 0) {
      _cache_count_failure (e);
    } else {
      e->addr = *serv_addr;
      e->resolved = 1;
    }
  }
  pthread_mutex_unlock (&ntputil_cache_lock);
  pthread_mutex_unlock (&ntputil_resolve_lock);

out:
  *entry = e;
  return (ret < 0) ? ret : 0;
}

/**
 * @brief Forget the cached address of the given hosts if it is the one that
 *        failed and count the failure, so that the hosts are resolved again on
 *        the next query that is not held off.
 */
static void
_cache_invalidate (uint32_t hnums, char **hnames, uint16_t * ports,
    const struct sockaddr_in *serv_addr)
{
  ntputil_cache_entry_t *e;

  pthread_mutex_lock (&ntputil_cache_lock);
  e = _cache_find (hnums, hnames, ports);
  if (e && e->resolved &&
      memcmp (&e->addr, serv_addr, sizeof (*serv_addr)) == 0) {
    e->resolved = 0;
    _cache_count_failure (e);
  }
  pthread_mutex_unlock (&ntputil_cache_lock);
}

/**
 * @brief End the failure streak of the given hosts after a query to them
 *        succeeded, even if a concurrent failure has invalidated their address.
 */
static void
_cache_reset_failures (uint32_t hnums, char **hnames, uint16_t * ports)
{
  ntputil_cache_entry_t *e;

  pthread_mutex_lock (&ntputil_cache_lock);
  e = _cache_find (hnums, hnames, ports);
  if (e)
    _cache_set_failures (e, 0);
  pthread_mutex_unlock (&ntputil_cache_lock);
}

/**
 * @brief Get NTP timestamps from the given or public NTP servers
 * @param[in] hnums A number of hostname and port pairs. If 0 is given,
 *                  the NTP server pool will be used.
 * @param[in] hnames A list of hostname
 * @param[in] ports A list of port
 * @return an Unix epoch time as microseconds on success,
 *         negative values on error
 * @note The server address is resolved once per host list and reused until
 *       a query to it fails. A query gives up if no reply arrives within
 *       NTPUTIL_RECV_TIMEOUT_SEC seconds, and after
 *       NTPUTIL_FAILURES_BEFORE_HOLD_OFF failures in a row the same hosts are
 *       not queried again for NTPUTIL_RETRY_HOLD_OFF_USEC.
 */
int64_t
ntputil_get_epoch (uint32_t hnums, char **hnames, uint16_t * ports)
{
  struct sockaddr_in serv_addr;
  struct timeval timeout = { NTPUTIL_RECV_TIMEOUT_SEC, 0 };
  ntputil_cache_entry_t *entry = NULL;
  int32_t sockfd = -1;
  int64_t ret;

  ret = _resolve_server (hnums, hnames, ports, &serv_addr, &entry);
  if (ret < 0)
    goto ret_normal;

  sockfd = socket (AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sockfd < 0) {
    ret = -1;
    goto ret_normal;
  }

  if (setsockopt (sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
          sizeof (timeout)) < 0) {
    ret = -errno;
    goto ret_close_sockfd;
  }

  ret = connect (sockfd, (struct sockaddr *) &serv_addr, sizeof (serv_addr));
  if (ret < 0) {
    ret = -errno;
    goto ret_invalidate;
  }

  {
    ntp_packet_t packet;
    uint32_t recv_sec;
    uint32_t recv_frac;
    double frac;
    ssize_t n;

    memset (&packet, 0, sizeof (packet));

    /* li = 0, vn = 3, mode = 3 */
    packet.li_vn_mode = 0x1B;

    /* Request */
    n = write (sockfd, &packet, sizeof (packet));
    if (n < 0) {
      ret = -errno;
      goto ret_invalidate;
    }

    /* Receive */
    n = read (sockfd, &packet, sizeof (packet));
    if (n < 0) {
      ret = -errno;
      goto ret_invalidate;
    }
    if ((size_t) n < sizeof (packet)) {
      ret = -1;
      goto ret_invalidate;
    }

    /**
     * @note ntp_timestamp_t recv_ts in ntp_packet_t means the timestamp as the packet
     * left the NTP server. 'sec' corresponds to the seconds passed since 1900
     * and 'frac' is needed to convert seconds to smaller units of a second
     * such as microsceonds. Note that the bit/byte order of those data should
     * be converted to the host's endianness.
     */
    recv_sec = _convert_to_host_byte_order (packet.xmit_ts.sec);
    recv_frac = _convert_to_host_byte_order (packet.xmit_ts.frac);

    /**
     * @note NTP uses an epoch of January 1, 1900 while the Unix epoch is
     * the number of seconds that have elapsed since January 1, 1970. For this
     * reason, we subtract 70 years worth of seconds from the seconds since 1900
     */
    if (recv_sec <= NTPUTIL_TIMESTAMP_DELTA) {
      ret = -1;
      goto ret_invalidate;
    }

    ret = (int64_t) (recv_sec - NTPUTIL_TIMESTAMP_DELTA);
    ret *= NTPUTIL_SEC_TO_USEC_MULTIPLIER;
    frac = ((double) recv_frac) / NTPUTIL_MAX_FRAC_DOUBLE;
    frac *= NTPUTIL_SEC_TO_USEC_MULTIPLIER;

    ret += (int64_t) frac;
  }

  if (entry && __atomic_load_n (&entry->failures, __ATOMIC_RELAXED) != 0)
    _cache_reset_failures (hnums, hnames, ports);
  goto ret_close_sockfd;

ret_invalidate:
  _cache_invalidate (hnums, hnames, ports, &serv_addr);

ret_close_sockfd:
  close (sockfd);

ret_normal:
  return ret;
}
