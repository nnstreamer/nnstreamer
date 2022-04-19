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
 * @todo    Need to support cacheing and polling timer mechanism
 */

#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
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

/**
 * @brief Wrapper function of ntohl.
 */
uint32_t
_convert_to_host_byte_order (uint32_t in)
{
  return ntohl (in);
}

/**
 * @brief Get NTP timestamps from the given or public NTP servers
 * @param[in] hnums A number of hostname and port pairs. If 0 is given,
 *                  the NTP server pool will be used.
 * @param[in] hnames A list of hostname
 * @param[in] ports A list of port
 * @return an Unix epoch time as microseconds on success,
 *         negative values on error
 */
int64_t
ntputil_get_epoch (uint32_t hnums, char **hnames, uint16_t * ports)
{
  struct sockaddr_in serv_addr;
  struct hostent *srv = NULL;
  struct hostent *default_srv = NULL;
  uint16_t port = -1;
  int32_t sockfd = -1;
  uint32_t i;
  int64_t ret;

  sockfd = socket (AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sockfd < 0) {
    ret = -1;
    goto ret_normal;
  }

  for (i = 0; i < hnums; ++i) {
    srv = gethostbyname (hnames[i]);
    if (srv != NULL) {
      port = ports[i];
      break;
    }
  }

  if (srv == NULL) {
    default_srv = gethostbyname (NTPUTIL_DEFAULT_HNAME);
    if (default_srv == NULL) {
      ret = -h_errno;
      goto ret_close_sockfd;
    }
    srv = default_srv;
    port = NTPUTIL_DEFAULT_PORT;
  }

  memset (&serv_addr, 0, sizeof (serv_addr));
  serv_addr.sin_family = AF_INET;
  memcpy ((uint8_t *) & serv_addr.sin_addr.s_addr,
      (uint8_t *) srv->h_addr_list[0], (size_t) srv->h_length);
  serv_addr.sin_port = htons (port);

  ret = connect (sockfd, (struct sockaddr *) &serv_addr, sizeof (serv_addr));
  if (ret < 0) {
    ret = -errno;
    goto ret_close_sockfd;
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
      goto ret_close_sockfd;
    }

    /* Recieve */
    n = read (sockfd, &packet, sizeof (packet));
    if (n < 0) {
      ret = -errno;
      goto ret_close_sockfd;
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
      goto ret_close_sockfd;
    }

    ret = (int64_t) (recv_sec - NTPUTIL_TIMESTAMP_DELTA);
    ret *= NTPUTIL_SEC_TO_USEC_MULTIPLIER;
    frac = ((double) recv_frac) / NTPUTIL_MAX_FRAC_DOUBLE;
    frac *= NTPUTIL_SEC_TO_USEC_MULTIPLIER;

    ret += (int64_t) frac;
  }

ret_close_sockfd:
  close (sockfd);

ret_normal:
  return ret;
}
