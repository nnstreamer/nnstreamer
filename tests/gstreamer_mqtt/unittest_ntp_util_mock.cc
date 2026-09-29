/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_ntp_util_mock.cc
 * @date        25 Apr 2022
 * @brief       Unit test for ntp util using GMock.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gmock/gmock.h>
#include <unittest_util.h>
#include "../gst/mqtt/ntputil.h"
#include "ntputil.h"

#include <errno.h>
#include <netdb.h>

#include <chrono>
#include <future>
#include <thread>

using ::testing::_;
using ::testing::Assign;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::SetErrnoAndReturn;

const uint64_t NTPUTIL_TIMESTAMP_DELTA = 2208988800ULL;
const ssize_t NTP_PACKET_SIZE = 48;
const int64_t NTP_RETRY_HOLD_OFF_USEC = 10 * 1000000;

/** The clock the retry hold-off of ntp util reads in this test */
static int64_t fake_monotonic_us = 0;

/**
 * @brief Interface for NTP util mock class
 */
class INtpUtil
{
  public:
  /**
   * @brief Destroy the INtpUtil object
   */
  virtual ~INtpUtil (){};
  virtual struct hostent *gethostbyname (const char *name) = 0;
  virtual int connect (int sockfd, const struct sockaddr *addr, socklen_t addrlen) = 0;
  virtual ssize_t write (int fd, const void *buf, size_t count) = 0;
  virtual ssize_t read (int fd, void *buf, size_t count) = 0;
  virtual uint32_t _convert_to_host_byte_order (uint32_t netlong) = 0;
};

/**
 * @brief Mock class for testing ntp util
 */
class NtpUtilMock : public INtpUtil
{
  public:
  MOCK_METHOD (struct hostent *, gethostbyname, (const char *name));
  MOCK_METHOD (int, connect, (int sockfd, const struct sockaddr *addr, socklen_t addrlen));
  MOCK_METHOD (ssize_t, write, (int fd, const void *buf, size_t count));
  MOCK_METHOD (ssize_t, read, (int fd, void *buf, size_t count));
  MOCK_METHOD (uint32_t, _convert_to_host_byte_order, (uint32_t netlong));
};
NtpUtilMock *mockInstance = nullptr;

/**
 * @brief Mocking function for gethostbyname
 */
struct hostent *
gethostbyname (const char *name)
{
  return mockInstance->gethostbyname (name);
}

/**
 * @brief Mocking function for gethostbyname
 */
int
connect (int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
  return mockInstance->connect (sockfd, addr, addrlen);
}

/** @note To avoid redundant declaration in the test */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
/**
 * @brief Mocking function for write.
 */
ssize_t
write (int fd, const void *buf, size_t count)
{
  return mockInstance->write (fd, buf, count);
}

/**
 * @brief Mocking function for read.
 */
ssize_t
read (int fd, void *buf, size_t count)
{
  return mockInstance->read (fd, buf, count);
}
#pragma GCC diagnostic pop

/**
 * @brief Mocking function for _convert_to_host_byte_order.
 */
uint32_t
_convert_to_host_byte_order (uint32_t netlong)
{
  return mockInstance->_convert_to_host_byte_order (netlong);
}

/**
 * @brief Fake clock for _get_monotonic_time_us.
 */
int64_t
_get_monotonic_time_us (void)
{
  return fake_monotonic_us;
}


/**
 * @brief  ntp util testing base class
 */
class ntpUtilMockTest : public ::testing::Test
{
  protected:
  struct hostent host;
  /**
   * @brief  Sets up the base fixture
   */
  void SetUp () override
  {
    /* Let any retry hold-off left by an earlier test expire */
    fake_monotonic_us += 3600LL * 1000000LL;

    host.h_name = g_strdup ("github.com");
    host.h_aliases = g_new0 (gchar *, 1);
    host.h_aliases[0] = g_strdup ("www.github.com");
    host.h_addrtype = AF_INET;
    host.h_length = 4;
    host.h_addr_list = g_new0 (gchar *, 1);
    host.h_addr_list[0] = g_strdup ("52.78.231.108");
  }
  /**
   * @brief tear down the base fixture
   */
  void TearDown () override
  {
    g_free (host.h_name);
    g_free (host.h_aliases[0]);
    g_free (host.h_aliases);
    g_free (host.h_addr_list[0]);
    g_free (host.h_addr_list);
  }
};

/**
 * @brief Test for ntp util to get epoch.
 */
TEST_F (ntpUtilMockTest, getEpochNormal_p)
{
  int64_t ret;
  const char *hnames[] = { "temp" };
  uint16_t ports[] = { 8080U };

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (1).WillOnce (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (1).WillOnce (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (1).WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (1).WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .Times (2)
      .WillOnce (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL))
      .WillOnce (Return (1ULL));

  ret = ntputil_get_epoch (1, (char **) hnames, ports);

  EXPECT_GE (ret, 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to get epoch when failed to get host name.
 */
TEST_F (ntpUtilMockTest, getEpochHostNameFail_n)
{
  int64_t ret;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_))
      .Times (1)
      .WillOnce (DoAll (testing::Assign (&h_errno, HOST_NOT_FOUND), Return (nullptr)));

  ret = ntputil_get_epoch (0, nullptr, nullptr);
  EXPECT_LT (ret, 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to get epoch when failed to connect.
 */
TEST_F (ntpUtilMockTest, getEpochConnectFail_n)
{
  int64_t ret;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (1).WillOnce (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (1).WillOnce (SetErrnoAndReturn (EINVAL, -1));

  ret = ntputil_get_epoch (0, nullptr, nullptr);
  EXPECT_LT (ret, 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to get epoch when failed to write.
 */
TEST_F (ntpUtilMockTest, getEpochWriteFail_n)
{
  int64_t ret;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (1).WillOnce (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (1).WillOnce (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (1).WillOnce (SetErrnoAndReturn (EINVAL, -1));

  ret = ntputil_get_epoch (0, nullptr, nullptr);

  EXPECT_LT (ret, 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to get epoch failed to read.
 */
TEST_F (ntpUtilMockTest, getEpochReadFail_n)
{
  int64_t ret;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (1).WillOnce (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (1).WillOnce (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (1).WillOnce (Return (0));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (1).WillOnce (SetErrnoAndReturn (EINVAL, -1));

  ret = ntputil_get_epoch (0, nullptr, nullptr);

  EXPECT_LT (ret, 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to get epoch.
 */
TEST_F (ntpUtilMockTest, getEpochInvalidTimestamp)
{
  int64_t ret;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (1).WillOnce (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (1).WillOnce (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (1).WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (1).WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .Times (2)
      .WillOnce (Return (1ULL))
      .WillOnce (Return (1ULL));

  ret = ntputil_get_epoch (0, nullptr, nullptr);

  EXPECT_LT (ret, 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to refuse a reply shorter than an NTP packet.
 */
TEST_F (ntpUtilMockTest, getEpochShortRead_n)
{
  int64_t ret;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (1).WillOnce (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (1).WillOnce (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (1).WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (1).WillOnce (Return (NTP_PACKET_SIZE - 1));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_)).Times (0);

  ret = ntputil_get_epoch (0, nullptr, nullptr);

  EXPECT_LT (ret, 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to report a resolver-internal error as an error.
 */
TEST_F (ntpUtilMockTest, getEpochResolverInternalError_n)
{
  int64_t ret;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_))
      .Times (1)
      .WillOnce (DoAll (testing::Assign (&h_errno, NETDB_INTERNAL), Return (nullptr)));

  ret = ntputil_get_epoch (0, nullptr, nullptr);
  EXPECT_LT (ret, 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to resolve a healthy server only once.
 */
TEST_F (ntpUtilMockTest, getEpochResolvedOnce)
{
  const char *hnames[] = { "once" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (1).WillOnce (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (2).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (2).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (2).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to resolve the server again, without a hold-off, after one query to it failed.
 */
TEST_F (ntpUtilMockTest, getEpochResolvedAgainAfterFailure)
{
  const char *hnames[] = { "again" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (2).WillRepeatedly (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (3).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (3).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _))
      .Times (3)
      .WillOnce (Return (NTP_PACKET_SIZE))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to resolve again when the server list changes.
 */
TEST_F (ntpUtilMockTest, getEpochResolvedForNewServers)
{
  const char *hnames1[] = { "first" };
  const char *hnames2[] = { "second" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (2).WillRepeatedly (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (2).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (2).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (2).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames1, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames2, ports), 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util not to query hosts again right after they failed three times in a row.
 */
TEST_F (ntpUtilMockTest, getEpochHeldOffAfterFailures_n)
{
  const char *hnames[] = { "held" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (4).WillRepeatedly (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (4).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (4).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _))
      .Times (4)
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  fake_monotonic_us += NTP_RETRY_HOLD_OFF_USEC - 1;
  EXPECT_EQ (ntputil_get_epoch (1, (char **) hnames, ports), -EAGAIN);

  fake_monotonic_us += 1;
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to hold hosts off again when the first query after a hold-off fails.
 */
TEST_F (ntpUtilMockTest, getEpochHeldOffAgainAfterRetryFails_n)
{
  const char *hnames[] = { "still-down" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (4).WillRepeatedly (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (4).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (4).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (4).WillRepeatedly (SetErrnoAndReturn (EAGAIN, -1));

  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  fake_monotonic_us += NTP_RETRY_HOLD_OFF_USEC;
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_EQ (ntputil_get_epoch (1, (char **) hnames, ports), -EAGAIN);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to keep querying hosts whose failures are not in a row.
 */
TEST_F (ntpUtilMockTest, getEpochFailureCountResetOnSuccess)
{
  const char *hnames[] = { "flaky" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (5).WillRepeatedly (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (6).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (6).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _))
      .Times (6)
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (Return (NTP_PACKET_SIZE))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util not to resolve hosts again right after they failed to resolve three times.
 */
TEST_F (ntpUtilMockTest, getEpochHeldOffAfterResolveFailures_n)
{
  const char *hnames[] = { "unresolvable" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  /* the host and the default pool, three times */
  EXPECT_CALL (*mockInstance, gethostbyname (_))
      .Times (6)
      .WillRepeatedly (
          DoAll (testing::Assign (&h_errno, HOST_NOT_FOUND), Return (nullptr)));

  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_EQ (ntputil_get_epoch (1, (char **) hnames, ports), -EAGAIN);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to query a new server list at once while the
 * previous one is held off, without ending that hold-off.
 */
TEST_F (ntpUtilMockTest, getEpochNewServersNotHeldOff)
{
  const char *hnames1[] = { "failing" };
  const char *hnames2[] = { "replacement" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (4).WillRepeatedly (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (4).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (4).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _))
      .Times (4)
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames1, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames1, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames1, ports), 0);
  EXPECT_EQ (ntputil_get_epoch (1, (char **) hnames1, ports), -EAGAIN);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames2, ports), 0);
  EXPECT_EQ (ntputil_get_epoch (1, (char **) hnames1, ports), -EAGAIN);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to end a failure streak with a query that succeeds while a concurrent
 *        query to the same hosts fails and invalidates their address.
 */
TEST_F (ntpUtilMockTest, getEpochOverlappingSuccessEndsStreak)
{
  const char *hnames[] = { "overlap" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  /* The nested query reuses the address the outer one resolved */
  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (5).WillRepeatedly (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (6).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (6).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _))
      .Times (6)
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (Invoke ([&] (int fd, void *buf, size_t count) -> ssize_t {
        (void) fd;
        (void) buf;
        (void) count;
        /* Another caller fails while this query waits for its reply */
        EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
        return NTP_PACKET_SIZE;
      }))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  /* Two more failures do not reach a streak of three */
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to keep counting the failures of hosts queried in turn with other hosts.
 */
TEST_F (ntpUtilMockTest, getEpochAlternatingServersHeldOff_n)
{
  const char *hnames1[] = { "alternating-down" };
  const char *hnames2[] = { "alternating-up" };
  uint16_t ports[] = { 123U };

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (4).WillRepeatedly (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (5).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (5).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _))
      .Times (5)
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (Return (NTP_PACKET_SIZE))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1))
      .WillOnce (Return (NTP_PACKET_SIZE))
      .WillOnce (SetErrnoAndReturn (EAGAIN, -1));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames1, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames2, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames1, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames2, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames1, ports), 0);
  EXPECT_EQ (ntputil_get_epoch (1, (char **) hnames1, ports), -EAGAIN);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to replace the least recently used of more host
 * lists than it caches, counting a cached query as a use.
 */
TEST_F (ntpUtilMockTest, getEpochCacheEvictsLeastRecentlyUsed)
{
  const int num_lists = 9;
  char names[9][16];
  char *hnames[9][1];
  uint16_t ports[] = { 123U };
  int i;

  for (i = 0; i < num_lists; i++) {
    snprintf (names[i], sizeof (names[i]), "lru-%d", i);
    hnames[i][0] = names[i];
  }

  mockInstance = new NtpUtilMock ();

  /* lists 0-8 once each, then list 1 again */
  EXPECT_CALL (*mockInstance, gethostbyname (_))
      .Times (num_lists + 1)
      .WillRepeatedly (Return ((struct hostent *) &host));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (num_lists + 3).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (num_lists + 3).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (num_lists + 3).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  for (i = 0; i < num_lists - 1; i++)
    EXPECT_GE (ntputil_get_epoch (1, hnames[i], ports), 0);

  /* A cached query makes list 0 recently used, so list 8 replaces list 1 */
  EXPECT_GE (ntputil_get_epoch (1, hnames[0], ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, hnames[num_lists - 1], ports), 0);

  EXPECT_GE (ntputil_get_epoch (1, hnames[0], ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, hnames[1], ports), 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to answer a query to cached hosts while another
 * host list waits for its DNS lookup.
 */
TEST_F (ntpUtilMockTest, getEpochCachedQueryNotBlockedByLookup)
{
  const char *hnames_cached[] = { "cached" };
  const char *hnames_slow[] = { "slow-dns" };
  uint16_t ports[] = { 123U };
  std::future<int64_t> cached_query;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_))
      .Times (2)
      .WillOnce (Return ((struct hostent *) &host))
      .WillOnce (Invoke ([&](const char *name) -> struct hostent * {
        (void) name;
        cached_query = std::async (std::launch::async, [&] () {
          return ntputil_get_epoch (1, (char **) hnames_cached, ports);
        });
        /* The cached query must not wait for this lookup to return */
        EXPECT_EQ (cached_query.wait_for (std::chrono::seconds (2)),
            std::future_status::ready);
        return (struct hostent *) &host;
      }));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (3).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (3).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (3).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames_cached, ports), 0);
  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames_slow, ports), 0);

  ASSERT_TRUE (cached_query.valid ());
  EXPECT_GE (cached_query.get (), 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to resolve hosts once when a second caller waits
 * for the lookup of the same hosts.
 */
TEST_F (ntpUtilMockTest, getEpochConcurrentLookupResolvedOnce)
{
  const char *hnames[] = { "shared-lookup" };
  uint16_t ports[] = { 123U };
  std::future<int64_t> second_query;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname (_)).Times (1).WillOnce (Invoke ([&](const char *name) -> struct hostent * {
    (void) name;
    second_query = std::async (std::launch::async,
        [&] () { return ntputil_get_epoch (1, (char **) hnames, ports); });
    /* Let the second caller reach the lookup and wait for this one */
    std::this_thread::sleep_for (std::chrono::milliseconds (300));
    return (struct hostent *) &host;
  }));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (2).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (2).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (2).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order (_))
      .WillRepeatedly (Return (NTPUTIL_TIMESTAMP_DELTA + 1ULL));

  EXPECT_GE (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  ASSERT_TRUE (second_query.valid ());
  EXPECT_GE (second_query.get (), 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to make a caller that waited for a failing lookup
 * of the same hosts follow the hold-off that lookup started.
 */
TEST_F (ntpUtilMockTest, getEpochWaiterFollowsHoldOffFromLookup_n)
{
  const char *hnames[] = { "held-during-lookup" };
  uint16_t ports[] = { 123U };
  std::future<int64_t> second_query;

  mockInstance = new NtpUtilMock ();

  /* Two resolved queries that fail, then a lookup that fails for the host and the pool */
  EXPECT_CALL (*mockInstance, gethostbyname (_))
      .Times (4)
      .WillOnce (Return ((struct hostent *) &host))
      .WillOnce (Return ((struct hostent *) &host))
      .WillOnce (Invoke ([&](const char *name) -> struct hostent * {
        (void) name;
        second_query = std::async (std::launch::async,
            [&] () { return ntputil_get_epoch (1, (char **) hnames, ports); });
        /* Let the second caller reach the lookup and wait for this one */
        std::this_thread::sleep_for (std::chrono::milliseconds (300));
        h_errno = HOST_NOT_FOUND;
        return nullptr;
      }))
      .WillOnce (DoAll (testing::Assign (&h_errno, HOST_NOT_FOUND), Return (nullptr)));
  EXPECT_CALL (*mockInstance, connect (_, _, _)).Times (2).WillRepeatedly (Return (0));
  EXPECT_CALL (*mockInstance, write (_, _, _)).Times (2).WillRepeatedly (Return (NTP_PACKET_SIZE));
  EXPECT_CALL (*mockInstance, read (_, _, _)).Times (2).WillRepeatedly (SetErrnoAndReturn (EAGAIN, -1));

  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);
  EXPECT_LT (ntputil_get_epoch (1, (char **) hnames, ports), 0);

  ASSERT_TRUE (second_query.valid ());
  EXPECT_EQ (second_query.get (), -EAGAIN);

  delete mockInstance;
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
