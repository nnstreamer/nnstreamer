/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_ntp_util_mock.cc
 * @date        25 Apr 2022
 * @brief       Unit test for ntp util using GMock.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <glib.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <unittest_util.h>
#include "../gst/mqtt/ntputil.h"
#include "ntputil.h"

#include <errno.h>
#include <netdb.h>

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::Assign;
using ::testing::SetErrnoAndReturn;

const uint64_t NTPUTIL_TIMESTAMP_DELTA = 2208988800ULL;

/**
 * @brief Interface for NTP util mock class
 */
class INtpUtil {
public:
  /**
   * @brief Destroy the INtpUtil object
   */
  virtual ~INtpUtil () {};
  virtual struct hostent *gethostbyname (const char *name) = 0;
  virtual int connect (int sockfd, const struct sockaddr *addr,
                  socklen_t addrlen) = 0;
  virtual ssize_t write (int fd, const void *buf, size_t count) = 0;
  virtual ssize_t read (int fd, void *buf, size_t count) = 0;
  virtual uint32_t  _convert_to_host_byte_order (uint32_t netlong) = 0;
};

/**
 * @brief Mock class for testing ntp util
 */
class NtpUtilMock : public INtpUtil {
public:
  MOCK_METHOD (struct hostent *, gethostbyname, (const char *name));
  MOCK_METHOD (int, connect, (int sockfd, const struct sockaddr *addr,
                  socklen_t addrlen));
  MOCK_METHOD (ssize_t, write, (int fd, const void *buf, size_t count));
  MOCK_METHOD (ssize_t, read, (int fd, void *buf, size_t count));
  MOCK_METHOD (uint32_t, _convert_to_host_byte_order, (uint32_t netlong));
};
NtpUtilMock *mockInstance = nullptr;

/**
 * @brief Mocking function for gethostbyname
 */
struct hostent *gethostbyname (const char *name)
{
  return mockInstance->gethostbyname (name);
}

/**
 * @brief Mocking function for gethostbyname
 */
int connect (int sockfd, const struct sockaddr *addr,
                  socklen_t addrlen)
{
  return mockInstance->connect (sockfd, addr, addrlen);
}

/** @note To avoid redundant declaration in the test */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
/**
 * @brief Mocking function for write.
 */
ssize_t write (int fd, const void *buf, size_t count)
{
  return mockInstance->write (fd, buf, count);
}

/**
 * @brief Mocking function for read.
 */
ssize_t read (int fd, void *buf, size_t count)
{
  return mockInstance->read (fd, buf, count);
}
#pragma GCC diagnostic pop

/**
 * @brief Mocking function for _convert_to_host_byte_order.
 */
uint32_t _convert_to_host_byte_order (uint32_t netlong)
{
  return mockInstance->_convert_to_host_byte_order (netlong);
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
    host.h_name = g_strdup ("github.com");
    host.h_aliases = (char **) calloc (1, sizeof (char *));
    host.h_aliases[0] = g_strdup ("www.github.com");
    host.h_addrtype = AF_INET;
    host.h_length = 4;
    host.h_addr_list = (char **) calloc (1, sizeof (char *));
    host.h_addr_list[0] = g_strdup ("52.78.231.108");
  }
  /**
   * @brief tear down the base fixture
   */
  void TearDown ()
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
  const char *hnames[] = {"temp"};
  uint16_t ports[] = {8080U};

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname(_))
      .Times(1).WillOnce(Return((struct hostent *)&host));
  EXPECT_CALL (*mockInstance, connect(_, _, _))
      .Times(1).WillOnce(Return(0));
  EXPECT_CALL (*mockInstance, write(_, _, _))
      .Times(1).WillOnce(Return(0));
  EXPECT_CALL (*mockInstance, read(_, _, _))
      .Times(1).WillOnce(Return(0));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order(_))
      .Times(2)
      .WillOnce(Return(NTPUTIL_TIMESTAMP_DELTA + 1ULL))
      .WillOnce(Return(1ULL));

  ret = ntputil_get_epoch (1, (char **)hnames, ports);

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

  EXPECT_CALL (*mockInstance, gethostbyname(_))
      .Times(1)
      .WillOnce(DoAll(
        testing::Assign(&h_errno, HOST_NOT_FOUND),
        Return (nullptr)));

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

  EXPECT_CALL (*mockInstance, gethostbyname(_))
      .Times(1).WillOnce(Return((struct hostent *)&host));
  EXPECT_CALL (*mockInstance, connect(_, _, _))
      .Times(1).WillOnce(SetErrnoAndReturn(EINVAL, -1));

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

  EXPECT_CALL (*mockInstance, gethostbyname(_))
      .Times(1).WillOnce(Return((struct hostent *)&host));
  EXPECT_CALL (*mockInstance, connect(_, _, _))
      .Times(1).WillOnce(Return(0));
  EXPECT_CALL (*mockInstance, write(_, _, _))
      .Times(1).WillOnce(SetErrnoAndReturn(EINVAL, -1));

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

  EXPECT_CALL (*mockInstance, gethostbyname(_))
      .Times(1).WillOnce(Return((struct hostent *)&host));
  EXPECT_CALL (*mockInstance, connect(_, _, _))
      .Times(1).WillOnce(Return(0));
  EXPECT_CALL (*mockInstance, write(_, _, _))
      .Times(1).WillOnce(Return(0));
  EXPECT_CALL (*mockInstance, read(_, _, _))
      .Times(1).WillOnce(SetErrnoAndReturn(EINVAL, -1));

  ret = ntputil_get_epoch (0, nullptr, nullptr);

  EXPECT_LT (ret, 0);

  delete mockInstance;
}

/**
 * @brief Test for ntp util to get epoch.
 */
TEST_F (ntpUtilMockTest, getEpochIvalidTimestamp)
{
  int64_t ret;

  mockInstance = new NtpUtilMock ();

  EXPECT_CALL (*mockInstance, gethostbyname(_))
      .Times(1).WillOnce(Return((struct hostent *)&host));
  EXPECT_CALL (*mockInstance, connect(_, _, _))
      .Times(1).WillOnce(Return(0));
  EXPECT_CALL (*mockInstance, write(_, _, _))
      .Times(1).WillOnce(Return(0));
  EXPECT_CALL (*mockInstance, read(_, _, _))
      .Times(1).WillOnce(Return(0));
  EXPECT_CALL (*mockInstance, _convert_to_host_byte_order(_))
      .Times(2)
      .WillOnce(Return(1ULL))
      .WillOnce(Return(1ULL));

  ret = ntputil_get_epoch (0, nullptr, nullptr);

  EXPECT_LT (ret, 0);

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
