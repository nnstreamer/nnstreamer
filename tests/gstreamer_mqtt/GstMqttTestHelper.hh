/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file GstMqttTestHelper.hh
 * @date 28 May 2021
 * @author  Wook Song <wook16.song@samsung.com>
 * @brief Helper class for testing mqttsink and mqttsrc without the real MQTT broker
 * @see https://github.com/nnstreamer/nnstreamer
 * @bug	No known bugs except for NYI items
 *
 *  Copyright 2021 Samsung Electronics
 *
 */

#include <MQTTAsync.h>

#include <glib.h>
#include <mutex>
#include <memory>

/**
 * @brief A helper class for testing the GstMQTT elements
 */
class GstMqttTestHelper
{
public:
  /**
   * @brief Make this class as a singletone
   */
  static GstMqttTestHelper &getInstance () {
    call_once (GstMqttTestHelper::mOnceFlag, []() {
      mInstance.reset(new GstMqttTestHelper);
    });
    return *(mInstance.get ());
  }

  /**
   * @brief An empty destructor for this class
   */
  ~GstMqttTestHelper () {};

  /**
   * @brief Initialize this class instead of explcit constuctors
   */
  void init (void *ctx) {
    this->context = ctx;
    this->is_connected = false;

    this->cl = nullptr;
    this->ma = nullptr;
    this->dc = nullptr;
  }

  /**
   * @brief Disable all flags that make specific APIs fail
   */
  void initFailFlags () {
    this->fail_send = false;
    this->fail_disconnect = false;
    this->fail_subscribe = false;
    this->fail_unsubscribe = false;
  }

  /**
   * @brief Set callbacks (a wrapper of MQTTAsync_setCallbacks())
   */
  void setCallbacks (MQTTAsync_connectionLost * cl,
      MQTTAsync_messageArrived * ma,
      MQTTAsync_deliveryComplete * dc) {
    this->cl = cl;
    this->ma = ma;
    this->dc = dc;
  }

  /**
   * @brief Setter for fail_send (if it is true, MQTTAsync_send() will be failed)
   */
  void setFailSend (bool flag) {
    this->fail_send = flag;
  }

  /**
   * @brief Setter for fail_disconnect (if it is true, MQTTAsync_disconnect() will be failed)
   */
  void setFailDisconnect (bool flag) {
    this->fail_disconnect = flag;
  }

  /**
   * @brief Setter for fail_subscribe
   */
  void setFailSubscribe (bool flag) {
    this->fail_subscribe = flag;
  }

  /**
   * @brief Setter for fail_unsubscribe
   */
  void setFailUnsubscribe (bool flag) {
    this->fail_subscribe = flag;
  }

  /**
   * @brief Setter for is_connected which is used by MQTTAsync_isConnected()
   */
  void setIsConnected (bool flag) {
    this->is_connected = flag;
  }

  /**
   * @brief Getter for the context pointer
   */
  void *getContext () {
    return this->context;
  }

  /**
   * @brief Getter for fail_send
   */
  bool getFailSend () {
    return this->fail_send;
  }

  /**
   * @brief Getter for fail_disconnect
   */
  bool getFailDisconnect () {
    return this->fail_disconnect;
  }

  /**
   * @brief Getter for fail_subscribe
   */
  bool getFailSubscribe () {
    return this->fail_subscribe;
  }

  /**
   * @brief Getter for fail_unsubscribe
   */
  bool getFailUnsubscribe () {
    return this->fail_unsubscribe;
  }

  /**
   * @brief Getter for is_connected
   */
  bool getIsConnected () {
    return this->is_connected;
  }

  /**
   * @brief Getter for the registered MQTTAsync_messageArrived callback
   */
  MQTTAsync_messageArrived *getCbMessageArrived() {
    return this->ma;
  }

private:
  /* Variables for instance mangement */
  static std::unique_ptr<GstMqttTestHelper> mInstance;
  static std::once_flag mOnceFlag;

  /* Constructor and destructor */
  /**
   * @brief Default Constructor
   */
  GstMqttTestHelper ():
      context (nullptr), cl (nullptr), ma (nullptr), dc (nullptr),
      fail_send (false), fail_disconnect (false), fail_subscribe (false),
      fail_unsubscribe (false), is_connected (false) {};

  GstMqttTestHelper (const GstMqttTestHelper &) = delete;
  GstMqttTestHelper &operator=(const GstMqttTestHelper &) = delete;

  void *context;
  MQTTAsync_connectionLost *cl;
  MQTTAsync_messageArrived *ma;
  MQTTAsync_deliveryComplete * dc;

  bool fail_send;
  bool fail_disconnect;
  bool fail_subscribe;
  bool fail_unsubscribe;
  bool is_connected;
};
