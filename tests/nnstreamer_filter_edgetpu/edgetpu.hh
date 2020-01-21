/**
 * @file        edgetpu.hh
 * @date        16 Dec 2019
 * @brief       Dummy implementation of tflite and edgetpu for unit tests.
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @brief	This header provides a dummy edgetpu library that redirects
 *		requests to CPU backends.
 */

#include <tensorflow/lite/kernels/register.h>

#define kTfLiteEdgeTpuContext kTfLiteEigenContext

namespace edgetpu {

/** @brief Dummy, making this equivalent to original edgetpu-runtime (2019-12) */
enum class DeviceType {
  kApexPci = 0,
  kApexUsb = 1,
};
class EdgeTpuContext;

/** @brief Dummy, making this equivalent to original edgetpu-runtime (2019-12) */
class EdgeTpuManager {
public:
  using DeviceOptions = std::unordered_map<std::string, std::string>;

  /** @brief Dummy, making this equivalent to original edgetpu-runtime (2019-12) */
  struct DeviceEnumerationRecord {
    DeviceType type;
    std::string path;
    friend bool operator==(const DeviceEnumerationRecord& lhs,
                           const DeviceEnumerationRecord& rhs) {
      return (lhs.type == rhs.type) && (lhs.path == rhs.path);
    }
    friend bool operator!=(const DeviceEnumerationRecord& lhs,
                           const DeviceEnumerationRecord& rhs) {
      return !(lhs == rhs);
    }
  };

  /** @brief Dummy singleton methos */
  static EdgeTpuManager * GetSingleton();

  /** @brief To make the binary format equivalent? Keep it dummy */
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext() = 0;
  /** @brief To make the binary format equivalent? Keep it dummy */
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type) = 0;
  /** @brief To make the binary format equivalent? Keep it dummy */
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type, const std::string& device_path) = 0;
  /** @brief To make the binary format equivalent? Keep it dummy */
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type, const std::string& device_path,
      const DeviceOptions& options) = 0;
  /** @brief To make the binary format equivalent? Keep it dummy */
  virtual std::vector<DeviceEnumerationRecord> EnumerateEdgeTpu() const = 0;

  /** @brief Dummy device open */
  virtual std::shared_ptr<EdgeTpuContext> OpenDevice() = 0;

  /** @brief Dummy destructor */
  virtual ~EdgeTpuManager() = default;
};

/** @brief Dummy, making this equivalent to original edgetpu-runtime (2019-12) */
class EdgeTpuContext : public TfLiteExternalContext {
public:
  /** @brief Dummy API */
  virtual ~EdgeTpuContext() = 0;

  /** @brief Dummy API */
  virtual const EdgeTpuManager::DeviceEnumerationRecord& GetDeviceEnumRecord()
      const = 0;

  /** @brief Dummy API */
  virtual EdgeTpuManager::DeviceOptions GetDeviceOptions() const = 0;

  /** @brief Dummy API */
  virtual bool IsReady() const = 0;
  /** @brief Dummy API */
  virtual EdgeTpuContext *get() = 0;
};

/** @brief Dummy, making this equivalent to original edgetpu-runtime (2019-12) */
class EdgeTpuManagerDummy: public EdgeTpuManager {
  std::shared_ptr<EdgeTpuContext> OpenDevice();

  /** @brief To make the binary format equivalent? Keep it dummy */
  std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext() { return nullptr; }
  /** @brief To make the binary format equivalent? Keep it dummy */
  std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type) { return nullptr; }
  /** @brief To make the binary format equivalent? Keep it dummy */
  std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type, const std::string& device_path) { return nullptr; }
  /** @brief To make the binary format equivalent? Keep it dummy */
  std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type, const std::string& device_path,
      const DeviceOptions& options) { return nullptr; }
  /** @brief To make the binary format equivalent? Keep it dummy */
  std::vector<DeviceEnumerationRecord> EnumerateEdgeTpu() const { return std::vector<DeviceEnumerationRecord>(0); }
};

static const char kCustomOp[] = "dummy-dummy-dummy";

/** @brief Dummy tflite custom op */
TfLiteRegistration * RegisterCustomOp();

static const EdgeTpuManager::DeviceEnumerationRecord dummyRecord =
    { .type = DeviceType::kApexUsb, .path = "Dummy" };

/** @brief Dummy, making this equivalent to original edgetpu-runtime (2019-12) */
class EdgeTpuContextDummy : public EdgeTpuContext {
public:
  /** @brief Dummy API */
  ~EdgeTpuContextDummy() { }

  /** @brief Dummy API */
  const EdgeTpuManager::DeviceEnumerationRecord& GetDeviceEnumRecord()
      const { return dummyRecord; }

  /** @brief Dummy API */
  EdgeTpuManager::DeviceOptions GetDeviceOptions() const
  { return std::unordered_map<std::string, std::string>(); }

  /** @brief Dummy API */
  bool IsReady() const { return true; }

  /** @brief Dummy edgetpu get function */
  EdgeTpuContext *get();
};

};
