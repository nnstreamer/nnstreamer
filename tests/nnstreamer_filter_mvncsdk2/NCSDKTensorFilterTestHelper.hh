/**
 * @file NCSDKTensorFilterTestHelper.hh
 * @date 8 Jan 2020
 * @author  Wook Song <wook16.song@samsung.com>
 * @brief Helper class for testing tensor_filter_mvncsdk2 without an actual device
 * @see https://github.com/nnsuite/nnstreamer
 * @bug	No known bugs except for NYI items
 *
 *  Copyright 2020 Samsung Electronics
 *
 */

#include <glib.h>
#include <gtest/gtest.h>
#include <mvnc2/mvnc.h>
#include <string.h>

#include <mutex>

enum _contants {
  TENSOR_RANK_LIMIT = 4,
  SUPPORT_MAX_NUMS_DEVICES = 8,
};

// Dimension inforamtion of Google LeNet
enum _google_lenet {
  GOOGLE_LENET_IN_DIM_C = 3,
  GOOGLE_LENET_IN_DIM_W = 224,
  GOOGLE_LENET_IN_DIM_H = 224,
  GOOGLE_LENET_IN_DIM_N = 1,
  GOOGLE_LENET_OUT_DIM_C = 1000,
  GOOGLE_LENET_OUT_DIM_W = 1,
  GOOGLE_LENET_OUT_DIM_H = 1,
  GOOGLE_LENET_OUT_DIM_N = 1,
};

enum _ncsdk_ver_idx {
  MAJOR = 0,
  MINOR = 1,
  HOTFIX= 2,
  RC = 3,
};

typedef enum _model {
  GOOGLE_LENET = 0,
  DEFAULT_MODEL = GOOGLE_LENET,
} model_t;

typedef enum _fail_stage_t {
  NONE,
  WRONG_SDK_VER,
  FAIL_GLBL_GET_OPT,
  FAIL_DEV_CREATE,
  FAIL_DEV_OPEN,
  FAIL_DEV_CLOSE,
  FAIL_GRAPH_CREATE,
  FAIL_GRAPH_ALLOC,
  FAIL_GRAPH_Q_INFER,
  FAIL_GRAPH_GET_INPUT_TENSOR_DESC,
  FAIL_GRAPH_GET_OUTPUT_TENSOR_DESC,
  FAIL_FIFO_CREATE_INPUT,
  FAIL_FIFO_CREATE_OUTPUT,
  FAIL_FIFO_ALLOC_INPUT,
  FAIL_FIFO_ALLOC_OUTPUT,
  FAIL_FIFO_WRT_ELEM,
  FAIL_FIFO_RD_ELEM,
  FAIL_FIFO_RM_ELEM,
} fail_stage_t;

typedef uint32_t ncsdk_ver_t[NC_VERSION_MAX_SIZE];

class NCSDKTensorFilterTestHelper
{
public:
  // Make this class as a singletone
  static NCSDKTensorFilterTestHelper &getInstance () {
    call_once (NCSDKTensorFilterTestHelper::mOnceFlag, []() {
      mInstance.reset(new NCSDKTensorFilterTestHelper);
    });
    return *(mInstance.get ());
  }
  ~NCSDKTensorFilterTestHelper ();
  void init (model_t model);
  void release ();
  // Set/Get fail-stage
  void setFailStage (const fail_stage_t stage);
  const fail_stage_t getFailStage ();

  // Mock methods that simulate NCSDK2 APIs
  // Mock Global APIs
  ncStatus_t ncGlobalGetOption (int option, void *data,
      unsigned int *dataLength);
  // Mock Device APIs
  ncStatus_t ncDeviceCreate (int index,struct ncDeviceHandle_t **deviceHandle);
  ncStatus_t ncDeviceOpen(struct ncDeviceHandle_t *deviceHandle);
  ncStatus_t ncDeviceClose(struct ncDeviceHandle_t *deviceHandle);
  ncStatus_t ncDeviceDestroy(struct ncDeviceHandle_t **deviceHandle);

  // Mock Graph APIs
  ncStatus_t ncGraphCreate(const char* name, struct ncGraphHandle_t **graphHandle);
  ncStatus_t ncGraphAllocate(struct ncDeviceHandle_t *deviceHandle,
      struct ncGraphHandle_t *graphHandle, const void *graphBuffer,
      unsigned int graphBufferLength);
  ncStatus_t ncGraphGetOption(struct ncGraphHandle_t *graphHandle,
     int option, void *data, unsigned int *dataLength);
  ncStatus_t ncGraphQueueInference(struct ncGraphHandle_t *graphHandle,
      struct ncFifoHandle_t** fifoIn, unsigned int inFifoCount,
      struct ncFifoHandle_t** fifoOut, unsigned int outFifoCount);
  ncStatus_t ncGraphDestroy(struct ncGraphHandle_t **graphHandle);

  // Mock FIFO APIs (returning only NC_OK)
  ncStatus_t ncFifoCreate(const char* name, ncFifoType_t type,
      struct ncFifoHandle_t** fifoHandle);
  ncStatus_t ncFifoAllocate(struct ncFifoHandle_t* fifoHandle,
      struct ncDeviceHandle_t* device, struct ncTensorDescriptor_t* tensorDesc,
      unsigned int numElem);
  ncStatus_t ncFifoSetOption(struct ncFifoHandle_t* fifoHandle, int option,
      const void *data, unsigned int dataLength);
  ncStatus_t ncFifoGetOption(struct ncFifoHandle_t* fifoHandle, int option,
      void *data, unsigned int *dataLength);
  ncStatus_t ncFifoDestroy(struct ncFifoHandle_t** fifoHandle);
  ncStatus_t ncFifoWriteElem(struct ncFifoHandle_t* fifoHandle,
      const void *inputTensor, unsigned int * inputTensorLength,
      void *userParam);
  ncStatus_t ncFifoReadElem(struct ncFifoHandle_t* fifoHandle, void *outputData,
      unsigned int* outputDataLen, void **userParam);
  ncStatus_t ncFifoRemoveElem(struct ncFifoHandle_t* fifoHandle); //not supported yet

private:
  // Variables for instance mangement
  static std::unique_ptr<NCSDKTensorFilterTestHelper> mInstance;
  static std::once_flag mOnceFlag;

  // Constructor and destructor
  NCSDKTensorFilterTestHelper ();
  NCSDKTensorFilterTestHelper (const NCSDKTensorFilterTestHelper &) = delete;
  NCSDKTensorFilterTestHelper &operator=(const NCSDKTensorFilterTestHelper &)
      = delete;

  struct ncDeviceHandle_t * mDevHandle;
  struct ncGraphHandle_t * mGraphHandle;
  struct ncTensorDescriptor_t * mTensorDescInput;
  struct ncTensorDescriptor_t * mTensorDescOutput;
  const void *mGraphBuf;
  uint32_t mLenGraphBuf;
  ncsdk_ver_t mVer;
  fail_stage_t mFailStage;
  gchar *mModelPath;
  model_t mModel;
};
