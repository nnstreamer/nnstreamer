/**
 * @file NCSDKTensorFilterTestHelper.hh
 * @date 8 Jan 2020
 * @author  Wook Song <wook16.song@samsung.com>
 * @brief Helper class for testing tensor_filter_mvncsdk2 without an actual device
 * @see https://github.com/nnstreamer/nnstreamer
 * @bug	No known bugs except for NYI items
 *
 *  Copyright 2020 Samsung Electronics
 *
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <mvnc2/mvnc.h>
#include <string.h>

#include <mutex>

enum _constants {
  TENSOR_RANK_LIMIT = 4,
  SUPPORT_MAX_NUMS_DEVICES = 8,
};

/* Dimension information of Google LeNet */
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
  HOTFIX = 2,
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

/**
 * @brief A helper class for testing the NCSDK tensor filter.
 */
class NCSDKTensorFilterTestHelper
{
  public:
  /**
   * @brief Make this class as a singletone
   */
  static NCSDKTensorFilterTestHelper &getInstance ()
  {
    call_once (NCSDKTensorFilterTestHelper::mOnceFlag,
        [] () { mInstance.reset (new NCSDKTensorFilterTestHelper); });
    return *(mInstance.get ());
  }
  /** @brief Destructor releasing the mocked device and graph resources */
  ~NCSDKTensorFilterTestHelper ();
  /** @brief Set up the fake device, graph and tensor descriptors */
  void init (model_t model);
  /** @brief Free the resources allocated by init () */
  void release ();
  /* Set/Get fail-stage */
  /** @brief Set the NCSDK stage at which the mock should report failure */
  void setFailStage (const fail_stage_t stage);
  /** @brief Get the NCSDK stage at which the mock reports failure */
  fail_stage_t getFailStage ();

  /* Mock methods that simulate NCSDK2 APIs */
  /* Mock Global APIs */
  /** @brief Mock of ncGlobalGetOption (); returns the NCSDK version */
  ncStatus_t ncGlobalGetOption (int option, void *data, unsigned int *dataLength);
  /* Mock Device APIs */
  /** @brief Mock of ncDeviceCreate (); hands out the fake device handle */
  ncStatus_t ncDeviceCreate (int index, struct ncDeviceHandle_t **deviceHandle);
  /** @brief Mock of ncDeviceOpen (); checks the given device handle */
  ncStatus_t ncDeviceOpen (struct ncDeviceHandle_t *deviceHandle);
  /** @brief Mock of ncDeviceClose (); checks the given device handle */
  ncStatus_t ncDeviceClose (struct ncDeviceHandle_t *deviceHandle);
  /** @brief Mock of ncDeviceDestroy (); frees the fake device handle */
  ncStatus_t ncDeviceDestroy (struct ncDeviceHandle_t **deviceHandle);

  /* Mock Graph APIs */
  /** @brief Mock of ncGraphCreate (); hands out the fake graph handle */
  ncStatus_t ncGraphCreate (const char *name, struct ncGraphHandle_t **graphHandle);
  /** @brief Mock of ncGraphAllocate (); records the given graph buffer */
  ncStatus_t ncGraphAllocate (struct ncDeviceHandle_t *deviceHandle,
      struct ncGraphHandle_t *graphHandle, const void *graphBuffer,
      unsigned int graphBufferLength);
  /** @brief Mock of ncGraphGetOption (); returns a tensor descriptor */
  ncStatus_t ncGraphGetOption (struct ncGraphHandle_t *graphHandle, int option,
      void *data, unsigned int *dataLength);
  /** @brief Mock of ncGraphQueueInference (); does nothing */
  ncStatus_t ncGraphQueueInference (struct ncGraphHandle_t *graphHandle,
      struct ncFifoHandle_t **fifoIn, unsigned int inFifoCount,
      struct ncFifoHandle_t **fifoOut, unsigned int outFifoCount);
  /** @brief Mock of ncGraphDestroy (); frees the fake graph handle */
  ncStatus_t ncGraphDestroy (struct ncGraphHandle_t **graphHandle);

  /* Mock FIFO APIs (returning only NC_OK) */
  /** @brief Mock of ncFifoCreate (); fails only at the matching fail stage */
  ncStatus_t ncFifoCreate (
      const char *name, ncFifoType_t type, struct ncFifoHandle_t **fifoHandle);
  /** @brief Mock of ncFifoAllocate (); fails only at the matching fail stage */
  ncStatus_t ncFifoAllocate (struct ncFifoHandle_t *fifoHandle,
      struct ncDeviceHandle_t *device, struct ncTensorDescriptor_t *tensorDesc,
      unsigned int numElem);
  /** @brief Mock of ncFifoSetOption (); always returns NC_OK */
  ncStatus_t ncFifoSetOption (struct ncFifoHandle_t *fifoHandle, int option,
      const void *data, unsigned int dataLength);
  /** @brief Mock of ncFifoGetOption (); always returns NC_OK */
  ncStatus_t ncFifoGetOption (struct ncFifoHandle_t *fifoHandle, int option,
      void *data, unsigned int *dataLength);
  /** @brief Mock of ncFifoDestroy (); always returns NC_OK */
  ncStatus_t ncFifoDestroy (struct ncFifoHandle_t **fifoHandle);
  /** @brief Mock of ncFifoWriteElem (); drops the given input tensor */
  ncStatus_t ncFifoWriteElem (struct ncFifoHandle_t *fifoHandle,
      const void *inputTensor, unsigned int *inputTensorLength, void *userParam);
  /** @brief Mock of ncFifoReadElem (); leaves the output buffer untouched */
  ncStatus_t ncFifoReadElem (struct ncFifoHandle_t *fifoHandle,
      void *outputData, unsigned int *outputDataLen, void **userParam);
  /** @brief Mock of ncFifoRemoveElem (); fails at the matching fail stage */
  ncStatus_t ncFifoRemoveElem (struct ncFifoHandle_t *fifoHandle); /* not supported yet */

  private:
  /* Variables for instance management */
  static std::unique_ptr<NCSDKTensorFilterTestHelper> mInstance;
  static std::once_flag mOnceFlag;

  /* Constructor and destructor */
  /** @brief Default constructor; init () must be invoked before use */
  NCSDKTensorFilterTestHelper ();
  /** @brief Disable the copy constructor to keep this class a singleton */
  NCSDKTensorFilterTestHelper (const NCSDKTensorFilterTestHelper &) = delete;
  NCSDKTensorFilterTestHelper &operator= (const NCSDKTensorFilterTestHelper &) = delete;

  struct ncDeviceHandle_t *mDevHandle;
  struct ncGraphHandle_t *mGraphHandle;
  struct ncTensorDescriptor_t *mTensorDescInput;
  struct ncTensorDescriptor_t *mTensorDescOutput;
  const void *mGraphBuf;
  uint32_t mLenGraphBuf;
  ncsdk_ver_t mVer;
  fail_stage_t mFailStage;
  gchar *mModelPath;
  model_t mModel;
};
