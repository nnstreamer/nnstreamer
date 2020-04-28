/**
 * @file NCSDKTensorFilterTestHelper.cc
 * @date 8 Jan 2020
 * @author  Wook Song <wook16.song@samsung.com>
 * @brief Helper class for testing tensor_filter_mvncsdk2 without an actual device
 * @see https://github.com/nnstreamer/nnstreamer
 * @bug	No known bugs except for NYI items
 *
 *  Copyright 2020 Samsung Electronics
 *
 */

#include "NCSDKTensorFilterTestHelper.hh"

/* Static member variables for instance management */
std::unique_ptr<NCSDKTensorFilterTestHelper>
    NCSDKTensorFilterTestHelper::mInstance;
std::once_flag NCSDKTensorFilterTestHelper::mOnceFlag;
static const char NNS_MVNCSDK2_NAME_INPUT_FIFO[] = "INPUT_FIFO";
static const char NNS_MVNCSDK2_NAME_OUTPUT_FIFO[] = "OUTPUT_FIFO";

/** @brief Compare two ncTensorDescriptor_t data */
static bool compareTensorDesc (const struct ncTensorDescriptor_t &tensor1,
    const struct ncTensorDescriptor_t &tensor2) {
  if (tensor1.c != tensor2.c)
    return false;
  else if (tensor1.n != tensor2.n)
    return false;
  else if (tensor1.w != tensor2.w)
    return false;
  else if (tensor1.h != tensor2.h)
    return false;
  else if (tensor1.dataType != tensor2.dataType)
    return false;
  /* Do not care other member variables */

  return true;
}

/**
 * @brief Default constuctor. Note that explicit invocation of init () is always
 * required after getting the instance.
 */
NCSDKTensorFilterTestHelper::NCSDKTensorFilterTestHelper ()
{
  /* Do nothing */
}

/**
 * @brief Destructor.Resources are deallocated.
 */
NCSDKTensorFilterTestHelper::~NCSDKTensorFilterTestHelper ()
{
  /* Do nothing */
}

/**
 * @brief A method to initialize this helper class
 * @param[in] model : A neural network model to simulate
 */
void
NCSDKTensorFilterTestHelper::init (model_t model)
{
  /* MAJOR should be 2 */
  this->mVer[MAJOR] = 2;
  /* Don't care other indexes, MINOR, HOTFIX, and RC */
  this->mVer[MINOR] = 3;
  this->mVer[HOTFIX] = 4;
  this->mVer[RC] = 5;

  try {
    this->mDevHandle = new ncDeviceHandle_t;
  } catch (const std::bad_alloc & e) {
    this->mDevHandle = nullptr;
  }

  try {
    this->mGraphHandle = new ncGraphHandle_t;
  } catch (const std::bad_alloc & e) {
    this->mGraphHandle = nullptr;
  }
  this->mModelPath = nullptr;
  this->mFailStage = fail_stage_t::NONE;

  switch (model) {
    default:
    case GOOGLE_LENET:
      this->mModel = model;
      try {
        this->mTensorDescInput = new struct ncTensorDescriptor_t ();
        this->mTensorDescInput->c = GOOGLE_LENET_IN_DIM_C;
        this->mTensorDescInput->n = GOOGLE_LENET_IN_DIM_N;
        this->mTensorDescInput->w = GOOGLE_LENET_IN_DIM_W;
        this->mTensorDescInput->h = GOOGLE_LENET_IN_DIM_H;
        this->mTensorDescInput->dataType = NC_FIFO_FP32;

        this->mTensorDescOutput = new struct ncTensorDescriptor_t ();
        this->mTensorDescOutput->c = GOOGLE_LENET_OUT_DIM_C;
        this->mTensorDescOutput->n = GOOGLE_LENET_OUT_DIM_N;
        this->mTensorDescOutput->w = GOOGLE_LENET_OUT_DIM_W;
        this->mTensorDescOutput->h = GOOGLE_LENET_OUT_DIM_H;
        this->mTensorDescOutput->dataType = NC_FIFO_FP32;
      } catch  (const std::bad_alloc & e) {
        this->mTensorDescInput = nullptr;
        this->mTensorDescOutput = nullptr;
      }
      break;
  }
}

/**
 * @brief A method to release allocated resources
 */
void
NCSDKTensorFilterTestHelper::release ()
{
  this->ncDeviceDestroy (&(this->mDevHandle));
  this->ncGraphDestroy (&(this->mGraphHandle));

  if (this->mTensorDescInput != nullptr) {
    delete this->mTensorDescInput;
    this->mTensorDescInput = nullptr;
  }

  if (this->mTensorDescOutput != nullptr) {
    delete this->mTensorDescOutput;
    this->mTensorDescOutput = nullptr;
  }

  g_free (this->mModelPath);
}

/**
 * @brief Set the stage where the NCSDK fails
 */
void
NCSDKTensorFilterTestHelper::setFailStage (const fail_stage_t stage)
{
  this->mFailStage = stage;
}

/**
 * @brief Get the stage where the NCSDK fails
 */
const fail_stage_t
NCSDKTensorFilterTestHelper::getFailStage ()
{
  return this->mFailStage;
}
/**
 * @brief A method mocking ncGlobalGetOption()
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncGlobalGetOption (int option, void *data,
      unsigned int *dataLength)
{
  if (this->mFailStage == fail_stage_t::WRONG_SDK_VER) {
    ncsdk_ver_t ver;

    /* MAJOR version number should be 2 */
    ver[MAJOR] = 3;
    ver[MINOR] = 4;
    ver[HOTFIX] = 5;
    ver[RC] = 6;
    if (sizeof(ncsdk_ver_t) != (*dataLength))
        return NC_ERROR;

    memcpy (data, ver, *dataLength);

    return NC_OK;
  } else if (this->mFailStage == FAIL_GLBL_GET_OPT) {
    return NC_ERROR;
  }

  switch (option) {
    case NC_RO_API_VERSION:
      if (sizeof(ncsdk_ver_t) != (*dataLength))
        return NC_ERROR;
      memcpy (data, this->mVer, *dataLength);
      break;
    default:
      return NC_ERROR;
  }
  return NC_OK;
}

/**
 * @brief A method mocking ncDeviceCreate()
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncDeviceCreate (int index,
    struct ncDeviceHandle_t **deviceHandle)
{
  if (this->mFailStage == fail_stage_t::FAIL_DEV_CREATE) {
    return NC_ERROR;
  }

  if ((index < 0) || (index >= SUPPORT_MAX_NUMS_DEVICES))
    return NC_INVALID_PARAMETERS;

  if (this->mDevHandle == nullptr)
    return NC_OUT_OF_MEMORY;

  *deviceHandle = this->mDevHandle;

  return NC_OK;
}

/**
 * @brief A method mocking ncDeviceOpen()
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncDeviceOpen(struct ncDeviceHandle_t *deviceHandle)
{
  if (this->mFailStage == fail_stage_t::FAIL_DEV_OPEN) {
    return NC_ERROR;
  }

  if (deviceHandle != this->mDevHandle)
    return NC_NOT_ALLOCATED;

  return NC_OK;
}

/**
 * @brief A method mocking ncDeviceClose()
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncDeviceClose(
    struct ncDeviceHandle_t *deviceHandle)
{
  if (this->mFailStage == fail_stage_t::FAIL_DEV_CLOSE) {
    return NC_ERROR;
  }

  if (deviceHandle != this->mDevHandle)
    return NC_NOT_ALLOCATED;

  return NC_OK;
}

/**
 * @brief A method mocking ncDeviceDestroy()
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncDeviceDestroy(
    struct ncDeviceHandle_t **deviceHandle)
{
  if ((this->mDevHandle != nullptr) && (*deviceHandle != nullptr)
      && (*deviceHandle == this->mDevHandle)) {
    delete this->mDevHandle;
    this->mDevHandle = nullptr;
    *deviceHandle = nullptr;
  } else {
    return NC_NOT_ALLOCATED;
  }

  return NC_OK;
}

/**
 * @brief A method mocking ncGraphCreate()
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncGraphCreate(const char* name,
    struct ncGraphHandle_t **graphHandle)
{
  if (this->mFailStage == fail_stage_t::FAIL_GRAPH_CREATE) {
    return NC_ERROR;
  }

  if (!g_file_test (name, G_FILE_TEST_IS_REGULAR)) {
    return NC_INVALID_PARAMETERS;
  }

  if (this->mGraphHandle == nullptr) {
    return NC_OUT_OF_MEMORY;
  }

  this->mModelPath = g_strdup (name);
  *graphHandle = this->mGraphHandle;

  return NC_OK;
}

/**
 * @brief A method mocking ncGraphAllocate()
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncGraphAllocate(
    struct ncDeviceHandle_t *deviceHandle, struct ncGraphHandle_t *graphHandle,
    const void *graphBuffer, unsigned int graphBufferLength)
{
  if (this->mFailStage == fail_stage_t::FAIL_GRAPH_ALLOC) {
    return NC_ERROR;
  }

  if ((this->mDevHandle != deviceHandle) || (this->mGraphHandle != graphHandle))
    return NC_INVALID_PARAMETERS;

  this->mGraphBuf = graphBuffer;
  this->mLenGraphBuf = graphBufferLength;

  return NC_OK;
}

/**
 * @brief A method mocking ncGraphGetOption().
 * A description of input or output tensor will be filled in data.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncGraphGetOption(
    struct ncGraphHandle_t *graphHandle, int option, void *data,
    unsigned int *dataLength)
{
  switch (option) {
    case NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS:
      if (this->mFailStage == fail_stage_t::FAIL_GRAPH_GET_INPUT_TENSOR_DESC) {
        return NC_ERROR;
      }
      if (this->mTensorDescInput == nullptr) {
        return NC_OUT_OF_MEMORY;
      }
      if (sizeof(*this->mTensorDescInput) != (*dataLength))
        return NC_INVALID_PARAMETERS;
      memcpy (data, (void *) this->mTensorDescInput, *dataLength);
      break;
    case NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS:
      if (this->mFailStage == fail_stage_t::FAIL_GRAPH_GET_OUTPUT_TENSOR_DESC) {
        return NC_ERROR;
      }
      if (this->mTensorDescOutput == nullptr) {
        return NC_OUT_OF_MEMORY;
      }
      if (sizeof(*this->mTensorDescOutput) != (*dataLength))
        return NC_INVALID_PARAMETERS;
      memcpy (data, (void *) this->mTensorDescOutput, *dataLength);
      break;
  }

  return NC_OK;
}

/**
 * @brief A method mocking ncGraphQueueInference(). Do nothing.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncGraphQueueInference(
    struct ncGraphHandle_t *graphHandle, struct ncFifoHandle_t** fifoIn,
    unsigned int inFifoCount, struct ncFifoHandle_t** fifoOut,
    unsigned int outFifoCount)
{
  if (this->mFailStage == fail_stage_t::FAIL_GRAPH_Q_INFER) {
    return NC_ERROR;
  }

  return NC_OK;
}

/**
 * @brief A method mocking ncGraphQueueInference(). As the destructor does,
 * this deallocates the resources allocated by ncGraphCreate()
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncGraphDestroy(
  struct ncGraphHandle_t **graphHandle)
{
  if ((*graphHandle == nullptr) || (*graphHandle != this->mGraphHandle)) {
    return NC_INVALID_PARAMETERS;
  }

  if (this->mModelPath != nullptr) {
    g_free (this->mModelPath);
    this->mModelPath = nullptr;
  }

  if (this->mGraphHandle != nullptr) {
    delete this->mGraphHandle;
    this->mGraphHandle = nullptr;
  }

  return NC_OK;
}

/**
 * @brief A method mocking ncFifoCreate(). Do nothing.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncFifoCreate(const char* name, ncFifoType_t type,
    struct ncFifoHandle_t** fifoHandle)
{
  if ((this->mFailStage == fail_stage_t::FAIL_FIFO_CREATE_INPUT)
      && !g_strcmp0 (name, NNS_MVNCSDK2_NAME_INPUT_FIFO)) {
    return NC_ERROR;
  } else if ((this->mFailStage == fail_stage_t::FAIL_FIFO_CREATE_OUTPUT)
      && !g_strcmp0 (name, NNS_MVNCSDK2_NAME_OUTPUT_FIFO)) {
    return NC_ERROR;
  }

  return NC_OK;
}

/**
 * @brief A method mocking ncFifoAllocate(). Do nothing.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncFifoAllocate(struct ncFifoHandle_t* fifoHandle,
    struct ncDeviceHandle_t* device, struct ncTensorDescriptor_t* tensorDesc,
    unsigned int numElem)
{
  if ((this->mFailStage == fail_stage_t::FAIL_FIFO_ALLOC_INPUT)
      && (compareTensorDesc (*tensorDesc, *(this->mTensorDescInput)))) {
    return NC_ERROR;
  } else if ((this->mFailStage == fail_stage_t::FAIL_FIFO_ALLOC_OUTPUT)
      && (compareTensorDesc (*tensorDesc, *(this->mTensorDescOutput)))) {
    return NC_ERROR;
  }

  return NC_OK;
}

/**
 * @brief A method mocking ncFifoSetOption(). Do nothing.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncFifoSetOption(struct ncFifoHandle_t* fifoHandle,
    int option, const void *data, unsigned int dataLength)
{
  return NC_OK;
}

/**
 * @brief A method mocking ncFifoGetOption(). Do nothing.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncFifoGetOption(struct ncFifoHandle_t* fifoHandle,
    int option, void *data, unsigned int *dataLength)
{
  return NC_OK;
}

/**
 * @brief A method mocking ncFifoDestroy(). Do nothing.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncFifoDestroy(struct ncFifoHandle_t** fifoHandle)
{
  return NC_OK;
}

/**
 * @brief A method mocking ncFifoWriteElem(). Do nothing.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncFifoWriteElem(struct ncFifoHandle_t* fifoHandle,
    const void *inputTensor, unsigned int * inputTensorLength, void *userParam)
{
 if (this->mFailStage == fail_stage_t::FAIL_FIFO_WRT_ELEM) {
    return NC_ERROR;
  }

  return NC_OK;
}

/**
 * @brief A method mocking ncFifoReadElem(). Do nothing.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncFifoReadElem(struct ncFifoHandle_t* fifoHandle,
    void *outputData, unsigned int* outputDataLen, void **userParam)
{
  if (this->mFailStage == fail_stage_t::FAIL_FIFO_RD_ELEM) {
    return NC_ERROR;
  }

  return NC_OK;
}

/**
 * @brief A method mocking ncFifoRemoveElem(). Do nothing.
 */
ncStatus_t
NCSDKTensorFilterTestHelper::ncFifoRemoveElem(struct ncFifoHandle_t* fifoHandle)
{
  if (this->mFailStage == fail_stage_t::FAIL_FIFO_RM_ELEM) {
    return NC_ERROR;
  }

  return NC_OK;
}

/**
 * @brief A function definition providing the dummy body of ncGlobalGetOption() to the NCSDK2 tensor filter
 */
ncStatus_t
ncGlobalGetOption (int option, void *data, unsigned int *dataLength)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncGlobalGetOption (option,
      data, dataLength);
}

/**
 * @brief A function definition providing the dummy body of ncDeviceCreate() to the NCSDK2 tensor filter
 */
ncStatus_t
ncDeviceCreate (int index, struct ncDeviceHandle_t **deviceHandle)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncDeviceCreate (index,
      deviceHandle);
}

/**
 * @brief A function definition providing the dummy body of ncDeviceOpen() to the NCSDK2 tensor filter
 */
ncStatus_t
ncDeviceOpen(struct ncDeviceHandle_t *deviceHandle)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncDeviceOpen (deviceHandle);
}

/**
 * @brief A function definition providing the dummy body of ncDeviceClose() to the NCSDK2 tensor filter
 */
ncStatus_t
ncDeviceClose(struct ncDeviceHandle_t *deviceHandle)
{
  return NCSDKTensorFilterTestHelper::getInstance()
      .ncDeviceClose (deviceHandle);
}

/**
 * @brief A function definition providing the dummy body of ncDeviceDestroy() to the NCSDK2 tensor filter
 */
ncStatus_t
ncDeviceDestroy(struct ncDeviceHandle_t **deviceHandle)
{
  return NCSDKTensorFilterTestHelper::getInstance()
      .ncDeviceDestroy (deviceHandle);
}

/**
 * @brief A function definition providing the dummy body of ncGraphCreate() to the NCSDK2 tensor filter
 */
ncStatus_t
ncGraphCreate (const char* name, struct ncGraphHandle_t **graphHandle)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncGraphCreate (name,
      graphHandle);
}

/**
 * @brief A function definition providing the dummy body of ncGraphAllocate() to the NCSDK2 tensor filter
 */
ncStatus_t
ncGraphAllocate(struct ncDeviceHandle_t *deviceHandle,
      struct ncGraphHandle_t *graphHandle, const void *graphBuffer,
      unsigned int graphBufferLength)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncGraphAllocate (
      deviceHandle, graphHandle, graphBuffer, graphBufferLength);
}

/**
 * @brief A function definition providing the dummy body of ncGraphGetOption() to the NCSDK2 tensor filter
 */
ncStatus_t
ncGraphGetOption(struct ncGraphHandle_t *graphHandle, int option, void *data,
    unsigned int *dataLength)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncGraphGetOption (
      graphHandle, option, data, dataLength);
}

/**
 * @brief A function definition providing the dummy body of ncGraphQueueInference() to the NCSDK2 tensor filter
 */
ncStatus_t
ncGraphQueueInference(struct ncGraphHandle_t *graphHandle,
    struct ncFifoHandle_t** fifoIn, unsigned int inFifoCount,
    struct ncFifoHandle_t** fifoOut, unsigned int outFifoCount)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncGraphQueueInference (
      graphHandle, fifoIn, inFifoCount, fifoOut, outFifoCount);
}

/**
 * @brief A function definition providing the dummy body of ncGraphDestroy() to the NCSDK2 tensor filter
 */
ncStatus_t
ncGraphDestroy(struct ncGraphHandle_t **graphHandle)
{
  return NCSDKTensorFilterTestHelper::getInstance()
      .ncGraphDestroy (graphHandle);
}

/**
 * @brief A function definition providing the dummy body of ncFifoCreate() to the NCSDK2 tensor filter
 */
ncStatus_t
ncFifoCreate(const char* name, ncFifoType_t type,
    struct ncFifoHandle_t** fifoHandle)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncFifoCreate (name, type,
      fifoHandle);
}

/**
 * @brief A function definition providing the dummy body of ncFifoAllocate() to the NCSDK2 tensor filter
 */
ncStatus_t
ncFifoAllocate(struct ncFifoHandle_t* fifoHandle,
    struct ncDeviceHandle_t* device, struct ncTensorDescriptor_t* tensorDesc,
    unsigned int numElem)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncFifoAllocate (fifoHandle,
      device, tensorDesc, numElem);
}

/**
 * @brief A function definition providing the dummy body of ncFifoSetOption() to the NCSDK2 tensor filter
 */
ncStatus_t
ncFifoSetOption(struct ncFifoHandle_t* fifoHandle, int option,
    const void *data, unsigned int dataLength)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncFifoSetOption (fifoHandle,
      option, data, dataLength);
}

/**
 * @brief A function definition providing the dummy body of ncFifoGetOption() to the NCSDK2 tensor filter
 */
ncStatus_t
ncFifoGetOption(struct ncFifoHandle_t* fifoHandle, int option, void *data,
    unsigned int *dataLength)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncFifoGetOption (fifoHandle,
      option, data, dataLength);
}

/**
 * @brief A function definition providing the dummy body of ncFifoDestroy() to the NCSDK2 tensor filter
 */
ncStatus_t
ncFifoDestroy(struct ncFifoHandle_t** fifoHandle)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncFifoDestroy (fifoHandle);
}

/**
 * @brief A function definition providing the dummy body of ncFifoWriteElem() to the NCSDK2 tensor filter
 */
ncStatus_t
ncFifoWriteElem(struct ncFifoHandle_t* fifoHandle, const void *inputTensor,
    unsigned int * inputTensorLength, void *userParam)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncFifoWriteElem (fifoHandle,
      inputTensor, inputTensorLength, userParam);
}

/**
 * @brief A function definition providing the dummy body of ncFifoReadElem() to the NCSDK2 tensor filter
 */
ncStatus_t
ncFifoReadElem(struct ncFifoHandle_t* fifoHandle, void *outputData,
    unsigned int* outputDataLen, void **userParam)
{
  return NCSDKTensorFilterTestHelper::getInstance().ncFifoWriteElem (fifoHandle,
      outputData, outputDataLen, userParam);
}

/**
 * @brief A function definition providing the dummy body of ncFifoRemoveElem() to the NCSDK2 tensor filter
 */
ncStatus_t
ncFifoRemoveElem(struct ncFifoHandle_t* fifoHandle)
{
  return NCSDKTensorFilterTestHelper::getInstance()
      .ncFifoRemoveElem (fifoHandle);
}
