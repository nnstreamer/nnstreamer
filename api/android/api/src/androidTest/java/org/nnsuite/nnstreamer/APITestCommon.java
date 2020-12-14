package org.nnsuite.nnstreamer;

import android.Manifest;
import android.content.Context;
import android.os.Environment;
import android.support.test.InstrumentationRegistry;
import android.support.test.rule.GrantPermissionRule;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;

import static org.junit.Assert.*;

/**
 * Common definition to test NNStreamer API.
 *
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class APITestCommon {
    private static boolean mInitialized = false;

    /**
     * Initializes NNStreamer API library.
     */
    public static void initNNStreamer() {
        if (!mInitialized) {
            try {
                Context context = InstrumentationRegistry.getTargetContext();
                mInitialized = NNStreamer.initialize(context);
            } catch (Exception e) {
                fail();
            }
        }
    }

    /**
     * Gets the context for the test application.
     */
    public static Context getContext() {
        return InstrumentationRegistry.getTargetContext();
    }

    /**
     * Grants required runtime permissions.
     */
    public static GrantPermissionRule grantPermissions() {
        return GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE);
    }

    /**
     * Gets the File object of tensorflow-lite model.
     * Note that, to invoke model in the storage, the permission READ_EXTERNAL_STORAGE is required.
     */
    public static File getTFLiteImgModel() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File model = new File(root + "/nnstreamer/test/imgclf/mobilenet_v1_1.0_224_quant.tflite");
        File meta = new File(root + "/nnstreamer/test/imgclf/metadata/MANIFEST");

        if (!model.exists() || !meta.exists()) {
            fail();
        }

        return model;
    }

    /**
     * Reads raw image file (orange) and returns TensorsData instance.
     */
    public static TensorsData readRawImageData() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File raw = new File(root + "/nnstreamer/test/orange.raw");

        if (!raw.exists()) {
            fail();
        }

        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{3,224,224,1});

        int size = info.getTensorSize(0);
        TensorsData data = TensorsData.allocate(info);

        try {
            byte[] content = Files.readAllBytes(raw.toPath());
            if (content.length != size) {
                fail();
            }

            ByteBuffer buffer = TensorsData.allocateByteBuffer(size);
            buffer.put(content);

            data.setTensorData(0, buffer);
        } catch (Exception e) {
            fail();
        }

        return data;
    }

    /**
     * Gets the label index with max score, for tensorflow-lite image classification.
     */
    public static int getMaxScore(ByteBuffer buffer) {
        int index = -1;
        int maxScore = 0;

        if (isValidBuffer(buffer, 1001)) {
            for (int i = 0; i < 1001; i++) {
                /* convert unsigned byte */
                int score = (buffer.get(i) & 0xFF);

                if (score > maxScore) {
                    maxScore = score;
                    index = i;
                }
            }
        }

        return index;
    }

    /**
     * Reads raw float image file (plastic_cup) and returns TensorsData instance.
     */
    public static TensorsData readRawImageDataSNPE() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File raw = new File(root + "/nnstreamer/snpe_data/plastic_cup.raw");

        if (!raw.exists()) {
            fail();
        }

        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{3, 299, 299, 1});

        int size = info.getTensorSize(0);
        TensorsData data = TensorsData.allocate(info);

        try {
            byte[] content = Files.readAllBytes(raw.toPath());
            if (content.length != size) {
                fail();
            }

            ByteBuffer buffer = TensorsData.allocateByteBuffer(size);
            buffer.put(content);

            data.setTensorData(0, buffer);
        } catch (Exception e) {
            fail();
        }

        return data;
    }

    /**
     * Gets the label index with max score, for SNPE image classification.
     */
    public static int getMaxScoreSNPE(ByteBuffer buffer) {
        int index = -1;
        float maxScore = -Float.MAX_VALUE;

        if (isValidBuffer(buffer, 4004)) {
            for (int i = 0; i < 1001; i++) {
                /* convert to float */
                float score = buffer.getFloat(i * 4);

                if (score > maxScore) {
                    maxScore = score;
                    index = i;
                }
            }
        }

        return index;
    }

    /**
     * Gets the path string of tensorflow-lite add.tflite model.
     */
    public static String getTFLiteAddModelPath() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        return root + "/nnstreamer/test/add";
    }

    /**
     * Gets the File object of tensorflow-lite add.tflite model.
     * Note that, to invoke model in the storage, the permission READ_EXTERNAL_STORAGE is required.
     */
    public static File getTFLiteAddModel() {
        String path = getTFLiteAddModelPath();
        File model = new File(path + "/add.tflite");
        File meta = new File(path + "/metadata/MANIFEST");

        if (!model.exists() || !meta.exists()) {
            fail();
        }

        return model;
    }

    public enum SNAPComputingUnit {
        CPU("ComputingUnit:CPU"),
        GPU("ComputingUnit:GPU,GpuCacheSource:/sdcard/nnstreamer/"),
        DSP("ComputingUnit:DSP"),
        NPU("ComputingUnit:NPU");

        private String computing_unit_option;

        SNAPComputingUnit(String computing_unit_option) {
            this.computing_unit_option = computing_unit_option;
        }

        public String getOptionString() {
            return computing_unit_option;
        }
    }

    /**
     * Gets the File objects of Caffe model for SNAP.
     * Note that, to invoke model in the storage, the permission READ_EXTERNAL_STORAGE is required.
     */
    public static File[] getSNAPCaffeModel() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();

        File model = new File(root + "/nnstreamer/snap_data/prototxt/squeezenet.prototxt");
        File weight = new File(root + "/nnstreamer/snap_data/model/squeezenet.caffemodel");

        if (!model.exists() || !weight.exists()) {
            fail();
        }

        return new File[]{model, weight};
    }

    /**
     * Gets the option string to run Caffe model for SNAP.
     *
     * CPU: "custom=ModelFWType:CAFFE,ExecutionDataType:FLOAT32,ComputingUnit:CPU"
     * GPU: "custom=ModelFWType:CAFFE,ExecutionDataType:FLOAT32,ComputingUnit:GPU,GpuCacheSource:/sdcard/nnstreamer/"
     */
    public static String getSNAPCaffeOption(SNAPComputingUnit CUnit) {
        String option = "ModelFWType:CAFFE,ExecutionDataType:FLOAT32,InputFormat:NHWC,OutputFormat:NCHW,";
        option = option + CUnit.getOptionString();

        return option;
    }

    /**
     * Gets the option string to run Tensorflow model for SNAP.
     *
     * CPU: "custom=ModelFWType:CAFFE,ExecutionDataType:FLOAT32,ComputingUnit:CPU"
     * GPU: Not supported for Tensorflow model
     * DSP: "custom=ModelFWType:CAFFE,ExecutionDataType:FLOAT32,ComputingUnit:DSP"
     * NPU: "custom=ModelFWType:CAFFE,ExecutionDataType:FLOAT32,ComputingUnit:NPU"
     */
    public static String getSNAPTensorflowOption(SNAPComputingUnit CUnit) {
        String option = "ModelFWType:TENSORFLOW,ExecutionDataType:FLOAT32,InputFormat:NHWC,OutputFormat:NHWC,";
        option = option + CUnit.getOptionString();
        return option;
    }

    /**
     * Gets the File objects of Tensorflow model for SNAP.
     * Note that, to invoke model in the storage, the permission READ_EXTERNAL_STORAGE is required.
     */
    public static File[] getSNAPTensorflowModel(SNAPComputingUnit CUnit) {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        String model_path = "/nnstreamer/snap_data/model/";

        switch (CUnit) {
            case CPU:
                model_path = model_path + "yolo_new.pb";
                break;
            case DSP:
                model_path = model_path + "yolo_new_tf_quantized.dlc";
                break;
            case NPU:
                model_path = model_path + "yolo_new_tf_quantized_hta.dlc";
                break;
            case GPU:
            default:
                fail();
        }

        File model = new File(root + model_path);
        if (!model.exists()) {
            fail();
        }

        return new File[]{model};
    }

    public static File getSNPEModel() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();

        File model = new File(root + "/nnstreamer/snpe_data/inception_v3_quantized.dlc");

        if (!model.exists()) {
            fail();
        }

        return model;
    }

    /**
     * Get the File object of SNPE model for testing multiple output.
     * The model is converted to dlc format with SNPE SDK and it's from
     * https://github.com/nnsuite/testcases/tree/master/DeepLearningModels/tensorflow/ssdlite_mobilenet_v2
     */
    public static File getMultiOutputSNPEModel() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();

        File model = new File(root + "/nnstreamer/snpe_data/ssdlite_mobilenet_v2.dlc");

        if (!model.exists()) {
            fail();
        }

        return model;
    }

    /**
     * Verifies the byte buffer is direct buffer with native order.
     *
     * @param buffer   The byte buffer
     * @param expected The expected capacity
     *
     * @return True if the byte buffer is valid.
     */
    public static boolean isValidBuffer(ByteBuffer buffer, int expected) {
        if (buffer != null && buffer.isDirect() && buffer.order() == ByteOrder.nativeOrder()) {
            int capacity = buffer.capacity();

            return (capacity == expected);
        }

        return false;
    }

    @Before
    public void setUp() {
        initNNStreamer();
    }

    @Test
    public void useAppContext() {
        Context context = InstrumentationRegistry.getTargetContext();

        assertEquals("org.nnsuite.nnstreamer.test", context.getPackageName());
    }

    @Test
    public void testInitWithInvalidCtx_n() {
        try {
            NNStreamer.initialize(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void enumTensorType() {
        assertEquals(NNStreamer.TensorType.INT32, NNStreamer.TensorType.valueOf("INT32"));
        assertEquals(NNStreamer.TensorType.UINT32, NNStreamer.TensorType.valueOf("UINT32"));
        assertEquals(NNStreamer.TensorType.INT16, NNStreamer.TensorType.valueOf("INT16"));
        assertEquals(NNStreamer.TensorType.UINT16, NNStreamer.TensorType.valueOf("UINT16"));
        assertEquals(NNStreamer.TensorType.INT8, NNStreamer.TensorType.valueOf("INT8"));
        assertEquals(NNStreamer.TensorType.UINT8, NNStreamer.TensorType.valueOf("UINT8"));
        assertEquals(NNStreamer.TensorType.FLOAT64, NNStreamer.TensorType.valueOf("FLOAT64"));
        assertEquals(NNStreamer.TensorType.FLOAT32, NNStreamer.TensorType.valueOf("FLOAT32"));
        assertEquals(NNStreamer.TensorType.INT64, NNStreamer.TensorType.valueOf("INT64"));
        assertEquals(NNStreamer.TensorType.UINT64, NNStreamer.TensorType.valueOf("UINT64"));
        assertEquals(NNStreamer.TensorType.UNKNOWN, NNStreamer.TensorType.valueOf("UNKNOWN"));
    }

    @Test
    public void enumNNFWType() {
        assertEquals(NNStreamer.NNFWType.TENSORFLOW_LITE, NNStreamer.NNFWType.valueOf("TENSORFLOW_LITE"));
        assertEquals(NNStreamer.NNFWType.SNAP, NNStreamer.NNFWType.valueOf("SNAP"));
        assertEquals(NNStreamer.NNFWType.NNFW, NNStreamer.NNFWType.valueOf("NNFW"));
        assertEquals(NNStreamer.NNFWType.SNPE, NNStreamer.NNFWType.valueOf("SNPE"));
        assertEquals(NNStreamer.NNFWType.UNKNOWN, NNStreamer.NNFWType.valueOf("UNKNOWN"));
    }
}
