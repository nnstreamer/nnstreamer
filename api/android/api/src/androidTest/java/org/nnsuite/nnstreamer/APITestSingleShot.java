package org.nnsuite.nnstreamer;

import android.os.Environment;
import android.support.test.rule.GrantPermissionRule;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.nio.ByteBuffer;

import static org.junit.Assert.*;

/**
 * Testcases for SingleShot.
 */
@RunWith(AndroidJUnit4.class)
public class APITestSingleShot {
    @Rule
    public GrantPermissionRule mPermissionRule = APITestCommon.grantPermissions();

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();
    }

    @Test
    public void testOptionsInvalidFW_n () {
        try {
            new SingleShot.Options(null, APITestCommon.getTFLiteImgModel());
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testOptionsUnknownFW_n () {
        try {
            new SingleShot.Options(NNStreamer.NNFWType.UNKNOWN, APITestCommon.getTFLiteImgModel());
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testOptionsInvalidModelFile_n () {
        try {
            File f = null;
            new SingleShot.Options(NNStreamer.NNFWType.TENSORFLOW_LITE, f);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetInputInfo() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());
            TensorsInfo info = single.getInputInfo();

            /* input: uint8 3:224:224:1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.UINT8, info.getTensorType(0));
            assertArrayEquals(new int[]{3,224,224,1}, info.getTensorDimension(0));

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetOutputInfo() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());
            TensorsInfo info = single.getOutputInfo();

            /* output: uint8 1001:1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.UINT8, info.getTensorType(0));
            assertArrayEquals(new int[]{1001,1,1,1}, info.getTensorDimension(0));

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSetNullInputInfo_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());
            single.setInputInfo(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidInputInfo_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            TensorsInfo newInfo = new TensorsInfo();
            newInfo.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,2,2,2});

            single.setInputInfo(newInfo);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInputInfo() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteAddModel());
            TensorsInfo info = single.getInputInfo();

            /* input: float32 with dimension 1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.FLOAT32, info.getTensorType(0));
            assertArrayEquals(new int[]{1,1,1,1}, info.getTensorDimension(0));

            TensorsInfo newInfo = new TensorsInfo();
            newInfo.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{10});

            single.setInputInfo(newInfo);

            info = single.getInputInfo();
            /* input: float32 with dimension 10 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.FLOAT32, info.getTensorType(0));
            assertArrayEquals(new int[]{10,1,1,1}, info.getTensorDimension(0));

            info = single.getOutputInfo();
            /* output: float32 with dimension 10 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.FLOAT32, info.getTensorType(0));
            assertArrayEquals(new int[]{10,1,1,1}, info.getTensorDimension(0));

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInvoke() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());
            TensorsInfo info = single.getInputInfo();

            /* let's ignore timeout (set 10 sec) */
            single.setTimeout(10000);

            /* single-shot invoke */
            for (int i = 0; i < 600; i++) {
                /* dummy input */
                TensorsData out = single.invoke(info.allocate());

                /* output: uint8 1001:1 */
                assertEquals(1, out.getTensorsCount());
                assertEquals(1001, out.getTensorData(0).capacity());

                Thread.sleep(30);
            }

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    /**
     * Run image classification and validate result.
     */
    private void runImageClassification(NNStreamer.NNFWType fw) {
        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel(), fw);

            /* single-shot invoke */
            TensorsData in = APITestCommon.readRawImageData();
            TensorsData out = single.invoke(in);
            int labelIndex = APITestCommon.getMaxScore(out.getTensorData(0));

            /* check label index (orange) */
            if (labelIndex != 951) {
                fail();
            }

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testClassificationResultTFLite() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        runImageClassification(NNStreamer.NNFWType.TENSORFLOW_LITE);
    }

    @Test
    public void testClassificationResultNNFW() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.NNFW)) {
            /* cannot run the test */
            return;
        }

        runImageClassification(NNStreamer.NNFWType.NNFW);
    }

    /**
     * Run dynamic invoke with add.tflite model.
     */
    private void runInvokeDynamic(NNStreamer.NNFWType fw) {
        try {
            File model = APITestCommon.getTFLiteAddModel();
            SingleShot single = new SingleShot(model, fw);
            TensorsInfo info = single.getInputInfo();

            /* single-shot invoke */
            for (int i = 1; i < 5; i++) {
                /* change input information */
                info.setTensorDimension(0, new int[]{i,1,1,1});
                single.setInputInfo(info);

                TensorsData input = TensorsData.allocate(info);
                ByteBuffer inBuffer = input.getTensorData(0);

                for (int j = 0; j < i; j++) {
                    inBuffer.putFloat(j * 4, j + 1.5f);
                }

                input.setTensorData(0, inBuffer);

                /* invoke */
                TensorsData output = single.invoke(input);

                /* output: float32 i:1:1:1 */
                assertEquals(1, output.getTensorsCount());

                ByteBuffer outBuffer = output.getTensorData(0);
                assertEquals(i * Float.BYTES, outBuffer.capacity());

                for (int j = 0; j < i; j++) {
                    float expected = j + 3.5f;
                    assertEquals(expected, outBuffer.getFloat(j * 4), 0.0f);
                }

                Thread.sleep(30);
            }

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInvokeDynamicTFLite() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        runInvokeDynamic(NNStreamer.NNFWType.TENSORFLOW_LITE);
    }

    @Test
    public void testInvokeDynamicNNFW() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.NNFW)) {
            /* cannot run the test */
            return;
        }

        runInvokeDynamic(NNStreamer.NNFWType.NNFW);
    }

    @Test
    public void testInvokeTimeout_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());
            TensorsInfo info = single.getInputInfo();

            /* timeout 1ms (not enough time to invoke the model) */
            single.setTimeout(1);

            /* dummy input */
            single.invoke(TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testNullFile_n() {
        try {
            File f = null;
            new SingleShot(f);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testNullFiles_n() {
        try {
            new SingleShot(null, null, null, NNStreamer.NNFWType.TENSORFLOW_LITE, null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvalidFile_n() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File model = new File(root + "/invalid_path/invalid.tflite");

        try {
            new SingleShot(model);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvalidInputType_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        /* input: uint8 3:224:224:1 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.UINT16, new int[]{3,224,224,1});

        try {
            new SingleShot(APITestCommon.getTFLiteImgModel(), info, null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvalidInputDimension_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        /* input: uint8 3:224:224:1 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,224,224});

        try {
            new SingleShot(APITestCommon.getTFLiteImgModel(), info, null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvalidOutputType_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        /* output: uint8 1001:1 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.INT16, new int[]{1001,1});

        try {
            new SingleShot(APITestCommon.getTFLiteImgModel(), null, info);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvalidOutputDimension_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        /* output: uint8 1001:1 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{1001,2,1,1});

        try {
            new SingleShot(APITestCommon.getTFLiteImgModel(), null, info);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvokeNullData_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.invoke(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvokeInvalidData_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        /* input data size: 3 * 224 * 224 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{100});

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.invoke(TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidTimeout_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.setTimeout(-1);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetInvalidPropertyName_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.getValue("");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetUnknownPropertyName_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.getValue("unknown_prop");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetNullPropertyName_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.getValue(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetUnknownPropertyName_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.setValue("unknown_prop", "unknown");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetNullPropertyName_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.setValue(null, "ANY");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetEmptyPropertyName_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.setValue("", "ANY");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetNullPropertyValue_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            single.setValue("inputlayout", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetPropertyDimension() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteAddModel());

            single.setValue("input", "5:1:1:1");
            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetPropertyDimension() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            assertEquals("3:224:224:1", single.getValue("input"));
            assertEquals("1001:1:1:1", single.getValue("output"));

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    /**
     * Run SNAP with Caffe model.
     */
    private void runSNAPCaffe(APITestCommon.SNAPComputingUnit CUnit) {
        File[] models = APITestCommon.getSNAPCaffeModel();
        String option = APITestCommon.getSNAPCaffeOption(CUnit);

        try {
            TensorsInfo in = new TensorsInfo();
            in.addTensorInfo("data", NNStreamer.TensorType.FLOAT32, new int[]{3,224,224,1});

            TensorsInfo out = new TensorsInfo();
            out.addTensorInfo("prob", NNStreamer.TensorType.FLOAT32, new int[]{1,1,1000,1});

            SingleShot single = new SingleShot(models, in, out, NNStreamer.NNFWType.SNAP, option);

            /* let's ignore timeout (set 60 sec) */
            single.setTimeout(60000);

            /* single-shot invoke */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                TensorsData output = single.invoke(in.allocate());

                /* output: float32 1:1:1000:1 (NCHW format) */
                assertEquals(1, output.getTensorsCount());
                assertEquals(4000, output.getTensorData(0).capacity());

                Thread.sleep(30);
            }
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSNAPCaffeCPU() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        runSNAPCaffe(APITestCommon.SNAPComputingUnit.CPU);
    }

    @Test
    public void testSNAPCaffeGPU() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        runSNAPCaffe(APITestCommon.SNAPComputingUnit.GPU);
    }

    /**
     * Run SNAP with Tensorflow model.
     */
    private void runSNAPTensorflow(APITestCommon.SNAPComputingUnit CUnit) {
        File[] model = APITestCommon.getSNAPTensorflowModel(CUnit);
        String option = APITestCommon.getSNAPTensorflowOption(CUnit);

        try {
            TensorsInfo in = new TensorsInfo();
            in.addTensorInfo("input", NNStreamer.TensorType.FLOAT32, new int[]{3,224,224,1});

            TensorsInfo out = new TensorsInfo();
            out.addTensorInfo("MobilenetV1/Predictions/Reshape_1:0", NNStreamer.TensorType.FLOAT32, new int[]{1001, 1});

            SingleShot single = new SingleShot(model, in, out, NNStreamer.NNFWType.SNAP, option);

            /* let's ignore timeout (set 60 sec) */
            single.setTimeout(60000);

            /* single-shot invoke */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                TensorsData output = single.invoke(in.allocate());

                /* output: float32 1:1001 */
                assertEquals(1, output.getTensorsCount());
                assertEquals(4004, output.getTensorData(0).capacity());

                Thread.sleep(30);
            }

        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSNAPTensorflowCPU() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        runSNAPTensorflow(APITestCommon.SNAPComputingUnit.CPU);
    }

    @Test
    public void testSNAPTensorflowDSP() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        if (!android.os.Build.HARDWARE.equals("qcom")) {
            /*
             * Tensorflow model using DSP runtime can only be executed on
             * Snapdragon SoC. Cannot run this test on exynos.
             */
            return;
        }

        runSNAPTensorflow(APITestCommon.SNAPComputingUnit.DSP);
    }

    @Test
    public void testSNAPTensorflowNPU() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        if (!android.os.Build.HARDWARE.equals("qcom")) {
            /*
             * Tensorflow model using NPU runtime can only be executed on
             * Snapdragon. Cannot run this test on exynos.
             */
            return;
        }

        runSNAPTensorflow(APITestCommon.SNAPComputingUnit.NPU);
    }

    @Test
    public void testNNFWTFLite() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.NNFW)) {
            /* cannot run the test */
            return;
        }

        try {
            File model = APITestCommon.getTFLiteAddModel();

            SingleShot single = new SingleShot(model, NNStreamer.NNFWType.NNFW);
            TensorsInfo in = single.getInputInfo();

            /* let's ignore timeout (set 60 sec) */
            single.setTimeout(60000);

            /* single-shot invoke */
            for (int i = 0; i < 5; i++) {
                /* input data */
                TensorsData input = in.allocate();

                ByteBuffer buffer = input.getTensorData(0);
                buffer.putFloat(0, i + 1.5f);

                input.setTensorData(0, buffer);

                /* invoke */
                TensorsData output = single.invoke(input);

                /* check output */
                float expected = i + 3.5f;
                assertEquals(expected, output.getTensorData(0).getFloat(0), 0.0f);

                Thread.sleep(30);
            }

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testNNFWOpenDir() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.NNFW)) {
            /* cannot run the test */
            return;
        }

        try {
            File model = new File(APITestCommon.getTFLiteAddModelPath());

            new SingleShot(model, NNStreamer.NNFWType.NNFW);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testNNFWOpenInvalidDir_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.NNFW)) {
            /* cannot run the test */
            return;
        }

        try {
            File model = new File(APITestCommon.getTFLiteAddModelPath() + "/invaliddir");

            new SingleShot(model, NNStreamer.NNFWType.NNFW);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSNPE() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNPE)) {
            /* cannot run the test */
            return;
        }

        try {
            File model = APITestCommon.getSNPEModel();

            SingleShot single = new SingleShot(model, NNStreamer.NNFWType.SNPE);
            TensorsInfo in = single.getInputInfo();

            /* let's ignore timeout (set 60 sec) */
            single.setTimeout(60000);

            /* single-shot invoke */
            for (int i = 0; i < 5; i++) {
                /* input data */
                TensorsData input = in.allocate();

                /* invoke */
                TensorsData output = single.invoke(input);

                /* check output: 1 tensor (float32 1:1001) */
                assertEquals(1, output.getTensorsCount());
                assertEquals(4004, output.getTensorData(0).capacity());

                Thread.sleep(30);
            }

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSNPEClassificationResult() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNPE)) {
            /* cannot run the test */
            return;
        }

        /* expected label is measuring_cup (648) */
        final int expected_label = 648;

        try {
            File model = APITestCommon.getSNPEModel();

            SingleShot single = new SingleShot(model, NNStreamer.NNFWType.SNPE);

            /* let's ignore timeout (set 10 sec) */
            single.setTimeout(10000);

            /* single-shot invoke */
            TensorsData in = APITestCommon.readRawImageDataSNPE();
            TensorsData out = single.invoke(in);
            int labelIndex = APITestCommon.getMaxScoreSNPE(out.getTensorData(0));

            /* check label index (measuring cup) */
            if (labelIndex != expected_label) {
                fail();
            }

            single.close();
        } catch (Exception e) {
            fail();
        }
    }
}
