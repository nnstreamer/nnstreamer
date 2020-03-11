package org.nnsuite.nnstreamer;

import android.os.Environment;
import android.support.test.rule.GrantPermissionRule;
import android.support.test.runner.AndroidJUnit4;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;

import static org.junit.Assert.*;

/**
 * Testcases for SingleShot.
 */
@RunWith(AndroidJUnit4.class)
public class APITestSingleShot {
    private SingleShot mSingle;

    @Rule
    public GrantPermissionRule mPermissionRule = APITestCommon.grantPermissions();

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        try {
            mSingle = new SingleShot(APITestCommon.getTFLiteImgModel());
        } catch (Exception e) {
            fail();
        }
    }

    @After
    public void tearDown() {
        mSingle.close();
    }

    @Test
    public void testGetInputInfo() {
        try {
            TensorsInfo info = mSingle.getInputInfo();

            /* input: uint8 3:224:224:1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.UINT8, info.getTensorType(0));
            assertArrayEquals(new int[]{3,224,224,1}, info.getTensorDimension(0));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetOutputInfo() {
        try {
            TensorsInfo info = mSingle.getOutputInfo();

            /* output: uint8 1001:1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.UINT8, info.getTensorType(0));
            assertArrayEquals(new int[]{1001,1,1,1}, info.getTensorDimension(0));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSetNullInputInfo_n() {
        try {
            mSingle.setInputInfo(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInputInfo() {
        try {
            SingleShot addSingle = new SingleShot(APITestCommon.getTFLiteAddModel());
            TensorsInfo info = addSingle.getInputInfo();

            /* input: float32 with dimension 1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.FLOAT32, info.getTensorType(0));
            assertArrayEquals(new int[]{1,1,1,1}, info.getTensorDimension(0));

            TensorsInfo newInfo = new TensorsInfo();
            newInfo.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{10});

            addSingle.setInputInfo(newInfo);

            info = addSingle.getInputInfo();
            /* input: float32 with dimension 10 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.FLOAT32, info.getTensorType(0));
            assertArrayEquals(new int[]{10,1,1,1}, info.getTensorDimension(0));

            info = addSingle.getOutputInfo();
            /* output: float32 with dimension 10 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.FLOAT32, info.getTensorType(0));
            assertArrayEquals(new int[]{10,1,1,1}, info.getTensorDimension(0));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInvoke() {
        try {
            TensorsInfo info = mSingle.getInputInfo();

            /* let's ignore timeout (set 10 sec) */
            mSingle.setTimeout(10000);

            /* single-shot invoke */
            for (int i = 0; i < 600; i++) {
                /* dummy input */
                TensorsData out = mSingle.invoke(info.allocate());

                /* output: uint8 1001:1 */
                assertEquals(1, out.getTensorsCount());
                assertEquals(1001, out.getTensorData(0).capacity());

                Thread.sleep(30);
            }
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInvokeDynamicVary() {
        try {
            SingleShot addSingle = new SingleShot(APITestCommon.getTFLiteAddModel());

            /* single-shot invoke */
            for (int i = 1; i < 2; i++) {
                TensorsInfo info = new TensorsInfo();
                info.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{1,1,1,i});

                /* dummy input */
                TensorsData out = addSingle.invoke(TensorsData.allocate(info));

                /* output: float32 1:1:1:i */
                assertEquals(1, out.getTensorsCount());
                assertEquals(i * Float.BYTES, out.getTensorData(0).capacity());

                Thread.sleep(30);
            }
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInvokeTimeout_n() {
        TensorsInfo info = mSingle.getInputInfo();

        /* timeout 5ms */
        mSingle.setTimeout(5);

        try {
            /* dummy input */
            mSingle.invoke(TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testNullFile_n() {
        try {
            new SingleShot(null);
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
        try {
            mSingle.invoke(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvokeInvalidData_n() {
        /* input data size: 3 * 224 * 224 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{100});

        try {
            mSingle.invoke(TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetZeroTimeout_n() {
        try {
            mSingle.setTimeout(0);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidTimeout_n() {
        try {
            mSingle.setTimeout(-1);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetInvalidPropertyName_n() {
        try {
            mSingle.getValue("");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetUnknownPropertyName_n() {
        try {
            mSingle.getValue("unknown_prop");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetNullPropertyName_n() {
        try {
            mSingle.getValue(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnknownPropertyName_n() {
        try {
            mSingle.setValue("unknown_prop", "unknown");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetNullPropertyName_n() {
        try {
            mSingle.setValue(null, "ANY");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetNullPropertyValue_n() {
        try {
            mSingle.setValue("inputlayout", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetPropertyDimension() {
        try {
            assertEquals("3:224:224:1", mSingle.getValue("input"));
            assertEquals("1001:1:1:1", mSingle.getValue("output"));
        } catch (Exception e) {
            fail();
        }
    }

    /**
     * Run SNAP with Caffe model.
     */
    private void runSNAPCaffe(boolean useGPU) {
        File[] models = APITestCommon.getSNAPCaffeModel();
        String option = APITestCommon.getSNAPCaffeOption(useGPU);

        try {
            TensorsInfo in = new TensorsInfo();
            in.addTensorInfo("data", NNStreamer.TensorType.FLOAT32, new int[]{3,224,224,1});

            TensorsInfo out = new TensorsInfo();
            out.addTensorInfo("prob", NNStreamer.TensorType.FLOAT32, new int[]{1,1,1000,1});

            SingleShot single = new SingleShot(models, in, out, NNStreamer.NNFWType.SNAP, option);

            /* let's ignore timeout (set 60 sec) */
            single.setTimeout(60000);

            /* set layout */
            single.setValue("inputlayout", "NHWC");
            single.setValue("outputlayout", "NCHW");

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

        runSNAPCaffe(false);
    }

    @Test
    public void testSNAPCaffeGPU() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.SNAP)) {
            /* cannot run the test */
            return;
        }

        runSNAPCaffe(true);
    }
}
