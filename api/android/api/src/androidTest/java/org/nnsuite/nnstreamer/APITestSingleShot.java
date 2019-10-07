package org.nnsuite.nnstreamer;

import android.Manifest;
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
    public GrantPermissionRule mPermissionRule =
            GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE);

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        try {
            mSingle = new SingleShot(APITestCommon.getTestModel());
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
            assertEquals(NNStreamer.TENSOR_TYPE_UINT8, info.getTensorType(0));
            assertArrayEquals(new int[]{3,224,224,1}, info.getTensorDimension(0));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetOutputInfo() {
        try {
            TensorsInfo info = mSingle.getOutputInfo();

            /* output: uint8 1001:1:1:1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TENSOR_TYPE_UINT8, info.getTensorType(0));
            assertArrayEquals(new int[]{1001,1,1,1}, info.getTensorDimension(0));
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
                TensorsData in = TensorsData.allocate(info);
                TensorsData out = mSingle.invoke(in);

                /* output: uint8 1001:1:1:1 */
                assertEquals(1, out.getTensorsCount());
                assertEquals(1001, out.getTensorData(0).capacity());

                Thread.sleep(30);
            }
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInvokeTimeout() {
        TensorsInfo info = mSingle.getInputInfo();

        /* timeout 5ms */
        mSingle.setTimeout(5);

        for (int i = 0; i < 5; i++) {
            /* dummy input */
            TensorsData in = TensorsData.allocate(info);

            try {
                mSingle.invoke(in);
                fail();
            } catch (Exception e) {
                /* expected */
            }
        }
    }

    @Test
    public void testNullFile() {
        try {
            new SingleShot(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvalidFile() {
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
    public void testInvalidInputType() {
        /* input: uint8 3:224:224:1 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TENSOR_TYPE_UINT16, new int[]{3,224,224,1});

        try {
            new SingleShot(APITestCommon.getTestModel(), info, null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvalidInputDimension() {
        /* input: uint8 3:224:224:1 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TENSOR_TYPE_UINT8, new int[]{2,224,224});

        try {
            new SingleShot(APITestCommon.getTestModel(), info, null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvalidOutputType() {
        /* output: uint8 1001:1:1:1 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TENSOR_TYPE_INT16, new int[]{1001,1,1,1});

        try {
            new SingleShot(APITestCommon.getTestModel(), null, info);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvalidOutputDimension() {
        /* output: uint8 1001:1:1:1 */
        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TENSOR_TYPE_UINT8, new int[]{1001,2,1,1});

        try {
            new SingleShot(APITestCommon.getTestModel(), null, info);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvokeNullData() {
        try {
            mSingle.invoke(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInvokeInvalidData() {
        /* input data size: 3 * 224 * 224 */
        TensorsData data = new TensorsData();
        data.addTensorData(TensorsData.allocateByteBuffer(100));

        try {
            mSingle.invoke(data);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetZeroTimeout() {
        try {
            mSingle.setTimeout(0);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidTimeout() {
        try {
            mSingle.setTimeout(-1);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }
}
