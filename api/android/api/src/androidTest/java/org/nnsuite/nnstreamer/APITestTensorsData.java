package org.nnsuite.nnstreamer;

import android.support.test.runner.AndroidJUnit4;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.*;

/**
 * Testcases for TensorsData.
 */
@RunWith(AndroidJUnit4.class)
public class APITestTensorsData {
    private TensorsData mData;

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        TensorsInfo info = new TensorsInfo();

        info.addTensorInfo(NNStreamer.TENSOR_TYPE_UINT8, new int[]{100});
        info.addTensorInfo(NNStreamer.TENSOR_TYPE_UINT8, new int[]{200});
        info.addTensorInfo(NNStreamer.TENSOR_TYPE_UINT8, new int[]{300});

        mData = TensorsData.allocate(info);
    }

    @After
    public void tearDown() {
        mData.close();
    }

    @Test
    public void testAllocateByteBuffer() {
        try {
            ByteBuffer buffer = TensorsData.allocateByteBuffer(300);

            assertTrue(APITestCommon.isValidBuffer(buffer, 300));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testAllocate() {
        try {
            TensorsInfo info = new TensorsInfo();

            info.addTensorInfo(NNStreamer.TENSOR_TYPE_INT16, new int[]{2});
            info.addTensorInfo(NNStreamer.TENSOR_TYPE_UINT16, new int[]{2,2});
            info.addTensorInfo(NNStreamer.TENSOR_TYPE_UINT32, new int[]{2,2,2});

            TensorsData data = TensorsData.allocate(info);

            /* index 0: 2 int16 */
            assertTrue(APITestCommon.isValidBuffer(data.getTensorData(0), 4));

            /* index 1: 2:2 uint16 */
            assertTrue(APITestCommon.isValidBuffer(data.getTensorData(1), 8));

            /* index 2: 2:2:2 uint32 */
            assertTrue(APITestCommon.isValidBuffer(data.getTensorData(2), 32));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testAllocateNullInfo() {
        try {
            TensorsData.allocate(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetData() {
        try {
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(0), 100));
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(1), 200));
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(2), 300));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSetData() {
        try {
            ByteBuffer buffer = TensorsData.allocateByteBuffer(200);
            mData.setTensorData(1, buffer);

            assertEquals(3, mData.getTensorsCount());
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(0), 100));
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(1), 200));
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(2), 300));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSetNullByteBuffer() {
        try {
            ByteBuffer buffer = null;

            mData.setTensorData(0, buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidOrderByteBuffer() {
        try {
            /* big-endian byte order */
            ByteBuffer buffer = ByteBuffer.allocateDirect(100);

            mData.setTensorData(0, buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetNonDirectByteBuffer() {
        try {
            /* non-direct byte buffer */
            ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.nativeOrder());

            mData.setTensorData(0, buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetInvalidIndex() {
        try {
            mData.getTensorData(5);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidIndex() {
        try {
            ByteBuffer buffer = TensorsData.allocateByteBuffer(500);

            mData.setTensorData(5, buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidSizeByteBuffer() {
        try {
            ByteBuffer buffer = TensorsData.allocateByteBuffer(500);

            mData.setTensorData(1, buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAllocateInvalidSize() {
        try {
            TensorsData.allocateByteBuffer(-1);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAllocateZeroSize() {
        try {
            TensorsData.allocateByteBuffer(0);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }
}
