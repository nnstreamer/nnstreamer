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
        mData = new TensorsData();
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

            mData = TensorsData.allocate(info);

            /* index 0: 2:1:1:1 int16 */
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(0), 4));

            /* index 1: 2:2:1:1 uint16 */
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(1), 8));

            /* index 0: 2:2:2:1 uint32 */
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(2), 32));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testAddData() {
        try {
            Object buffer = ByteBuffer.allocateDirect(100).order(ByteOrder.nativeOrder());

            mData.addTensorData(buffer);
            assertEquals(1, mData.getTensorsCount());

            mData.addTensorData(new byte[200]);
            assertEquals(2, mData.getTensorsCount());

            mData.addTensorData(TensorsData.allocateByteBuffer(300));
            assertEquals(3, mData.getTensorsCount());
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetData() {
        try {
            testAddData();

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
            testAddData();

            ByteBuffer buffer = TensorsData.allocateByteBuffer(500);
            mData.setTensorData(1, buffer);

            assertEquals(3, mData.getTensorsCount());
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(0), 100));
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(1), 500));
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(2), 300));
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

        assertEquals(0, mData.getTensorsCount());
    }

    @Test
    public void testAddNullByteBuffer() {
        try {
            ByteBuffer buffer = null;

            mData.addTensorData(buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mData.getTensorsCount());
    }

    @Test
    public void testAddInvalidType() {
        try {
            Object buffer = new int[8];

            mData.addTensorData(buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mData.getTensorsCount());
    }

    @Test
    public void testAddInvalidByteBuffer() {
        try {
            /* big-endian byte order */
            Object buffer = ByteBuffer.allocateDirect(100);

            mData.addTensorData(buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mData.getTensorsCount());
    }

    @Test
    public void testAddNonDirectBuffer() {
        try {
            /* non-direct byte buffer */
            Object buffer = ByteBuffer.allocate(100);

            mData.addTensorData(buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mData.getTensorsCount());
    }

    @Test
    public void testAddNullObject() {
        try {
            Object buffer = null;

            mData.addTensorData(buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mData.getTensorsCount());
    }

    @Test
    public void testAddNullByteArray() {
        try {
            byte[] buffer = null;

            mData.addTensorData(buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }

        assertEquals(0, mData.getTensorsCount());
    }

    @Test
    public void testGetInvalidIndex() {
        try {
            mData.getTensorData(0);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidIndex() {
        try {
            ByteBuffer buffer = TensorsData.allocateByteBuffer(500);

            mData.setTensorData(1, buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidByteBuffer() {
        testAddData();

        try {
            /* non-direct byte buffer */
            ByteBuffer buffer = ByteBuffer.allocate(100);

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

    @Test
    public void testAddMaxData() {
        try {
            for (int i = 0; i <= NNStreamer.TENSOR_SIZE_LIMIT; i++) {
                mData.addTensorData(TensorsData.allocateByteBuffer(10));
            }
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }
}
