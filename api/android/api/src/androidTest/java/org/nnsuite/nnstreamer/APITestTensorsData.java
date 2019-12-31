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

        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{100});
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{200});
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{300});

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

            info.addTensorInfo(NNStreamer.TensorType.INT16, new int[]{2});
            info.addTensorInfo(NNStreamer.TensorType.UINT16, new int[]{2,2});
            info.addTensorInfo(NNStreamer.TensorType.UINT32, new int[]{2,2,2});

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
    public void testAllocateNullInfo_n() {
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
    public void testSetNullByteBuffer_n() {
        try {
            ByteBuffer buffer = null;

            mData.setTensorData(0, buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidOrderByteBuffer_n() {
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
    public void testSetNonDirectByteBuffer_n() {
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
    public void testGetInvalidIndex_n() {
        try {
            mData.getTensorData(5);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidIndex_n() {
        try {
            ByteBuffer buffer = TensorsData.allocateByteBuffer(500);

            mData.setTensorData(5, buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidSizeByteBuffer_n() {
        try {
            ByteBuffer buffer = TensorsData.allocateByteBuffer(500);

            mData.setTensorData(1, buffer);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAllocateInvalidSize_n() {
        try {
            TensorsData.allocateByteBuffer(-1);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAllocateZeroSize_n() {
        try {
            TensorsData.allocateByteBuffer(0);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetInfo() {
        try {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo("name1", NNStreamer.TensorType.INT64, new int[]{10,1,1,1});
            info.addTensorInfo("name2", NNStreamer.TensorType.UINT64, new int[]{20,1,1,1});

            /* allocate data, info is cloned */
            TensorsData data = TensorsData.allocate(info);

            /* update info */
            info.setTensorName(0, "test1");
            info.setTensorType(0, NNStreamer.TensorType.INT16);
            info.setTensorDimension(0, new int[]{1,1,1,1});

            info.setTensorName(1, "test2");
            info.setTensorType(1, NNStreamer.TensorType.UINT16);
            info.setTensorDimension(1, new int[]{2,2,1,1});

            info.addTensorInfo("test3", NNStreamer.TensorType.FLOAT64, new int[]{3,3,3,1});

            assertEquals(3, info.getTensorsCount());

            /* check cloned info */
            TensorsInfo cloned = data.getTensorsInfo();

            assertEquals(2, cloned.getTensorsCount());

            assertEquals("name1", cloned.getTensorName(0));
            assertEquals(NNStreamer.TensorType.INT64, cloned.getTensorType(0));
            assertArrayEquals(new int[]{10,1,1,1}, cloned.getTensorDimension(0));

            assertEquals("name2", cloned.getTensorName(1));
            assertEquals(NNStreamer.TensorType.UINT64, cloned.getTensorType(1));
            assertArrayEquals(new int[]{20,1,1,1}, cloned.getTensorDimension(1));
        } catch (Exception e) {
            fail();
        }
    }
}
