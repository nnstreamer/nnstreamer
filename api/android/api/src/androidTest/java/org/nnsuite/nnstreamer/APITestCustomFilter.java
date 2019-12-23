package org.nnsuite.nnstreamer;

import android.support.test.runner.AndroidJUnit4;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.nio.ByteBuffer;

import static org.junit.Assert.*;

/**
 * Testcases for CustomFilter.
 */
@RunWith(AndroidJUnit4.class)
public class APITestCustomFilter {
    private int mReceived = 0;
    private boolean mInvalidState = false;
    private boolean mRegistered = false;
    private CustomFilter mCustomPassthrough;
    private CustomFilter mCustomConvert;
    private CustomFilter mCustomAdd;

    private Pipeline.NewDataCallback mSinkCb = new Pipeline.NewDataCallback() {
        @Override
        public void onNewDataReceived(TensorsData data) {
            if (data == null || data.getTensorsCount() != 1) {
                mInvalidState = true;
                return;
            }

            TensorsInfo info = data.getTensorsInfo();

            if (info == null || info.getTensorsCount() != 1) {
                mInvalidState = true;
                return;
            }

            ByteBuffer output = data.getTensorData(0);

            for (int i = 0; i < 10; i++) {
                float expected = i + 1.5f;

                if (expected != output.getFloat(i * 4)) {
                    mInvalidState = true;
                }
            }

            mReceived++;
        }
    };

    private void registerCustomFilters() {
        try {
            /* register custom-filter (passthrough) */
            mCustomPassthrough = CustomFilter.registerCustomFilter("custom-passthrough",
                new CustomFilter.CustomFilterCallback() {
                    @Override
                    public TensorsInfo getOutputInfo(TensorsInfo in) {
                        return in;
                    }

                    @Override
                    public TensorsData invoke(TensorsData in) {
                        return in;
                    }
                });

            /* register custom-filter (convert data type to float) */
            mCustomConvert = CustomFilter.registerCustomFilter("custom-convert",
                new CustomFilter.CustomFilterCallback() {
                    @Override
                    public TensorsInfo getOutputInfo(TensorsInfo in) {
                        in.setTensorType(0, NNStreamer.TensorType.FLOAT32);
                        return in;
                    }

                    @Override
                    public TensorsData invoke(TensorsData in) {
                        TensorsInfo info = in.getTensorsInfo();
                        ByteBuffer input = in.getTensorData(0);

                        info.setTensorType(0, NNStreamer.TensorType.FLOAT32);

                        TensorsData out = info.allocate();
                        ByteBuffer output = out.getTensorData(0);

                        for (int i = 0; i < 10; i++) {
                            float value = (float) input.getInt(i * 4);
                            output.putFloat(i * 4, value);
                        }

                        out.setTensorData(0, output);
                        return out;
                    }
                });

            /* register custom-filter (add constant) */
            mCustomAdd = CustomFilter.registerCustomFilter("custom-add",
                new CustomFilter.CustomFilterCallback() {
                    @Override
                    public TensorsInfo getOutputInfo(TensorsInfo in) {
                        return in;
                    }

                    @Override
                    public TensorsData invoke(TensorsData in) {
                        TensorsInfo info = in.getTensorsInfo();
                        ByteBuffer input = in.getTensorData(0);

                        TensorsData out = info.allocate();
                        ByteBuffer output = out.getTensorData(0);

                        for (int i = 0; i < 10; i++) {
                            float value = input.getFloat(i * 4);

                            /* add constant */
                            value += 1.5f;
                            output.putFloat(i * 4, value);
                        }

                        out.setTensorData(0, output);
                        return out;
                    }
                });

            mRegistered = true;
        } catch (Exception e) {
            /* failed to register custom-filters */
            fail();
        }
    }

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        mReceived = 0;
        mInvalidState = false;

        if (!mRegistered) {
            registerCustomFilters();
        }
    }

    @After
    public void tearDown() {
        if (mRegistered) {
            mCustomPassthrough.close();
            mCustomConvert.close();
            mCustomAdd.close();
        }
    }

    @Test
    public void testGetName() {
        try {
            assertEquals("custom-passthrough", mCustomPassthrough.getName());
            assertEquals("custom-convert", mCustomConvert.getName());
            assertEquals("custom-add", mCustomAdd.getName());
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testCustomFilters() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)10:1:1:1,type=(string)int32,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=" + mCustomPassthrough.getName() + " ! " +
                "tensor_filter framework=" + mCustomConvert.getName() + " ! " +
                "tensor_filter framework=" + mCustomAdd.getName() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                TensorsData in = TensorsData.allocate(info);
                ByteBuffer input = in.getTensorData(0);

                for (int j = 0; j < 10; j++) {
                    input.putInt(j * 4, j);
                }

                in.setTensorData(0, input);

                pipe.inputData("srcx", in);
                Thread.sleep(50);
            }

            /* sleep 300 to pass all input buffers to sink */
            Thread.sleep(300);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(15, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInputBuffer() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)10:1:1:1,type=(string)int32,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=" + mCustomPassthrough.getName() + " ! " +
                "tensor_filter framework=" + mCustomConvert.getName() + " ! " +
                "tensor_filter framework=" + mCustomAdd.getName() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.INT32, new int[]{10,1,1,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer repeatedly */
            for (int i = 0; i < 2048; i++) {
                TensorsData in = TensorsData.allocate(info);
                ByteBuffer input = in.getTensorData(0);

                for (int j = 0; j < 10; j++) {
                    input.putInt(j * 4, j);
                }

                in.setTensorData(0, input);

                pipe.inputData("srcx", in);
                Thread.sleep(20);
            }

            /* sleep 300 to pass input buffers to sink */
            Thread.sleep(300);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testRegisterNullName() {
        try {
            CustomFilter.registerCustomFilter(null,
                new CustomFilter.CustomFilterCallback() {
                    @Override
                    public TensorsInfo getOutputInfo(TensorsInfo in) {
                        return in;
                    }

                    @Override
                    public TensorsData invoke(TensorsData in) {
                        return in;
                    }
            });

            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterNullCallback() {
        try {
            CustomFilter.registerCustomFilter("custom-invalid-cb", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDuplicatedName() {
        try {
            CustomFilter.registerCustomFilter(mCustomPassthrough.getName(),
                new CustomFilter.CustomFilterCallback() {
                    @Override
                    public TensorsInfo getOutputInfo(TensorsInfo in) {
                        return in;
                    }

                    @Override
                    public TensorsData invoke(TensorsData in) {
                        return in;
                    }
                });

            fail();
        } catch (Exception e) {
            /* expected */
        }
    }
}
