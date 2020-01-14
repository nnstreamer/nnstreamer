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
import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Testcases for Pipeline.
 */
@RunWith(AndroidJUnit4.class)
public class APITestPipeline {
    private int mReceived = 0;
    private boolean mInvalidState = false;
    private Pipeline.State mPipelineState = Pipeline.State.NULL;

    private Pipeline.NewDataCallback mSinkCb = new Pipeline.NewDataCallback() {
        @Override
        public void onNewDataReceived(TensorsData data) {
            if (data == null ||
                data.getTensorsCount() != 1 ||
                data.getTensorData(0).capacity() != 200) {
                mInvalidState = true;
                return;
            }

            TensorsInfo info = data.getTensorsInfo();

            /* validate received data (unit8 2:10:10:1) */
            if (info == null ||
                info.getTensorsCount() != 1 ||
                info.getTensorName(0) != null ||
                info.getTensorType(0) != NNStreamer.TensorType.UINT8 ||
                !Arrays.equals(info.getTensorDimension(0), new int[]{2,10,10,1})) {
                /* received data is invalid */
                mInvalidState = true;
            }

            mReceived++;
        }
    };

    @Rule
    public GrantPermissionRule mPermissionRule = APITestCommon.grantPermissions();

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        mReceived = 0;
        mInvalidState = false;
        mPipelineState = Pipeline.State.NULL;
    }

    @Test
    public void enumPipelineState() {
        assertEquals(Pipeline.State.UNKNOWN, Pipeline.State.valueOf("UNKNOWN"));
        assertEquals(Pipeline.State.NULL, Pipeline.State.valueOf("NULL"));
        assertEquals(Pipeline.State.READY, Pipeline.State.valueOf("READY"));
        assertEquals(Pipeline.State.PAUSED, Pipeline.State.valueOf("PAUSED"));
        assertEquals(Pipeline.State.PLAYING, Pipeline.State.valueOf("PLAYING"));
    }

    @Test
    public void testConstructInvalidElement_n() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "invalidelement ! tensor_converter ! tensor_sink";

        try {
            new Pipeline(desc);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testConstructNullDescription_n() {
        try {
            new Pipeline(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testConstructNullStateCb() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink";

        try (Pipeline pipe = new Pipeline(desc, null)) {
            Thread.sleep(100);
            assertEquals(Pipeline.State.PAUSED, pipe.getState());
            Thread.sleep(100);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testConstructWithStateCb() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink";

        /* pipeline state callback */
        Pipeline.StateChangeCallback stateCb = new Pipeline.StateChangeCallback() {
            @Override
            public void onStateChanged(Pipeline.State state) {
                mPipelineState = state;
            }
        };

        try (Pipeline pipe = new Pipeline(desc, stateCb)) {
            Thread.sleep(100);
            assertEquals(Pipeline.State.PAUSED, mPipelineState);

            /* start pipeline */
            pipe.start();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PLAYING, mPipelineState);

            /* stop pipeline */
            pipe.stop();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PAUSED, mPipelineState);
            Thread.sleep(100);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetState() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PLAYING, pipe.getState());

            /* stop pipeline */
            pipe.stop();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PAUSED, pipe.getState());
            Thread.sleep(100);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testRegisterNullDataCb_n() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback("sinkx", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDataCbInvalidName_n() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback("invalid_sink", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDataCbNullName_n() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback(null, mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterNullDataCb_n() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback("sinkx", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterDataCbNullName_n() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback(null, mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisteredDataCb_n() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRemoveDataCb() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* pause pipeline and unregister sink callback */
            Thread.sleep(100);
            pipe.stop();

            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            Thread.sleep(100);

            /* start pipeline again */
            pipe.start();

            /* push input buffer again */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testDuplicatedDataCb() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register three callbacks */
            Pipeline.NewDataCallback cb1 = new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    mReceived++;
                }
            };

            Pipeline.NewDataCallback cb2 = new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    mReceived++;
                }
            };

            pipe.registerSinkCallback("sinkx", mSinkCb);
            pipe.registerSinkCallback("sinkx", cb1);
            pipe.registerSinkCallback("sinkx", cb2);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* pause pipeline and unregister sink callback */
            Thread.sleep(100);
            pipe.stop();

            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            pipe.unregisterSinkCallback("sinkx", cb1);
            Thread.sleep(100);

            /* start pipeline again */
            pipe.start();

            /* push input buffer again */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(40, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testRunModel() {
        File model = APITestCommon.getTestModel();
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)3:224:224:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=tensorflow-lite model=" + model.getAbsolutePath() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{3,224,224,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    TensorsInfo info = data.getTensorsInfo();

                    if (info == null || info.getTensorsCount() != 1) {
                        mInvalidState = true;
                    } else {
                        ByteBuffer output = data.getTensorData(0);

                        if (!APITestCommon.isValidBuffer(output, 1001)) {
                            mInvalidState = true;
                        }
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(100);
            }

            /* sleep 500 to invoke */
            Thread.sleep(500);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInputBuffer() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer repeatedly */
            for (int i = 0; i < 2048; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
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
    public void testInputInvalidName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* start pipeline */
            pipe.start();

            pipe.inputData("invalid_src", TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputNullName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* start pipeline */
            pipe.start();

            pipe.inputData(null, TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputNullData_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            pipe.inputData("srcx", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectSwitch() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(50);

                if (i == 9) {
                    /* select pad */
                    pipe.selectSwitchPad("outs", "src_1");
                }
            }

            /* sleep 300 to pass all input buffers to sink */
            Thread.sleep(300);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetSwitchPad() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* get pad list of output-selector */
            String[] pads = pipe.getSwitchPads("outs");

            assertEquals(2, pads.length);
            assertEquals("src_0", pads[0]);
            assertEquals("src_1", pads[1]);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetSwitchInvalidName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* get pad list with invalid switch name */
            pipe.getSwitchPads("invalid_outs");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetSwitchNullName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* get pad list with null param */
            pipe.getSwitchPads(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectInvalidPad_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* select invalid pad name */
            pipe.selectSwitchPad("outs", "invalid_src");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectNullPad_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* null pad name */
            pipe.selectSwitchPad("outs", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectNullSwitchName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* null switch name */
            pipe.selectSwitchPad(null, "src_1");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testControlValve() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tee name=t " +
                "t. ! queue ! tensor_sink " +
                "t. ! queue ! valve name=valvex ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);

                if (i == 9) {
                    /* close valve */
                    pipe.controlValve("valvex", false);
                }
            }

            /* sleep 300 to pass all input buffers to sink */
            Thread.sleep(300);

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testControlInvalidValve_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tee name=t " +
                "t. ! queue ! tensor_sink " +
                "t. ! queue ! valve name=valvex ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* control valve with invalid name */
            pipe.controlValve("invalid_valve", false);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testControlNullValveName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tee name=t " +
                "t. ! queue ! tensor_sink " +
                "t. ! queue ! valve name=valvex ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* control valve with invalid name */
            pipe.controlValve(null, false);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAMCsrc() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        String media = root + "/nnstreamer/test/test_video.mp4";

        String desc = "amcsrc location=" + media + " ! " +
                "videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240 ! " +
                "tensor_converter ! tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    TensorsInfo info = data.getTensorsInfo();

                    if (info == null || info.getTensorsCount() != 1) {
                        mInvalidState = true;
                    } else {
                        ByteBuffer output = data.getTensorData(0);

                        if (!APITestCommon.isValidBuffer(output, 230400)) {
                            mInvalidState = true;
                        }
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* sleep 2 seconds to invoke */
            Thread.sleep(2000);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);

            /* sleep 1 second and restart */
            Thread.sleep(1000);
            mReceived = 0;

            pipe.start();

            /* sleep 2 seconds to invoke */
            Thread.sleep(2000);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }
}
