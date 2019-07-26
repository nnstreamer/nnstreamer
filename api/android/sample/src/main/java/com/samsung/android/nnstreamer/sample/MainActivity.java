package com.samsung.android.nnstreamer.sample;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;

import com.samsung.android.nnstreamer.NNStreamer;
import com.samsung.android.nnstreamer.Pipeline;
import com.samsung.android.nnstreamer.SingleShot;
import com.samsung.android.nnstreamer.TensorsData;
import com.samsung.android.nnstreamer.TensorsInfo;

import java.io.File;
import java.nio.ByteBuffer;

/**
 * Sample code to run the application with nnstreamer-api.
 * Before building this sample, copy nnstreamer-api library file into 'libs' directory.
 */
public class MainActivity extends Activity {
    private static final String TAG = "NNStreamer-Sample";

    private static final int PERMISSION_REQUEST_CODE = 3;
    private static final String[] requiredPermissions = new String[] {
            Manifest.permission.READ_EXTERNAL_STORAGE
    };

    private boolean initialized = false;
    private boolean isFailed = false;
    private CountDownTimer exampleTimer = null;
    private int exampleRun = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        /* check permissions */
        for (String permission : requiredPermissions) {
            if (!checkPermission(permission)) {
                ActivityCompat.requestPermissions(this,
                        requiredPermissions, PERMISSION_REQUEST_CODE);
                return;
            }
        }

        initNNStreamer();
    }

    @Override
    public void onResume() {
        super.onResume();

        if (initialized) {
            /* set timer to run examples */
            exampleRun = 0;
            isFailed = false;
            setExampleTimer(200);
        }
    }

    @Override
    public void onPause() {
        super.onPause();

        stopExampleTimer();
    }

    /**
     * Check the permission is granted.
     */
    private boolean checkPermission(final String permission) {
        return (ContextCompat.checkSelfPermission(this, permission)
                == PackageManager.PERMISSION_GRANTED);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        if (requestCode == PERMISSION_REQUEST_CODE) {
            for (int grant : grantResults) {
                if (grant != PackageManager.PERMISSION_GRANTED) {
                    Log.i(TAG, "Permission denied, close app.");
                    finish();
                    return;
                }
            }

            initNNStreamer();
            return;
        }

        finish();
    }

    /**
     * Initialize NNStreamer.
     */
    private void initNNStreamer() {
        if (initialized) {
            return;
        }

        try {
            initialized = NNStreamer.initialize(this);
        } catch(Exception e) {
            e.printStackTrace();
            Log.e(TAG, e.getMessage());
        } finally {
            if (initialized) {
                Log.i(TAG, "Version: " + NNStreamer.getVersion());
            } else {
                Log.e(TAG, "Failed to initialize NNStreamer");
                finish();
            }
        }
    }

    /**
     * Set timer to run the examples.
     */
    private void setExampleTimer(long time) {
        stopExampleTimer();

        exampleTimer = new CountDownTimer(time, time) {
            @Override
            public void onTick(long millisUntilFinished) {
            }

            @Override
            public void onFinish() {
                /* run the examples repeatedly */
                if (exampleRun > 15) {
                    Log.d(TAG, "Stop timer to run example");

                    if (isFailed) {
                        Log.d(TAG, "Error occurs while running the examples");
                    }

                    return;
                }

                int option = (exampleRun % 5);

                if (option == 1) {
                    Log.d(TAG, "==== Run pipeline example with state callback ====");
                    runPipe(true);
                } else if (option == 2) {
                    Log.d(TAG, "==== Run pipeline example ====");
                    runPipe(false);
                } else if (option == 3) {
                    Log.d(TAG, "==== Run pipeline example with valve ====");
                    runPipeValve();
                } else if (option == 4) {
                    Log.d(TAG, "==== Run pipeline example with switch ====");
                    runPipeSwitch();
                } else {
                    Log.d(TAG, "==== Run single-shot example ====");
                    runSingle();
                }

                exampleRun++;
                setExampleTimer(500);
            }
        };

        exampleTimer.start();
    }

    /**
     * Cancel example timer.
     */
    private void stopExampleTimer() {
        if (exampleTimer != null) {
            exampleTimer.cancel();
            exampleTimer = null;
        }
    }

    /**
     * Print tensors info.
     *
     * The data type of tensor in NNStreamer:
     * {@link NNStreamer#TENSOR_TYPE_INT32}
     * {@link NNStreamer#TENSOR_TYPE_UINT32}
     * {@link NNStreamer#TENSOR_TYPE_INT16}
     * {@link NNStreamer#TENSOR_TYPE_UINT16}
     * {@link NNStreamer#TENSOR_TYPE_INT8}
     * {@link NNStreamer#TENSOR_TYPE_UINT8}
     * {@link NNStreamer#TENSOR_TYPE_FLOAT64}
     * {@link NNStreamer#TENSOR_TYPE_FLOAT32}
     * {@link NNStreamer#TENSOR_TYPE_UNKNOWN}
     *
     * The maximum rank that NNStreamer supports.
     * {@link NNStreamer#TENSOR_RANK_LIMIT}
     *
     * The maximum number of tensor instances that tensors may have.
     * {@link NNStreamer#TENSOR_SIZE_LIMIT}
     */
    private void printTensorsInfo(TensorsInfo info) {
        int num = info.getTensorsCount();

        Log.d(TAG, "The number of tensors in info: " + num);
        for (int i = 0; i < num; i++) {
            int[] dim = info.getTesorDimension(i);

            Log.d(TAG, "Info index " + i +
                    " name: " + info.getTesorName(0) +
                    " type: " + info.getTesorType(0) +
                    " dim: " + dim[0] + ":" + dim[1] + ":" + dim[2] + ":" + dim[3]);
        }
    }

    /**
     * Print tensors data.
     *
     * The maximum number of tensor instances that tensors may have.
     * {@link NNStreamer#TENSOR_SIZE_LIMIT}
     */
    private void printTensorsData(TensorsData data) {
        int num = data.getTensorsCount();

        Log.d(TAG, "The number of tensors in data: " + num);
        for (int i = 0; i < num; i++) {
            ByteBuffer buffer = data.getTensorData(i);

            Log.d(TAG, "Data index " + i + " received " + buffer.capacity());
        }
    }

    /**
     * Example to run single-shot.
     */
    private void runSingle() {
        /* example with image classification tf-lite model */
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File model = new File(root + "/nnstreamer/tflite_model_img/mobilenet_v1_1.0_224_quant.tflite");

        if (!model.exists()) {
            Log.w(TAG, "Cannot find the model file");
            return;
        }

        try {
            SingleShot single = new SingleShot(model);

            Log.d(TAG, "Get input tensors info");
            TensorsInfo inInfo = single.getInputInfo();
            printTensorsInfo(inInfo);

            Log.d(TAG, "Get output tensors info");
            TensorsInfo outInfo = single.getOutputInfo();
            printTensorsInfo(outInfo);

            /* single-shot invoke */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                TensorsData in = TensorsData.allocate(inInfo);

                Log.d(TAG, "Try to invoke data " + (i + 1));

                TensorsData out = single.invoke(in);
                printTensorsData(out);

                Thread.sleep(50);
            }

            single.close();
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, e.getMessage());
            isFailed = true;
        }
    }

    /**
     * Example to run pipeline.
     *
     * The state of pipeline:
     * {@link NNStreamer#PIPELINE_STATE_UNKNOWN}
     * {@link NNStreamer#PIPELINE_STATE_NULL}
     * {@link NNStreamer#PIPELINE_STATE_READY}
     * {@link NNStreamer#PIPELINE_STATE_PAUSED}
     * {@link NNStreamer#PIPELINE_STATE_PLAYING}
     */
    private void runPipe(boolean addStateCb) {
        /* example with image classification tf-lite model */
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File model = new File(root + "/nnstreamer/tflite_model_img/mobilenet_v1_1.0_224_quant.tflite");

        if (!model.exists()) {
            Log.w(TAG, "Cannot find the model file");
            return;
        }

        try {
            String desc = "appsrc name=srcx ! other/tensor,dimension=(string)3:224:224:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                    "tensor_filter framework=tensorflow-lite model=" + model.getAbsolutePath() + " ! " +
                    "tensor_sink name=sinkx";

            /* pipeline state callback */
            Pipeline.StateChangeCallback stateCb = null;

            if (addStateCb) {
                stateCb = new Pipeline.StateChangeCallback() {
                    @Override
                    public void onStateChanged(int state) {
                        Log.d(TAG, "The pipeline state changed to " + state);
                    }
                };
            }

            Pipeline pipe = new Pipeline(desc, stateCb);

            /* register sink callback */
            pipe.setSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                int received = 0;

                @Override
                public void onNewDataReceived(TensorsData data, TensorsInfo info) {
                    Log.d(TAG, "Received new data callback " + (++received));

                    printTensorsInfo(info);
                    printTensorsData(data);
                }
            });

            Log.d(TAG, "Current state is " + pipe.getState());

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                TensorsData in = new TensorsData();
                in.addTensorData(TensorsData.allocateByteBuffer(3 * 224 * 224));

                Log.d(TAG, "Push input data " + (i + 1));

                pipe.inputData("srcx", in);
                Thread.sleep(50);
            }

            Log.d(TAG, "Current state is " + pipe.getState());

            pipe.close();
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, e.getMessage());
            isFailed = true;
        }
    }

    /**
     * Example to run pipeline with valve.
     */
    private void runPipeValve() {
        try {
            String desc = "appsrc name=srcx ! other/tensor,dimension=(string)3:100:100:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                    "tee name=t " +
                    "t. ! queue ! tensor_sink name=sink1 " +
                    "t. ! queue ! valve name=valvex ! tensor_sink name=sink2";

            Pipeline pipe = new Pipeline(desc);

            /* register sink callback */
            pipe.setSinkCallback("sink1", new Pipeline.NewDataCallback() {
                int received = 0;

                @Override
                public void onNewDataReceived(TensorsData data, TensorsInfo info) {
                    Log.d(TAG, "Received new data callback at sink1 " + (++received));

                    printTensorsInfo(info);
                    printTensorsData(data);
                }
            });

            pipe.setSinkCallback("sink2", new Pipeline.NewDataCallback() {
                int received = 0;

                @Override
                public void onNewDataReceived(TensorsData data, TensorsInfo info) {
                    Log.d(TAG, "Received new data callback at sink2 " + (++received));

                    printTensorsInfo(info);
                    printTensorsData(data);
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                TensorsData in = new TensorsData();
                in.addTensorData(TensorsData.allocateByteBuffer(3 * 100 * 100));

                Log.d(TAG, "Push input data " + (i + 1));

                pipe.inputData("srcx", in);
                Thread.sleep(50);

                if (i == 10) {
                    /* close valve */
                    pipe.controlValve("valvex", false);
                }
            }

            pipe.close();
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, e.getMessage());
            isFailed = true;
        }
    }

    /**
     * Example to run pipeline with output-selector.
     */
    private void runPipeSwitch() {
        try {
            /* Note that the sink element needs option 'async=false'
             *
             * Prerolling problem
             * For running the pipeline, set async=false in the sink element when using an output selector.
             * The pipeline state can be changed to paused after all sink element receive buffer.
             */
            String desc = "appsrc name=srcx ! other/tensor,dimension=(string)3:100:100:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                    "output-selector name=outs " +
                    "outs.src_0 ! tensor_sink name=sink1 async=false " +
                    "outs.src_1 ! tensor_sink name=sink2 async=false";

            Pipeline pipe = new Pipeline(desc);

            /* register sink callback */
            pipe.setSinkCallback("sink1", new Pipeline.NewDataCallback() {
                int received = 0;

                @Override
                public void onNewDataReceived(TensorsData data, TensorsInfo info) {
                    Log.d(TAG, "Received new data callback at sink1 " + (++received));

                    printTensorsInfo(info);
                    printTensorsData(data);
                }
            });

            pipe.setSinkCallback("sink2", new Pipeline.NewDataCallback() {
                int received = 0;

                @Override
                public void onNewDataReceived(TensorsData data, TensorsInfo info) {
                    Log.d(TAG, "Received new data callback at sink2 " + (++received));

                    printTensorsInfo(info);
                    printTensorsData(data);
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                TensorsData in = new TensorsData();
                in.addTensorData(TensorsData.allocateByteBuffer(3 * 100 * 100));

                Log.d(TAG, "Push input data " + (i + 1));

                pipe.inputData("srcx", in);
                Thread.sleep(50);

                if (i == 10) {
                    /* select pad */
                    pipe.selectSwitchPad("outs", "src_1");
                }
            }

            /* get pad list of output-selector */
            String[] pads = pipe.getSwitchPads("outs");
            Log.d(TAG, "Total pad in output-selector: " + pads.length);
            for (String pad : pads) {
                Log.d(TAG, "Pad name: " + pad);
            }

            pipe.close();
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, e.getMessage());
            isFailed = true;
        }
    }
}
