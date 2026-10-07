# GstMQTT: MQTT GStreamer pub(sink)/sub(src)

GstMQTT is a subproject of nnstreamer for Edge-AI project.

## Elements

GstMQTT elements are independent GStreamer elements that do not depend on other NNStreamer modules or ```other/tensor(s)``` types.

The specifics will evolve as the corresponding developers' understangins of MQTT evolve.

### mqttsink

- Accepts "ANY". Users are supposed to designate the capability with caps-filter as it may be used to find a corresponding mqttsrc.

### mqttsrc

- Provides "ANY". Users are supposed to designate the capability with caps-filter as it may be used to find a corresponding mqttsink.
- `max-buffers` bounds the messages received from the broker but not yet pushed. If the publisher sends faster than the pipeline processes, the oldest message is dropped. It is 0 (no limit) by default. Set it if the publisher or the broker is not trusted: without a limit, a publisher that keeps sending can exhaust the memory, as every waiting message is kept whole. A limit set while more messages are waiting takes effect when the next message arrives. A dropped message is reported in the debug log only; the buffers after it are not flagged.

## Usage Example

Before using the GstMQTT elements, make sure that the MQTT broker runs on the local/remote machine.
In the following example, the broker's URL is ```127.0.0.1:1883```, the GstMQTT elements' default value.
To specify it according to your environment, use ```host``` and ```port``` properties.

```bash
$ gst-launch-1.0 videotestsrc is-live=true ! video/x-raw,format=RGB,width=640,height=480,framerate=5/1 ! mqttsink pub-topic=test/videotestsrc
...
```

```bash
$ gst-launch-1.0 mqttsrc sub-topic=test/videotestsrc ! video/x-raw,format=RGB,width=640,height=480,framerate=5/1 ! videoconvert ! ximagesink
...
```

With "commented capabilities", the capability may become:

```bash
video/x-raw,format=RGB,width=640,height=480,framerate=5/1,extra=YOURSTRINGMESSAGEFOREDGEAICAPS
```

, then the key-value pair of ```extra``` is only used to pair mqttsink and mqttsrc, which should be ignored by the reset (i.e., ```videotestsrc```, ```videoconvert```, and ```ximagesink``` in the example).

We need further investigation on how to realize this: we may need to either

1. add another "caps-filter-for-mqtt" so that this ```extra``` caps are removed for the rest, or
2. express such extra capabilities directly in mqttsrc/mqttsink elements as their properties.

## MQTT Implementation

Use the mqtt implementation already available in Tizen.org (/platform/upstream/paho-mqtt-c).
