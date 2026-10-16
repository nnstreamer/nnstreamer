# GstMQTT: MQTT GStreamer pub(sink)/sub(src)

GstMQTT is a subproject of nnstreamer for Edge-AI project.

## Elements

GstMQTT elements are independent GStreamer elements that do not depend on other NNStreamer modules or ```other/tensor(s)``` types.

The specifics will evolve as the corresponding developers' understangins of MQTT evolve.

### mqttsink

- Accepts "ANY". Users are supposed to designate the capability with caps-filter as it may be used to find a corresponding mqttsrc.

### mqttsrc

- Provides "ANY". Users are supposed to designate the capability with caps-filter as it may be used to find a corresponding mqttsink.

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

## Client ID and Authentication

Both elements have the following properties to identify and authenticate themselves to the broker.

- `client-id`: if it is not set, a random one is generated when the element starts: `nnssink` or `nnssrc` followed by 16 hexadecimal digits. It used to be `<host>_<pid>_sink<n>` (`_src<n>` for mqttsrc); update broker rules written for that form. It is not derived from the host name or the process ID, so a remote party cannot guess it and disconnect the element by connecting with the same ID. If `pub-topic` of mqttsink is not set, it is `<client-id>/topic`, which tells the ID to whoever may subscribe to it: set `pub-topic`, or restrict the subscriptions with `username`/`password` and the broker's ACL, to keep the ID private.
- `username` and `password`: if they are not set (default), the element connects without authentication. `password` needs `username`; the element refuses to start with a password only. A broker that refuses them fails the start like an unreachable broker, after the connection timeout (5 seconds). They are sent as they are, as the connection to the broker (`tcp://`) is not encrypted: use them on a trusted network or a tunnel. `password` cannot be read back. Prefer setting it from the application to giving it on a `gst-launch-1.0` command line, which other local users can see.

## MQTT Implementation

Use the mqtt implementation already available in Tizen.org (/platform/upstream/paho-mqtt-c).
