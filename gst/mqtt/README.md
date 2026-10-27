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
- A malformed message (shorter than its header, declaring more memory blocks than its header holds, or a memory block larger than the rest of its payload) is dropped with a warning in the debug log, and the stream goes on with the next message. A lost connection still stops the stream with an error.

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

- `client-id`: if it is not set, a random one is generated when the element starts: `nnssink` or `nnssrc` followed by 16 hexadecimal digits. It used to be `<host>_<pid>_sink<n>` (`_src<n>` for mqttsrc); update broker rules written for that form. It is not derived from the host name or the process ID, so a remote party cannot guess it and disconnect the element by connecting with the same ID. The random bits come from the operating system (`getrandom ()` on Linux, or `/dev/urandom`); if neither is available, the element refuses to start unless `client-id` is set. On a Linux kernel older than 5.18, an element started before the kernel's random pool is initialized (early boot) reads `/dev/urandom` that may not be seeded yet; set `client-id` for such boot-time services. If `pub-topic` of mqttsink is not set, it is `<client-id>/topic`, which tells the ID to whoever may subscribe to it: set `pub-topic`, or restrict the subscriptions with `username`/`password` and the broker's ACL, to keep the ID private.
- `username` and `password`: if they are not set (default), the element connects without authentication. `password` needs `username`; the element refuses to start with a password only. A broker that refuses them fails the start like an unreachable broker, after the connection timeout (5 seconds). They are sent as they are; enable TLS to encrypt them, or use a trusted network or a tunnel for plain TCP. `password` cannot be read back. Prefer setting it from the application to giving it on a `gst-launch-1.0` command line, which other local users can see.

## TLS

TLS is optional at build time and disabled by default. To build it, provide
Paho's SSL-enabled asynchronous library (`libpaho-mqtt3as`) and configure:

```sh
meson setup build -Dmqtt-support=enabled -Dmqtt-tls=true
```

A build without `mqtt-tls` keeps its existing Paho dependency and does not
add an OpenSSL dependency for MQTT. TLS builds link the SSL-enabled library;
`mqtt-tls=true` with `mqtt-support=disabled` is rejected. Cross builds for
Tizen or Android must provide that library and its headers in the target
sysroot; native Ubuntu validation does not verify those packaging environments.

Both elements expose:

- `tls`: opt in to TLS (default `false`). The server's certificate chain and
  hostname are always verified. Set `host` to a bare hostname; TLS URI schemes
  in `host` cannot be used to bypass this setting.
- `tls-ca-file`: a PEM CA trust store, or `NULL` to use the TLS library's
  default trust store.
- `tls-cert-file`: a PEM client certificate chain, for mutual TLS.
- `tls-key-file`: its PEM private key. It requires `tls-cert-file`; a key
  bundled into the certificate file does not need a separate key path.
- `tls-key-password`: a write-only password for an encrypted client key.

The host and port defaults remain unchanged. Set the broker's TLS port
explicitly, for example:

```sh
gst-launch-1.0 videotestsrc num-buffers=10 ! mqttsink tls=true host=broker.example.com port=8883 tls-ca-file=/path/to/ca.pem pub-topic=test/video
gst-launch-1.0 mqttsrc tls=true host=broker.example.com port=8883 tls-ca-file=/path/to/ca.pem sub-topic=test/video ! fakesink
```

TLS properties take effect when the element next starts. Settings passed to
Paho are separate copies retained for reconnect, so updating a property on
an active element does not invalidate its connection options. Set key
passwords from the application rather than a command line visible to other
local users. Unreadable or invalid PEM contents, an untrusted certificate,
or a hostname mismatch fail the connection; verification cannot be disabled.
A non-TLS build reports a settings error when `tls=true` is requested.

## MQTT Implementation

Use the mqtt implementation already available in Tizen.org (/platform/upstream/paho-mqtt-c).
