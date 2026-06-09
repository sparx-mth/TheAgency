# XTEND: Video (RTSP) + Telemetry (WebSocket) Connection Guide

This guide explains how to connect a laptop to an XTEND system (GCU / suitcase) and verify:
- **Telemetry** via WebSocket (`ws://192.0.0.15:8000`)
- **Video** via RTSP (`rtsp://192.0.0.15:8556/osd_snapshot`)

---

## 1) Physical Setup

1. Connect an **Ethernet cable** from your laptop to the **GCU / suitcase**.
2. Turn on the **GCU / suitcase**.
3. Make sure the **antenna is connected** (RF link stability depends on it).
4. Insert the **battery into the drone**.
5. Verify you can **see the drone on the suitcase screen** (this confirms the GCU sees the drone).

---

## 2) Network Configuration (Laptop)

The XTEND network is **static** (no DHCP).

### Set your laptop IP
Pick a free IP in the same subnet, e.g. `192.0.0.100`.

```bash
# Replace <iface> with your Ethernet interface (e.g. enx00e04c68082d)
sudo ip addr flush dev <iface>
sudo ip link set <iface> up
sudo ip addr add 192.0.0.100/24 dev <iface>
sudo ip route replace 192.0.0.0/24 dev <iface>


sudo ip addr flush dev enx00e04c680829 && \
sudo ip link set enx00e04c680829 up && \
sudo ip addr add 192.0.0.100/24 dev enx00e04c680829 && \
ping -c 1 -I enx00e04c680829 192.0.0.15
```
Verify connectivity
```bash
ping -c 2 192.0.0.15
```
Discover services (optional but useful)

```bash 
sudo nmap 192.0.0.15
sudo nmap -sV -p 8000,8556 192.0.0.15

```

Expected:

* 8000/tcp open (WebSocket)
* 8556/tcp open (RTSP, GStreamer rtspd)


## 3) Telemetry WebSocket Check
The endpoint on the GCU requires WebSocket upgrade:
```bash 
curl -v http://192.0.0.15:8000/
```
Expected response:
* HTTP/1.1 426 Upgrade Required
* Server: WebSocket++/...

## 4) Video RTSP Check (GStreamer)
Quick sanity test (no GUI):
```bash
gst-launch-1.0 -v rtspsrc location=rtsp://192.0.0.15:8556/osd_snapshot latency=0 protocols=tcp \
  ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! fakesink sync=false
```
If you want a display:
```bash 
gst-launch-1.0 rtspsrc location=rtsp://192.0.0.15:8556/osd_snapshot latency=0 protocols=tcp \
  ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```
display without overlay of specific drone XT42B drone_id = drnb177ede2
```bash 
gst-launch-1.0 -v rtspsrc location=rtsp://192.0.0.15:8510/drone_video/drnb177ede2/low latency=0 protocols=tcp   ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false 
```
to see with port and drone id are connect run 
```bash 
python3 sparx_agency/demos/Demo_No4_XTEND_MapRoom/test_video.py
``` 
return something like when GCU is available: 
```text
 /home/user/GIT/TheAgency/myenv/bin/python /home/user/GIT/TheAgency/sparx_agency/demos/Demo_No4-XTEND_MapRoom/test_video.py
connected ws://192.0.0.15:8000

=== RESPONSE GET_PILOT_STATION_VIDEO_STREAM ===
{
  "content": {
    "value": "rtsp://192.0.0.18:8510/active_drone_fpv"
  },
  "header": {
    "command": "GET_PILOT_STATION_VIDEO_STREAM",
    "timestamp": "2026-02-26T16:44:01Z"
  }
}

=== RESPONSE GET_PILOT_STATION_VIDEO_STREAM ===
{
  "content": {
    "value": "rtsp://192.0.0.18:8556/osd_snapshot"
  },
  "header": {
    "command": "GET_PILOT_STATION_VIDEO_STREAM",
    "timestamp": "2026-02-26T16:44:01Z"
  }
}

=== RESPONSE GET_ROBOT_VIDEO_STREAMS ===
{
  "content": {
    "data": [
      "rtsp://192.0.0.18:8510/drone_video/drn77f3b8f5/low",
      "rtsp://192.0.0.18:8510/drone_video/drnb177ede2/low"
    ]
  },
  "header": {
    "command": "GET_ROBOT_VIDEO_STREAMS",
    "timestamp": "2026-02-26T16:44:01Z"
  }
}
```

## 5) Combined Probe Script (Video + Telemetry)

Run the probe to print telemetry and show video:

```bash 
python3 robots/XTEND/get_xtend_probe.py \
  --host 192.0.0.15 \
  --port 8000 \
  --robot-uid drn77f3b8f5 \
  --rtsp-uri rtsp://192.0.0.15:8510/active_drone_fpv \
  --show-video
```

Notes:
* --robot-uid is the drone UID (e.g. drn77f3b8f5) shown in telemetry messages.
* If you want full raw JSON for the first few seconds:

## Troubleshooting
#### Only PILOT_STATION_* messages, no ROBOT_STATUS
* The GCU may not be linked to the drone.
* Verify the drone appears on the suitcase screen and the antenna is connected.

####  RTSP works in gst-launch but not in Python/OpenCV
* Many OpenCV builds have GStreamer: NO.
* Use the GStreamer appsink-based probe implementation (recommended).

#### Can’t ping 192.0.0.15
* Confirm you are configuring the correct Ethernet interface.
* Verify your laptop IP is 192.0.0.X/24.
* Check cable / link status:
```bash 
sudo ethtool <iface> | egrep 'Link detected|Speed|Duplex'
```