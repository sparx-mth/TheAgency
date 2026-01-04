# WebSocket Virtual Controller Client

A Python script that sends and receives virtual controller data via WebSocket at a configurable rate. Supports send-only, listen-only, and bidirectional communication modes.

## Installation

### 1. Activate the Virtual Environment

The project includes a Python virtual environment. Activate it first:

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 2. Install Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Deactivate Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## Usage

### Basic Usage

Run with default settings (localhost:8765 at 10 Hz):

```bash
python automation.py
```

### Custom Configuration

Configure the server IP, port, and frequency:

```bash
python automation.py --host 192.168.1.100 --port 9000 --frequency 20
```

### Command-Line Arguments

- `--host` : WebSocket server IP address or hostname (default: localhost)
- `--port` : WebSocket server port (default: 8765)
- `--frequency` : Message transmission frequency in Hz (default: 10.0)
- `--random` : Generate random button/axis values instead of fixed values
- `--mode` : Operation mode - `send` (send only), `listen` (receive only), or `both` (bidirectional) (default: send)

### Examples

**Send messages at 30 Hz to a remote server:**
```bash
python automation.py --host 192.168.1.50 --port 8080 --frequency 30
```

**Send random data at 5 Hz:**
```bash
python automation.py --frequency 5 --random
```

**Listen for incoming messages only:**
```bash
python automation.py --mode listen
```

**Send and listen simultaneously (bidirectional):**
```bash
python automation.py --mode both --frequency 20
```

**Connect to a specific server in bidirectional mode:**
```bash
python automation.py --host 10.0.0.100 --port 3000 --mode both
```

## Message Schema

The script sends JSON messages with the following structure:

```json
{
    "header": {
        "timestamp": "2025-06-29T13:01:00Z",
        "command": "VIRTUAL_CONTROLLER"
    },
    "content": {
        "robot_uid": "drn12345678",
        "pilot_station_uid": "gcu12345678",
        "user_uid": "user12345",
        "type": 1,
        "buttons": [0, 1, 1, 0, 0, 0],
        "axes": [500, 250, 1000, 200, 300]
    }
}
```

### Data Constraints

- **Buttons**: Each value is either `0` (off) or `1` (on)
  - Index 0: Switch
  - Index 1: Side
  - Index 2: A
  - Index 3: B
  - Index 4: C
  - Index 5: Joystick

- **Axes**: Each value is between `-1000` and `1000`
  - Index 0: Joystick Horizontal
  - Index 1: Joystick Vertical
  - Index 2: Trigger
  - Index 3: Marker Horizontal
  - Index 4: Marker Vertical

## Testing

A test WebSocket server is included for testing purposes. Run it in a separate terminal:

```bash
python test_server.py
```

Or specify custom host/port:

```bash
python test_server.py --host 0.0.0.0 --port 9000
```

### Test Server Options

- `--echo` : Send acknowledgment messages back for each received message
- `--send-acks` : Send periodic status updates to clients (every 5 seconds)

**Example - Run server with echo enabled:**
```bash
python test_server.py --echo
```

**Example - Run server with periodic status updates:**
```bash
python test_server.py --send-acks
```

### Testing Bidirectional Communication

**Terminal 1 - Start server with echo and periodic updates:**
```bash
python test_server.py --echo --send-acks
```

**Terminal 2 - Run client in bidirectional mode:**
```bash
python automation.py --mode both --frequency 2
```

**Terminal 3 - Run another client in listen-only mode:**
```bash
python automation.py --mode listen
```

## Stopping the Script

Press `Ctrl+C` to gracefully stop the script.

