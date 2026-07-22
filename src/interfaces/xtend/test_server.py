#!/usr/bin/env python3
"""
Simple WebSocket test server to receive virtual controller messages.
Use this to test the automation.py client.
Can optionally echo messages back or send periodic acknowledgments.
"""

import asyncio
import websockets
import json
import argparse
from datetime import datetime


async def handle_client(websocket, path, echo=False, send_acks=False):
    """Handle incoming WebSocket connections and messages"""
    client_address = websocket.remote_address
    print(f"✓ Client connected from {client_address[0]}:{client_address[1]}")
    
    message_count = 0
    
    # Start acknowledgment sender if enabled
    ack_task = None
    if send_acks:
        ack_task = asyncio.create_task(send_periodic_acks(websocket))
    
    try:
        async for message in websocket:
            message_count += 1
            
            # Parse and display the message
            try:
                data = json.loads(message)
                timestamp = data.get('header', {}).get('timestamp', 'N/A')
                command = data.get('header', {}).get('command', 'N/A')
                buttons = data.get('content', {}).get('buttons', [])
                axes = data.get('content', {}).get('axes', [])
                
                print(f"[{message_count:04d}] {timestamp} | CMD: {command}")
                print(f"       Buttons: {buttons}")
                print(f"       Axes: {axes}")
                print("-" * 60)
                
                # Echo message back if enabled
                if echo:
                    response = {
                        "header": {
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "command": "ACK"
                        },
                        "content": {
                            "status": "received",
                            "original_timestamp": timestamp,
                            "message_number": message_count
                        }
                    }
                    await websocket.send(json.dumps(response))
                    print(f"       → Sent ACK")
                
            except json.JSONDecodeError:
                print(f"[{message_count:04d}] Invalid JSON received")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"✗ Client {client_address[0]}:{client_address[1]} disconnected")
    finally:
        if ack_task:
            ack_task.cancel()
        print(f"Total messages received: {message_count}")


async def send_periodic_acks(websocket, interval=5):
    """Send periodic status messages to the client"""
    count = 0
    try:
        while True:
            await asyncio.sleep(interval)
            count += 1
            
            status_message = {
                "header": {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "command": "STATUS"
                },
                "content": {
                    "status": "alive",
                    "count": count,
                    "message": f"Server status update #{count}"
                }
            }
            
            await websocket.send(json.dumps(status_message))
            print(f"→ Sent periodic status update #{count}")
            
    except asyncio.CancelledError:
        pass


async def main(host, port, echo=False, send_acks=False):
    """Start the WebSocket server"""
    print(f"Starting WebSocket test server on {host}:{port}")
    print(f"Echo mode: {'Enabled' if echo else 'Disabled'}")
    print(f"Periodic status updates: {'Enabled' if send_acks else 'Disabled'}")
    print("Waiting for connections...")
    print("=" * 60)
    
    async def client_handler(websocket, path):
        await handle_client(websocket, path, echo, send_acks)
    
    async with websockets.serve(client_handler, host, port):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebSocket Test Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Echo acknowledgment for each received message"
    )
    parser.add_argument(
        "--send-acks",
        action="store_true",
        help="Send periodic status updates to clients (every 5 seconds)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.host, args.port, args.echo, args.send_acks))
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")

