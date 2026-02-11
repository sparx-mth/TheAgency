#!/usr/bin/env python3
"""
Connectivity Test Script

Tests network connectivity between Docker container and host InternNav server.
Run this inside the Docker container to diagnose connection issues.

Usage:
    python3 test_connectivity.py
    python3 test_connectivity.py --host 172.17.0.1 --port 8087
"""

import argparse
import socket
import sys
import subprocess
import os

def test_dns_resolution(host: str) -> bool:
    """Test if hostname can be resolved."""
    print(f"\n[1] Testing DNS resolution for '{host}'...")
    try:
        ip = socket.gethostbyname(host)
        print(f"    ✓ Resolved to: {ip}")
        return True
    except socket.gaierror as e:
        print(f"    ✗ Failed: {e}")
        return False

def test_tcp_connection(host: str, port: int) -> bool:
    """Test TCP connection to host:port."""
    print(f"\n[2] Testing TCP connection to {host}:{port}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"    ✓ TCP connection successful")
            return True
        else:
            print(f"    ✗ TCP connection failed (error code: {result})")
            return False
    except Exception as e:
        print(f"    ✗ TCP connection failed: {e}")
        return False

def test_http_request(host: str, port: int, endpoint: str = "/health") -> bool:
    """Test HTTP request to server."""
    print(f"\n[3] Testing HTTP request to http://{host}:{port}{endpoint}...")
    try:
        import requests
        url = f"http://{host}:{port}{endpoint}"
        response = requests.get(url, timeout=10)
        print(f"    ✓ HTTP {response.status_code}: {response.text[:100]}")
        return response.status_code in [200, 404, 405]  # 404/405 means server is responding
    except ImportError:
        print("    ! requests library not installed, using urllib")
        try:
            import urllib.request
            url = f"http://{host}:{port}{endpoint}"
            req = urllib.request.urlopen(url, timeout=10)
            print(f"    ✓ HTTP {req.status}")
            return True
        except Exception as e:
            print(f"    ✗ HTTP request failed: {e}")
            return False
    except Exception as e:
        print(f"    ✗ HTTP request failed: {e}")
        return False

def get_docker_host_ip() -> str:
    """Try to find the Docker host IP."""
    print("\n[0] Detecting Docker host IP...")
    
    # Method 1: Check for host.docker.internal
    try:
        ip = socket.gethostbyname("host.docker.internal")
        print(f"    Found host.docker.internal: {ip}")
        return ip
    except:
        pass
    
    # Method 2: Check gateway from /proc/net/route
    try:
        with open("/proc/net/route", "r") as f:
            for line in f:
                fields = line.strip().split()
                if fields[1] != '00000000' or not int(fields[3], 16) & 2:
                    continue
                gateway = socket.inet_ntoa(bytes.fromhex(fields[2])[::-1])
                print(f"    Found gateway from route table: {gateway}")
                return gateway
    except:
        pass
    
    # Method 3: Parse ip route
    try:
        result = subprocess.run(["ip", "route"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "default" in line:
                parts = line.split()
                if "via" in parts:
                    idx = parts.index("via") + 1
                    gateway = parts[idx]
                    print(f"    Found gateway from ip route: {gateway}")
                    return gateway
    except:
        pass
    
    # Default Docker bridge
    print("    Using default Docker bridge IP: 172.17.0.1")
    return "172.17.0.1"

def test_inference_api(host: str, port: int) -> bool:
    """Test the inference API with a dummy request."""
    print(f"\n[4] Testing inference API...")
    
    try:
        import requests
        import base64
        import numpy as np
        
        # Create dummy image
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Try different image encodings
        try:
            from PIL import Image
            import io
            pil_image = Image.fromarray(dummy_image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except:
            import cv2
            _, buffer = cv2.imencode('.jpg', dummy_image)
            image_b64 = base64.b64encode(buffer).decode("utf-8")
        
        # Try different endpoints
        endpoints = ["/inference", "/v1/inference", "/predict", "/api/inference"]
        
        payload = {
            "image": image_b64,
            "instruction": "test",
            "images": [image_b64],
            "text": "test"
        }
        
        for endpoint in endpoints:
            url = f"http://{host}:{port}{endpoint}"
            try:
                response = requests.post(url, json=payload, timeout=30)
                print(f"    Endpoint {endpoint}: HTTP {response.status_code}")
                if response.status_code == 200:
                    print(f"    ✓ Inference API working at {endpoint}")
                    print(f"    Response: {response.text[:200]}")
                    return True
            except Exception as e:
                print(f"    Endpoint {endpoint}: {e}")
                
        print("    ✗ No working inference endpoint found")
        return False
        
    except ImportError as e:
        print(f"    ! Missing library: {e}")
        return False
    except Exception as e:
        print(f"    ✗ Inference test failed: {e}")
        return False

def print_network_info():
    """Print network debugging information."""
    print("\n[5] Network Information:")
    
    # Get container IP
    try:
        hostname = socket.gethostname()
        container_ip = socket.gethostbyname(hostname)
        print(f"    Container hostname: {hostname}")
        print(f"    Container IP: {container_ip}")
    except:
        pass
    
    # List network interfaces
    try:
        result = subprocess.run(["ip", "addr"], capture_output=True, text=True)
        print("    Network interfaces:")
        for line in result.stdout.split("\n"):
            if "inet " in line:
                print(f"      {line.strip()}")
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description="Test Docker-to-host connectivity")
    parser.add_argument("--host", help="Server host (auto-detect if not specified)")
    parser.add_argument("--port", type=int, default=8087, help="Server port")
    args = parser.parse_args()
    
    print("=" * 60)
    print("InternNav Bridge Connectivity Test")
    print("=" * 60)
    
    # Determine host to test
    if args.host:
        host = args.host
    else:
        host = get_docker_host_ip()
    
    port = args.port
    print(f"\nTesting connection to: {host}:{port}")
    
    results = {
        "dns": test_dns_resolution(host),
        "tcp": test_tcp_connection(host, port),
        "http": test_http_request(host, port, "/health"),
        "inference": test_inference_api(host, port)
    }
    
    print_network_info()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test.upper()}: {status}")
    
    if all_passed:
        print("\n✓ All tests passed! The bridge should work.")
        print(f"\nUpdate your config with:")
        print(f"  host: \"{host}\"")
        print(f"  port: {port}")
    else:
        print("\n✗ Some tests failed. Troubleshooting suggestions:")
        
        if not results["dns"]:
            print("  - Check if the hostname is correct")
            print("  - Try using IP address directly: 172.17.0.1")
            
        if not results["tcp"]:
            print("  - Check if InternNav server is running on host")
            print("  - Check if port 8087 is not blocked by firewall")
            print("  - Try: nc -zv {host} {port}")
            print("  - On host, check: ss -tlnp | grep 8087")
            
        if not results["http"]:
            print("  - Server may not have a /health endpoint")
            print("  - Check server logs for errors")
            
        if not results["inference"]:
            print("  - Check the inference endpoint path")
            print("  - Check server logs for the expected request format")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
