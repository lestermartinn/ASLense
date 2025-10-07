"""
Minimal test version of landmark service
Just outputs test JSON to verify communication works
"""
import json
import sys
import time

def send_message(msg_type, data):
    """Send a JSON message to stdout."""
    message = {
        "type": msg_type,
        "timestamp": time.time(),
        "frame": 0,
        "data": data
    }
    print(json.dumps(message), flush=True)

# Send ready message
send_message("ready", {"status": "initialized"})

# Simulate sending some landmark data
for i in range(10):
    time.sleep(0.5)
    
    # Create fake 63-element landmark array
    fake_landmarks = [0.1 * j for j in range(63)]
    
    send_message("landmarks", {
        "features": fake_landmarks,
        "count": 21
    })

# Send stats
send_message("stats", {
    "fps": 2.0,
    "frames_processed": 10
})

# Send shutdown
send_message("shutdown", {"reason": "test complete"})
