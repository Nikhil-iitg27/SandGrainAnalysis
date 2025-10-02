import os
import time
import json
import socket
import numpy as np
import logging
from datetime import datetime
from gpiozero import Button
from picamera import PiCamera
import tflite_runtime.interpreter as tflite
from requests import post, get, ConnectionError

# --- CONFIGURATION ---
BUTTON_PIN = 17
MODEL_PATH = 'model.tflite'
SERVER_URL = 'http://your_server_address/api/upload'
IMAGE_DIR = '/home/pi/images/'
SD_CARD_PATH = '/media/sdcard/'
LOG_DIR = '/home/pi/logs/'
CLASS_LABELS = ['Fine sand', 'Medium sand', 'Coarse sand', 'Granule']

# --- LOGGING SETUP ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "system.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- HARDWARE INITIALIZATION ---
logger.info("Initializing hardware...")

try:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(SD_CARD_PATH, exist_ok=True)

    capture_button = Button(BUTTON_PIN)
    camera = PiCamera()
    camera.resolution = (224, 224)
    logger.info("✅ Camera and button initialized.")
except Exception as e:
    camera = None
    capture_button = None
    logger.error(f"⚠️ Hardware initialization issue: {e}")

# --- MODEL INITIALIZATION ---
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("✅ TensorFlow Lite model loaded successfully.")
except Exception as e:
    interpreter = None
    logger.error(f"❌ Failed to load TensorFlow Lite model: {e}")

# --- UTILITY FUNCTIONS ---
def get_gps_coordinates():
    """Fetch GPS coordinates using gpsd."""
    try:
        from gps import gps, WATCH_ENABLE, WATCH_NEWSTYLE
        session = gps(mode=WATCH_NEWSTYLE)
        session.stream(WATCH_ENABLE | WATCH_NEWSTYLE)
        logger.info("Fetching GPS coordinates...")
        for _ in range(10):
            report = session.next()
            if report['class'] == 'TPV' and hasattr(report, 'lat'):
                return report.lat, report.lon
            time.sleep(1)
    except Exception as e:
        logger.warning(f"⚠️ GPS fetch failed: {e}")
    return None, None

def check_internet():
    """Check internet by pinging a reliable server."""
    try:
        get("http://google.com", timeout=3)
        return True
    except Exception:
        return False

def process_image(image_path):
    """Run inference using the TFLite model."""
    if not interpreter:
        logger.error("❌ Model not available. Skipping inference.")
        return "Unknown"

    try:
        from PIL import Image
        img = Image.open(image_path).resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = int(np.argmax(output_data))
        return CLASS_LABELS[predicted_class_index]
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        return "Unknown"

def send_to_server(payload):
    """Send payload to server."""
    try:
        response = post(SERVER_URL, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("✅ Data sent to server.")
        return True
    except (ConnectionError, Exception) as e:
        logger.warning(f"⚠️ Server upload failed: {e}")
        return False

def store_data_locally(payload):
    """Save payload to SD card."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SD_CARD_PATH, f"data_{timestamp}.json")
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=4)
        logger.info(f"💾 Data stored locally: {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to store data locally: {e}")

# --- MAIN LOGIC ---
def on_button_press():
    """Triggered when button is pressed."""
    logger.info("🔘 Button pressed — starting data collection...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(IMAGE_DIR, f"image_{timestamp}.jpg")

    # 1. Capture image
    if camera:
        try:
            camera.capture(image_path)
            logger.info(f"📸 Image captured: {image_path}")
        except Exception as e:
            logger.error(f"❌ Image capture failed: {e}")
            return
    else:
        logger.warning("⚠️ No camera detected.")
        return

    # 2. Get GPS coordinates
    lat, lon = get_gps_coordinates()
    if lat is None or lon is None:
        logger.warning("⚠️ GPS unavailable, storing null coordinates.")

    # 3. Process image
    classification = process_image(image_path)
    logger.info(f"🧠 Classification result: {classification}")

    # 4. Create data payload
    data_payload = {
        'timestamp': timestamp,
        'classification': classification,
        'gps_coordinates': {'latitude': lat, 'longitude': lon}
    }

    # 5. Send or store
    if check_internet():
        if not send_to_server(data_payload):
            store_data_locally(data_payload)
    else:
        logger.info("🌐 No internet — saving locally.")
        store_data_locally(data_payload)

# --- ENTRY POINT ---
if __name__ == "__main__":
    logger.info("✅ System ready. Press the button to capture data.")
    try:
        if capture_button:
            capture_button.when_pressed = on_button_press
            from signal import pause
            pause()
        else:
            logger.error("⚠️ No button detected. Exiting.")
    except KeyboardInterrupt:
        logger.info("🛑 Program terminated by user.")
    finally:
        if camera:
            camera.close()
