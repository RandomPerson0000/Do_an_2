import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone


TOPIC = "FS_nano_V2"


def device_name(value):
    if not re.fullmatch(r"[\w.-]{1,80}", value, re.ASCII) or value in (".", ".."):
        raise ValueError("FS_DEVICE_ID must be a simple device name, e.g. pi5-01")
    return value


def mqtt_client(client_id="", persistent=False):
    import paho.mqtt.client as mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                         clean_session=not persistent, protocol=mqtt.MQTTv311)
    if os.environ.get("MQTT_USERNAME"):
        client.username_pw_set(os.environ["MQTT_USERNAME"], os.environ.get("MQTT_PASSWORD"))
    if os.environ.get("MQTT_CA_FILE"):
        client.tls_set(ca_certs=os.environ["MQTT_CA_FILE"])
    client.enable_logger()
    return client


def start_client(client):
    tls = bool(os.environ.get("MQTT_CA_FILE"))
    client.connect_async(os.getenv("MQTT_HOST", "127.0.0.1"),
                         int(os.getenv("MQTT_PORT", "8883" if tls else "1883")))
    client.loop_start()


class AlarmPublisher:
    """Persist each transition until the QoS 1 broker acknowledgement."""

    def __init__(self, directory, device, backend):
        directory.mkdir(parents=True, exist_ok=True)
        self.db = directory / "mqtt_outbox.sqlite3"
        self.device = device
        self.backend = backend
        self.topic = f"{TOPIC}/{device}/{backend}/alerts"
        with closing(sqlite3.connect(self.db)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS outbox (payload TEXT NOT NULL)")
        self.client = mqtt_client()
        self.stop = threading.Event()
        start_client(self.client)
        self.worker = threading.Thread(target=self._send, daemon=True)
        self.worker.start()

    def enqueue(self, kind, label, confidence):
        event = dict(event_id=uuid.uuid4().hex, version=TOPIC, device=self.device,
                     backend=self.backend, utc=datetime.now(timezone.utc).isoformat(),
                     kind=kind, label=label, confidence=confidence)
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("INSERT INTO outbox VALUES (?)", (json.dumps(event),))

    def _send(self):
        with closing(sqlite3.connect(self.db)) as db:
            while not self.stop.is_set():
                row = db.execute("SELECT rowid, payload FROM outbox ORDER BY rowid LIMIT 1").fetchone()
                if row and self.client.is_connected():
                    info = self.client.publish(self.topic, row[1], qos=1, retain=False)
                    if info.rc == 0:
                        while not info.is_published() and not self.stop.wait(0.1):
                            pass
                        if info.is_published():
                            with db:
                                db.execute("DELETE FROM outbox WHERE rowid = ?", (row[0],))
                            continue
                self.stop.wait(0.25)

    def close(self):
        # Give a healthy connection time to flush the final video frame.
        with closing(sqlite3.connect(self.db)) as db:
            for _ in range(20):
                if not db.execute("SELECT 1 FROM outbox LIMIT 1").fetchone():
                    break
                self.stop.wait(0.1)
            else:
                logging.warning("Pending MQTT alerts saved in %s for the next run", self.db)
        self.stop.set()
        self.worker.join()
        self.client.disconnect()
        self.client.loop_stop()
