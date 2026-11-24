"""Local MQTT receiver, web dashboard and Telegram delivery for FS_nano_V2."""
import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from messaging.mqtt_alerts import TOPIC, device_name, mqtt_client, start_client


HERE = Path(__file__).resolve().parent
FILTER = f"{TOPIC}/+/+/alerts"
FIELDS = "event_id, device, backend, utc, kind, label, confidence"


def decode_alert(topic, payload):
    if len(payload) > 8192:
        raise ValueError("oversized message")
    event = json.loads(payload)
    if not isinstance(event, dict):
        raise ValueError("expected a JSON object")
    device = device_name(event["device"])
    backend = event["backend"]
    if (event["version"] != TOPIC or backend not in ("pt", "onnx")
            or topic != f"{TOPIC}/{device}/{backend}/alerts"):
        raise ValueError("topic and payload disagree")
    if not re.fullmatch(r"[a-f0-9]{32}", event["event_id"]):
        raise ValueError("invalid event_id")
    if event["label"] not in ("fire", "smoke") or event["kind"] not in ("alarm_start", "alarm_end"):
        raise ValueError("unknown alarm")
    if not isinstance(event["utc"], str):
        raise ValueError("invalid timestamp")
    timestamp = datetime.fromisoformat(event["utc"].replace("Z", "+00:00"))
    if timestamp.utcoffset() is None:
        raise ValueError("timestamp needs a timezone")
    event["utc"] = timestamp.astimezone(timezone.utc).isoformat(timespec="microseconds")
    score = event["confidence"]
    if score is None and event["kind"] == "alarm_end":
        return event
    if type(score) not in (int, float) or not 0 <= score <= 1:
        raise ValueError("invalid confidence")
    return event


class AlertStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS alerts (
                    event_id TEXT UNIQUE NOT NULL, device TEXT NOT NULL,
                    backend TEXT NOT NULL, utc TEXT NOT NULL, kind TEXT NOT NULL,
                    label TEXT NOT NULL, confidence REAL, telegram TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS alerts_source ON alerts(device,backend,label,utc);
                CREATE INDEX IF NOT EXISTS alerts_time ON alerts(utc);
                CREATE INDEX IF NOT EXISTS alerts_telegram ON alerts(telegram);
            """)

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.row_factory = sqlite3.Row
            yield db

    def record(self, event, telegram_enabled):
        with self.connect() as db:
            db.execute(f"""INSERT INTO alerts ({FIELDS}, telegram)
                VALUES (:event_id,:device,:backend,:utc,:kind,:label,:confidence,:telegram)
                ON CONFLICT(event_id) DO NOTHING""",
                       {**event, "telegram": "pending" if telegram_enabled else "off"})

    def snapshot(self):
        with self.connect() as db:
            db.execute("BEGIN")
            recent = db.execute(f"SELECT {FIELDS} FROM alerts ORDER BY utc DESC,rowid DESC LIMIT 100")
            history = [dict(row) for row in recent]
            active = db.execute(f"""
                SELECT {FIELDS} FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY device,backend,label ORDER BY utc DESC,rowid DESC
                    ) AS position FROM alerts
                ) WHERE position=1 AND kind='alarm_start' ORDER BY utc DESC""")
            return {"active": [dict(row) for row in active], "history": history,
                    "pending": db.execute("SELECT COUNT(*) FROM alerts WHERE telegram='pending'").fetchone()[0]}

    def pending(self):
        with self.connect() as db:
            row = db.execute(f"SELECT {FIELDS} FROM alerts WHERE telegram='pending' ORDER BY rowid LIMIT 1").fetchone()
            return dict(row) if row else None

    def delivered(self, event_id):
        with self.connect() as db:
            db.execute("UPDATE alerts SET telegram='sent' WHERE event_id=?", (event_id,))


def telegram_send(token, chat, event):
    label = "LỬA" if event["label"] == "fire" else "KHÓI"
    kind = "Bắt đầu cảnh báo" if event["kind"] == "alarm_start" else "Kết thúc cảnh báo"
    local_time = datetime.fromisoformat(event["utc"]).astimezone().strftime("%d/%m/%Y %H:%M:%S %z")
    message = f"{kind} {label}\n{event['device']} · {event['backend'].upper()} · {TOPIC}\n{local_time}"
    if event["confidence"] is not None:
        message += f"\nĐộ tin cậy {event['confidence']:.0%}"
    req = Request(f"https://api.telegram.org/bot{token}/sendMessage",
                  data=json.dumps({"chat_id": chat, "text": message}).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=8) as response:
        if not json.load(response).get("ok"):
            raise ValueError("Telegram rejected the message")


def telegram_worker(store, token, chat, stop):
    delay = 2
    while not stop.is_set():
        try:
            event = store.pending()
            if not event:
                stop.wait(1)
                continue
            telegram_send(token, chat, event)
            store.delivered(event["event_id"])
            delay = 2
            continue
        except HTTPError as exc:
            logging.warning("Telegram HTTP %s. Alert queued for retry.", exc.code)
            if exc.code == 429:
                try:
                    delay = max(delay, int(json.load(exc)["parameters"]["retry_after"]))
                except (KeyError, ValueError, TypeError):
                    pass
            exc.close()
        except (OSError, HTTPException, ValueError, sqlite3.Error) as exc:
            # Some HTTP exceptions contain the bot token in their URL.
            logging.warning("Telegram delivery pending (%s)", type(exc).__name__)
        stop.wait(delay)
        delay = min(delay * 2, 60)


def receiver(store, telegram_enabled, subscribed, failed):
    client = mqtt_client("FS_nano_V2-dashboard", persistent=True)
    client.manual_ack_set(True)

    def connected(client, userdata, flags, reason, properties):
        if reason.is_failure:
            logging.error("MQTT connection refused (%s)", reason)
        else:
            client.subscribe(FILTER, qos=1)

    def subscription(client, userdata, mid, reasons, properties):
        if reasons and all(not reason.is_failure for reason in reasons):
            subscribed.set()
            logging.info("Subscribed to %s", FILTER)
        else:
            logging.error("MQTT subscription refused for %s", FILTER)

    def received(client, userdata, message):
        try:
            event = decode_alert(message.topic, message.payload)
            store.record(event, telegram_enabled)
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
            logging.warning("Ignored invalid MQTT alert")
        except sqlite3.Error:
            logging.exception("Cannot store MQTT alert. Restart after checking the database.")
            failed.set()
            return
        if message.qos:
            client.ack(message.mid, message.qos)

    client.on_connect = connected
    client.on_subscribe = subscription
    client.on_disconnect = lambda *_: subscribed.clear()
    client.on_message = received
    return client


def dashboard_handler(store, telegram_enabled, subscribed):
    page = (HERE / "dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                body, mime = page, "text/html; charset=utf-8"
            elif path == "/api/alerts":
                data = store.snapshot()
                data.update(mqtt=subscribed.is_set(), telegram=telegram_enabled)
                body, mime = json.dumps(data).encode(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            logging.debug(fmt, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="FS_nano_V2 MQTT dashboard for the local network")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--db", type=Path, default=HERE.parents[2] / "runs/FS_nano_V2/dashboard.sqlite3")
    args = parser.parse_args()
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if bool(token) != bool(chat):
        parser.error("Set both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable Telegram")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = AlertStore(args.db)
    subscribed, failed, stop = threading.Event(), threading.Event(), threading.Event()
    client = receiver(store, bool(token), subscribed, failed)
    worker = threading.Thread(target=telegram_worker, args=(store, token, chat, stop), daemon=True)
    with ThreadingHTTPServer((args.bind, args.port), dashboard_handler(store, bool(token), subscribed)) as server:
        server.timeout = 0.5
        start_client(client)
        if token:
            worker.start()
        logging.info("Dashboard listening on %s:%s. Database: %s", *server.server_address, args.db)
        try:
            while not failed.is_set():
                server.handle_request()
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            client.disconnect()
            client.loop_stop()
            if worker.is_alive():
                worker.join(10)
    return int(failed.is_set())


