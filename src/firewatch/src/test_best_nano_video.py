"""Run best_nano.onnx on a local video and publish FireWatch alarms to MQTT."""
import argparse
import json
import logging
import os
from pathlib import Path
from threading import Event


CONNECT_TIMEOUT = 10
PUBLISH_TIMEOUT = 5


def choose_video():
    import tkinter as tk
    from tkinter import filedialog

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("Cannot open the file picker. Pass --video PATH.") from exc
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            parent=root,
            title="Chọn video để kiểm thử FireWatch",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm"), ("All files", "*")],
        )
    finally:
        root.destroy()


def run_with_mqtt(cfg, args):
    from paho.mqtt import client as mqtt

    from firewatch.app import run
    from firewatch.capture import VideoReader

    connack = Event()
    connection_error = None
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, reconnect_on_failure=False)
    client.connect_timeout = CONNECT_TIMEOUT

    def on_connect(_client, _userdata, _flags, reason, _properties):
        nonlocal connection_error
        connection_error = str(reason) if reason.is_failure else None
        connack.set()

    def publish_alarm(record):
        payload = {
            "test": True,
            "model": Path(cfg.model).name,
            "video": Path(cfg.source).name,
            **record,
        }
        info = client.publish(args.mqtt_topic, json.dumps(payload, ensure_ascii=False),
                              qos=1, retain=False)
        info.wait_for_publish(timeout=PUBLISH_TIMEOUT)
        if not info.is_published():
            raise TimeoutError(f"MQTT did not acknowledge {record['kind']} for {record['label']}")
        logging.info("MQTT acknowledged %s %s", record["kind"], record["label"])

    client.on_connect = on_connect
    if args.mqtt_username:
        client.username_pw_set(args.mqtt_username, os.getenv("FIREWATCH_MQTT_PASSWORD"))
    if args.mqtt_tls:
        client.tls_set(ca_certs=args.mqtt_ca)

    port = args.mqtt_port or (8883 if args.mqtt_tls else 1883)
    logging.info("Connecting to MQTT %s:%s on topic %s", args.mqtt_host, port, args.mqtt_topic)
    try:
        client.connect(args.mqtt_host, port)
        client.loop_start()
        if not connack.wait(CONNECT_TIMEOUT):
            raise TimeoutError("MQTT broker did not acknowledge the connection")
        if connection_error is not None:
            raise RuntimeError(f"MQTT broker rejected the connection: {connection_error}")
        return run(cfg, source_factory=lambda config: VideoReader(config.source),
                   on_alarm=publish_alarm)
    finally:
        try:
            client.disconnect()
        finally:
            client.loop_stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Local video path. Omit to open a file picker.")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, help="Defaults to 1883, or 8883 with TLS.")
    parser.add_argument("--mqtt-topic", default="firewatch/test/alerts")
    parser.add_argument("--mqtt-username", help="Set the password in FIREWATCH_MQTT_PASSWORD.")
    parser.add_argument("--mqtt-tls", action="store_true")
    parser.add_argument("--mqtt-ca", help="CA certificate path for a private TLS broker.")
    parser.add_argument("--headless", action="store_true", help="Disable the video preview.")
    args = parser.parse_args(argv)
    if args.mqtt_port is not None and not 1 <= args.mqtt_port <= 65535:
        parser.error("--mqtt-port must be between 1 and 65535")
    if args.mqtt_ca and not args.mqtt_tls:
        parser.error("--mqtt-ca requires --mqtt-tls")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        video = args.video or choose_video()
        if not video:
            logging.info("Video selection cancelled.")
            return 0
        video = Path(video).resolve(strict=True)
        if not video.is_file():
            raise ValueError(f"Expected a local video file: {video}")

        from firewatch.config import load_config

        script = Path(__file__).resolve()
        project_root = script.parents[3]
        cfg = load_config(script.parents[1] / "configs" / "firewatch.yaml")
        model = project_root / "models" / "released" / "best_nano.onnx"
        if not model.is_file():
            raise FileNotFoundError(f"Model not found: {model}")
        cfg.project_root = str(project_root)
        cfg.model = str(model)
        cfg.source = str(video)
        cfg.output_dir = str(project_root / "runs" / "firewatch_video_mqtt")
        cfg.display = cfg.display and not args.headless
        return run_with_mqtt(cfg, args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
