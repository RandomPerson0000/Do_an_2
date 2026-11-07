def annotate(frame, detections, active_labels, fps):
    import cv2
    image = frame.copy()
    for d in detections:
        color = (0, 80, 255) if d.label == "fire" else (0, 210, 255)
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{d.label} {d.confidence:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    status = "ALARM: " + ", ".join(active_labels) if active_labels else "MONITORING"
    color = (0, 0, 255) if active_labels else (0, 220, 0)
    cv2.putText(image, f"{status} | {fps:.1f} FPS", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return image
