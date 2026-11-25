from ultralytics import YOLO


def main():
    model = YOLO(r"D:\\Do_An\\Fire_Smoke_develop\\runs\\train\\fire_smoke_V3\\fire_yolo26n\\weights\\best.pt")
    print(model.names)

    result = model.predict(
        source=r"D:\\Do_An\\Fire_Smoke_develop\\datasets\\3classes\\train\\images\\large_-13-_jpg.rf.c81f2e1243899ffa9a6a155e7eaa138d.jpg",
        imgsz=640,
        conf=0.05,
        classes=None,
        save=True,
    )[0]

    for box in result.boxes:
        class_id = int(box.cls.item())
        print(model.names[class_id], round(box.conf.item(), 3))

    model.val(
        data=r"D:\\Do_An\\Fire_Smoke_develop\\datasets\\Data_V3_2classes\\data.yaml",
        split="val",
        imgsz=640,
        verbose=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
