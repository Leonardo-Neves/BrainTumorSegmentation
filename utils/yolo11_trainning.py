from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r'C:\Users\leosn\Desktop\PIM\datasets\weights\yolo11s-obb.pt')
    results = model.train(data=r"C:\Users\leosn\Desktop\PIM\dataset.yaml", epochs=100, imgsz=640, device = 0, batch=16)