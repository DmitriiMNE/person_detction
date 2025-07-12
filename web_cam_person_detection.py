import cv2
import numpy as np
import requests
import json
import time
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch


# === Загрузка конфига ===
with open('config.json') as f:
    config = json.load(f)

TELEGRAM_TOKEN = config['telegram_token']
CHAT_ID = config['telegram_chat_id']


# === Функция отправки фото в Telegram ===
def send_photo(image_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        try:
            r = requests.post(url, data={'chat_id': CHAT_ID}, files={'photo': photo})
            if r.status_code == 200:
                print("✅ Фото отправлено в Telegram")
            else:
                print(f"❌ Ошибка отправки: {r.status_code}, {r.text}")
        except Exception as e:
            print(f"Ошибка отправки фото: {e}")


# === Загрузка модели RT-DETR ===
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# === Главный цикл ===
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    last_sent = 0
    cooldown = 10  # Секунд между отправками
    frame_skip = 3
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = processor(images=image_rgb, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image_rgb.shape[:2]]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        person_detected = False

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_name = model.config.id2label[label.item()]
            if class_name.lower() != "person":
                continue

            person_detected = True

            box = box.to("cpu").numpy().astype(int)
            (x1, y1, x2, y2) = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if person_detected and time.time() - last_sent > cooldown:
            path = "detected.jpg"
            cv2.imwrite(path, frame)
            send_photo(path)
            last_sent = time.time()

        cv2.imshow('RT-DETR Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
