# Person Detection with RT-DETR and Telegram Notification

This project uses a webcam to detect people in real-time using the RT-DETR object detection model. When a person is detected, a photo is sent to a specified Telegram chat.

## Features

- Real-time person detection using [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50)
- Sends detected frames to Telegram via bot
- Configurable cooldown between notifications

## Requirements

- Python 3.8+
- [transformers](https://pypi.org/project/transformers/)
- [torch](https://pypi.org/project/torch/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [requests](https://pypi.org/project/requests/)

Install dependencies:

```sh
pip install transformers torch opencv-python requests
```

## Configuration

Create a `config.json` file in the project root:

```json
{
  "telegram_token": "YOUR_TELEGRAM_BOT_TOKEN",
  "telegram_chat_id": "YOUR_TELEGRAM_CHAT_ID"
}
```

## Usage

Run the main detection script:

```sh
python web_cam_person_detection.py
```

Press `q` to quit.

## Files

- [`web_cam_person_detection.py`](web_cam_person_detection.py): Main detection script
- [`config.json`](config.json): Telegram bot configuration
- [`detected.jpg`](detected.jpg): Last detected frame (overwritten each time)
