FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

LABEL authors="tomokazu"

EXPOSE 8000
RUN apt update && apt install libopencv-dev git -y --no-install-recommends
RUN pip install uvicorn["standard"] fastapi["all"] retinaface-pytorch torchvision Pillow numpy facenet-pytorch slowapi aiofiles

RUN git clone https://github.com/yayoimizuha/hp_face_recognizer.git .
COPY model.pth /workspace/model.pth

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0"]
