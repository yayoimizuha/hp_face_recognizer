FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

LABEL authors="tomokazu"

EXPOSE 8000
RUN apt update && apt install libopencv-dev curl -y --no-install-recommends
RUN pip install uvicorn["standard"] fastapi["all"] retinaface-pytorch torchvision Pillow numpy facenet-pytorch slowapi aiofiles

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" /dev/null
RUN shopt -s dotglob

RUN curl -L https://github.com/yayoimizuha/hp_face_recognizer/archive/refs/heads/master.tar.gz | tar xzf - && mv hp_face_recognizer-master/* . && rmdir hp_face_recognizer-master
RUN mkdir -p "uploaded"
COPY model.pth /workspace/model.pth

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0"]
