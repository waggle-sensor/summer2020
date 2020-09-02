FROM waggle/plugin-base:0.1.0

COPY *.py requirements.txt /app/
COPY analysis /app/analysis
COPY config /app/config
COPY retrain /app/retrain
COPY yolov3 /app/yolov3

WORKDIR /app
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/usr/bin/python3", "/app/"]