FROM waggle/plugin-torch:1.4.0

WORKDIR /plugin

RUN apt-get update \
  && apt-get install -y \
  libjpeg8-dev \
  zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /plugin
RUN pip3 install --no-cache-dir -r /plugin/requirements.txt

COPY run.sh ./
COPY yolov3 ./

ENTRYPOINT ["/bin/bash", "run.sh"]
