FROM tensorflow/tensorflow:2.11.0-gpu

# COPY training /vp/src

WORKDIR /usr/src/app
COPY . .

RUN apt install htop
RUN pip install --upgrade pip
# RUN /usr/bin/python3 -m pip install --upgrade pip
# RUN pip install -r src/requirements.txt --no-cache-dir
RUN pip install -r src/requirements.txt

USER 1014:1015
# USER 197609:197121