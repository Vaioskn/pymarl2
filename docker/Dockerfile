FROM python:3.8.18

ENV PYTHONUNBUFFERED=1

RUN apt-get update &&  \
    apt-get install --no-install-recommends -y gcc libc-dev build-essential python3-dev vim libgl1-mesa-glx libglu1-mesa xvfb swig && \
    apt-get clean autoclean && \
    apt-get autoremove --yes && \
    rm -rf /var/lib/apt/lists/*

USER root

RUN /bin/bash -c "python3 -m pip install pip==24.0"
RUN /bin/bash -c "python3 -m pip install wheel==0.38.4 setuptools==65.5.0"

WORKDIR /opt
RUN git clone https://github.com/AILabDsUnipi/pymarlzooplus.git pymarlzooplus

WORKDIR /opt/pymarlzooplus

RUN /bin/bash -c "python3 -m pip install -e ."
RUN /bin/bash -c "AutoROM -y"
CMD ["tail", "-f", "/dev/null"]


