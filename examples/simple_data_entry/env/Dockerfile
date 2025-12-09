FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
     && apt-get install -y \
        xvfb \        
        x11vnc \
        mutter \
        xterm \
        xdotool \
        x11-utils \
        curl \
        python3 \
        python3-tk \
        python3-dev \
        python3-xdg \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi>=0.117.1 \
    pandas>=2.3.2 \
    pillow>=11.3.0 \
    playwright>=1.47 \
    python-xlib>=0.33 \
    screeninfo>=0.8.1 \
    uvicorn>=0.37.0 \
    pyautogui>=0.9.54 \
    pynput>=1.8.1

EXPOSE 5900 8000

RUN playwright install-deps chromium

RUN useradd -ms /bin/bash ui-rl

USER ui-rl
RUN playwright install chromium

WORKDIR /home/ui-rl
COPY start.sh *.py data.csv ./

ENTRYPOINT ["./start.sh"]