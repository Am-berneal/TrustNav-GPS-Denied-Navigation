FROM python:3.11-slim
WORKDIR /app
COPY src/ src/
# pure Python 3 standard library: nothing to install
# docker run --rm -v /path/to/folder:/data trustnav /data/degraded_imu.csv /data/estimated_trajectory.txt
ENTRYPOINT ["python", "src/main.py"]
