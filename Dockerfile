FROM gcc:13
WORKDIR /app
COPY src/ src/
RUN g++ -O2 -std=c++17 -o /usr/local/bin/trustnav src/main.cpp
# docker run --rm -v /path/to/mav0:/data trustnav /data/imu0/data.csv /data/estimated_trajectory.txt
ENTRYPOINT ["trustnav"]
