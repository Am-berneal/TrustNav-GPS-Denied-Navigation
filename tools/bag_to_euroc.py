"""Convert an EuRoC ROS1 bag (e.g. Kaggle 'Euroc MH_01_easy ros bag dataset') to the ASL layout
needed by TrustNav:  <out>/mav0/imu0/data.csv  (#timestamp [ns], w_RS_S_x..z, a_RS_S_x..z).

No ROS install needed:  pip install rosbags
Usage:  python tools/bag_to_euroc.py MH_01_easy.bag MH_01_easy

The bag has no 6-DoF ground truth. Take mav0/state_groundtruth_estimate0/data.csv for the same
sequence from the Kaggle dataset 'drmainak/uzh-fpv-and-euroc' and put it at
<out>/mav0/state_groundtruth_estimate0/data.csv.
"""
import os
import sys

from rosbags.highlevel import AnyReader
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    bag, out = Path(sys.argv[1]), sys.argv[2]
    dst = os.path.join(out, 'mav0', 'imu0', 'data.csv')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    n = 0
    with AnyReader([bag]) as reader, open(dst, 'w') as fh:
        fh.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                 'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        conns = [c for c in reader.connections if c.topic == '/imu0']
        if not conns:
            sys.exit('no /imu0 topic; topics: %s' % sorted({c.topic for c in reader.connections}))
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            t = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
            w, a = m.angular_velocity, m.linear_acceleration
            fh.write('%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n' % (t, w.x, w.y, w.z, a.x, a.y, a.z))
            n += 1
    print('wrote %d IMU rows to %s' % (n, dst))


if __name__ == '__main__':
    main()
