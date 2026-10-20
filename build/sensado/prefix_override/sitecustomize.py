import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/verine/Capstone/ros2_ws/src/sensado/install/sensado'
