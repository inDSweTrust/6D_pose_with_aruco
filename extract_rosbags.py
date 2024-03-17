import os
import rosbag

from sensor_msgs.msg import Image
import cv2
import numpy as np

class ExtractRosbag():
    def __init__(self, bag_file, root_dir = "./data/rosbags"):
        self.bag_file = bag_file
        self.root_dir = root_dir

        self.bag_dir = os.path.join(root_dir, bag_file)
        self.bag = rosbag.Bag(self.bag_dir, 'r')


    def extract_rgb_depth_data(self):
        bag = rosbag.Bag(self.bag_dir, 'r')

        rgb_data = []
        depth_data = []

        for topic, msg, t in bag.read_messages(topics=['/device_0/sensor_0/Color_0/image/data', '/device_0/sensor_0/Depth_0/image/data']):
            if "Color_0" in topic:
                rgb_data.append((t, msg))
            elif "Depth_0" in topic:
                depth_data.append((t, msg))

        bag.close()

        return rgb_data, depth_data


    def save_images(self, rgb_data, depth_data, path, max_files=100000):
        count = 0
        for t, rgb_msg in rgb_data:
            if count == max_files:
                break
            rgb_image = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape((rgb_msg.height, rgb_msg.width, 3))
            cv2.imwrite(os.path.join(path, f'rgb/{count}.png'), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            count += 1
            
        count = 0
        for t, depth_msg in depth_data:
            if count == max_files:
                break
            depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape((depth_msg.height, depth_msg.width))
            cv2.imwrite(os.path.join(path, f'depth/{count}.png'), depth_image)
            count += 1


if __name__ == "__main__":
    pass