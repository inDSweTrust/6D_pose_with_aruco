import os
import json
import shutil

import cv2
import numpy as np

from extract_rosbags import ExtractRosbag
from pose_dataset_generator import PoseDatasetGenerator


if __name__ == "__main__": 
    # TODO create argparser
    bag_name = "21_test.bag"
    scene_id = "00"
    class_id = 2
    scene_split = "train"
    root_dir = "./"
    obj_offset = (13.2, 23.5, -6.6)

    # Define box corner dims/coords
    # TODO, re-do offset and object measurements, dims temporarily padded
    
    box_corners =  [
        (0, 0, 0),
        (0, 11.5, 0),
        (11.5, 11.5, 0),
        (11.5, 0, 0),
        (0, 0, 6.6),
        (0, 11.5, 6.6),
        (11.5, 11.5, 6.6),
        (11.5, 0, 6.6),
    ]
    # Create required dataset dirs if they don't exist
    if not os.path.exists(os.path.join(root_dir, "./hmi2")):
        os.makedirs(os.path.join(root_dir, "./hmi2"))

    if not os.path.exists(os.path.join(root_dir, "./hmi2", scene_split)):
        os.makedirs(os.path.join(root_dir, "./hmi2", scene_split))
    
    if not os.path.exists(os.path.join(root_dir, "./hmi2", scene_split, scene_id)):
        os.makedirs(os.path.join(root_dir, "./hmi2", scene_split, scene_id))
        os.makedirs(os.path.join(root_dir, "./hmi2", scene_split, scene_id, "rgb"))
        os.makedirs(os.path.join(root_dir, "./hmi2", scene_split, scene_id, "depth"))

    if not os.path.exists(os.path.join(root_dir, "./hmi2/annotations")):
        os.makedirs(os.path.join(root_dir, "./hmi2/annotations"))

    if not os.path.exists(os.path.join(root_dir, "./hmi2/models_eval")):
        os.makedirs(os.path.join(root_dir, "./hmi2/models_eval"))
    
    if not os.path.exists(os.path.join(root_dir, "./data", scene_id)):
        os.makedirs(os.path.join(root_dir, "./data", scene_id))
        os.makedirs(os.path.join(root_dir, "./data", scene_id, "rgb"))
        os.makedirs(os.path.join(root_dir, "./data", scene_id, "depth"))

    if not os.path.exists(os.path.join(root_dir, "./output", scene_id)):
        os.makedirs(os.path.join(root_dir, "./output", scene_id))

    # TODO implement function that creates classes and symm JSONs

    # Define input/output image paths
    image_dir = os.path.join("./data", scene_id, "rgb")
    output_dir = os.path.join("./output", scene_id)

    rosbag = ExtractRosbag(bag_name, root_dir = "./data/rosbags")

    pose_datagen = PoseDatasetGenerator(
        image_dir,
        output_dir,
        "DICT_6X6_250",
        board_size=(2,2), 
        marker_width=6.0,
        dist_between_markers=1.2,
    )
    print(rosbag.bag_dir)
    rgb_data, depth_data = rosbag.extract_rgb_depth_data()
    rosbag.save_images(rgb_data, depth_data, os.path.join("./data", scene_id))

    # For each RGB image, get 6D pose, bounding box, camera params
    # If Aruco board was not detected, image is not added to dataset
    # Dump to dataset folder in BOP format
    i = 0
    for index, file in enumerate(os.listdir(image_dir)):
        image = cv2.imread(os.path.join(image_dir, file))
        # success, pose_image = aruco.estimate_pose_aruco(image, (20.9, 20.05, -12.5))

        success, out = pose_datagen.estimate_pose_aruco(image, obj_offset)

        if success <= 0:
            # Aruco Pose not found - skip the image
            continue

        else:
            # Copy raw images to dataset folder
            shutil.copy(os.path.join("data", scene_id, f"rgb/{index}.png"), os.path.join("hmi2", scene_split, scene_id, f"rgb/{i}.png"))
            shutil.copy(os.path.join("data", scene_id, f"depth/{index}.png"), os.path.join("hmi2", scene_split, scene_id, f"depth/{i}.png"))

            w2c_tvec, rotation_matrix, obj_tvec, w2c_rvec = out
            pose_image = pose_datagen.visualize_pose(image, w2c_tvec, obj_tvec, w2c_rvec)
            
            # Add pose to scene gt dict
            pose_datagen.update_scene_gt(w2c_tvec, rotation_matrix, i, class_id)


            # treat obj as box, get all edge points in world space
            edge_points = pose_datagen.get_edge_points_rectangle(box_corners, obj_offset)
            
            # from list of points in world space, get screen space coordinates of boundingbox that contains all 3d points
            min_point, max_point, screen_points = pose_datagen.get_bounding_box(edge_points, w2c_tvec, rotation_matrix)
            pose_datagen.plot_2d_points([min_point, max_point], screen_points, image, i)

            # Add bb to scene gt info dict
            pose_datagen.update_scene_gt_info(min_point, max_point, i)
            # Add scene camera info 
            pose_datagen.update_scene_camera(w2c_tvec, rotation_matrix, i)
            i += 1


    # Dump dicts to JSON
    with open(os.path.join('./hmi2', scene_split, scene_id, 'scene_gt.json'), 'w', encoding='utf-8') as f:
        json.dump(pose_datagen.scene_gt_dict, f)

    with open(os.path.join('./hmi2', scene_split, scene_id, 'scene_gt_info.json'), 'w', encoding='utf-8') as f:
        json.dump(pose_datagen.scene_gt_info_dict, f)

    with open(os.path.join('./hmi2', scene_split, scene_id, 'scene_camera.json'), 'w', encoding='utf-8') as f:
        json.dump(pose_datagen.scene_camera_dict, f)

    