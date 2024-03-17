import os
import json
import shutil

import numpy as np
import cv2
print(cv2.__version__)

from collections import defaultdict

#TODO rosbag class

class PoseDatasetGenerator():
    """
    Generates 6D pose dataset with Aruco board.

    """
    def __init__(self, input_dir, output_dir, dict_type, board_size=(2,2), marker_width=4.2, dist_between_markers=0.8) -> None:

        # Define input/output paths
        self.input_dir = input_dir
        self.output_dir = output_dir
        # self.dataset_dir = dataset_dir

        # Define names of each possible ArUco tag OpenCV supports
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }

        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])

        # Aruco Params
        self.board_size = board_size
        self.marker_width = marker_width
        self.dist_between_markers = dist_between_markers

        # Camera Params
        self.camera_matrix = np.array([[433.34295654296875, 0.0, 314.0463562011719], [0.0, 432.9460754394531, 244.36111450195312], [0.0, 0.0, 1.0]])
        self.distortion_coeffs = np.array([-0.053771115839481354, 0.05535884201526642, 0.0008205653284676373, 0.0009527259971946478, -0.018054140731692314])

        # Scene GT dicts
        self.scene_gt_dict = {}
        self.scene_gt_info_dict = {}
        self.scene_camera_dict = {}

    def create_aruco_marker(self):
        # Create an aruco marker
        marker_id = 23
        marker_size = 200
        # Generate aruco marker image
        marker_image = cv2.aruco.generateImageMarker(self.dictionary, marker_id, marker_size)

        # Save image in dir
        cv2.imwrite("./markers/marker.png", marker_image)

    def create_aruco_board(self):
        # Create an ArUco board
        board = cv2.aruco.GridBoard(self.board_size, 4.2, 0.8, self.dictionary)
        # Generate the board image
        board_image = board.generateImage((self.board_size[0]*250, self.board_size[1]*250), marginSize=10)
        # aruco_corners = np.array(board.getObjPoints())[:, :, :2]

        # Save images in dir
        cv2.imwrite("./boards/aruco_board.png", board_image)

    def detect_aruco(self, image):
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, aruco_params)
        (corners, ids, rejected) = aruco_detector.detectMarkers(image)

        return corners, ids

    def estimate_pose_aruco(self, image, world_offset, visual=True):
        # create the board
        board = cv2.aruco.GridBoard(self.board_size, self.marker_width, self.dist_between_markers, self.dictionary)

        rvec = None
        tvec = None

        # detect Aruco markers
        corners, ids = self.detect_aruco(image)
        if ids is None:
            return False, image
        
        success, w2c_rvec, w2c_tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, self.camera_matrix, self.distortion_coeffs, rvec, tvec)

        # If need to get pose of an object, transform coords
        # Only the translation coords need to be transformed

        rotation_matrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(w2c_rvec, rotation_matrix)

        # Project world-relative coordinates to camera coordinate space
        obj_tvec = self.world_to_cam(w2c_tvec, rotation_matrix, world_offset)

        out = (w2c_tvec, rotation_matrix, obj_tvec, w2c_rvec)

        return success, out

    
    def world_to_cam(self, w2c_tvec, rotation_matrix, world_offset):
        world_tvec = np.matrix([[world_offset[0]],[world_offset[1]],[world_offset[2]]])
        return rotation_matrix * world_tvec + w2c_tvec


    def get_edge_points_rectangle(self, edge_dims, world_offset):
        # Returns a list with all object corner points
        points = np.zeros((8, 3))
        for i, corner_offset in enumerate(edge_dims):
            # point = self.world_to_cam(w2c_tvec, rotation_matrix, world_offset + corner_offset)
            points[i] = np.array(world_offset) + np.array(corner_offset)

        return points

    def get_bounding_box(self, edge_points, w2c_tvec, rotation_matrix):
        # project points to 2d plane and get bounding box
        screen_points = np.zeros((8, 2))
        # print(edge_points)
        for i, p in enumerate(edge_points):
            # Project points to 3D
            screen_point, _  = cv2.projectPoints(p, rotation_matrix, w2c_tvec, self.camera_matrix, self.distortion_coeffs)
            screen_points[i] = np.squeeze(screen_point)
        
        x_min = np.min(screen_points[:,0]).astype('int')
        y_min = np.min(screen_points[:,1]).astype('int')
        x_max = np.max(screen_points[:,0]).astype('int')
        y_max = np.max(screen_points[:,1]).astype('int')

        return (x_min, y_min), (x_max, y_max), screen_points.astype('int')
    
    
    def plot_2d_points(self, points, screen_points, image, i):

        for point in points:
            image = cv2.circle(image, (int(point[0]), int(point[1])), 2, 255, -1)

        for point in screen_points:
            image = cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        
        image = cv2.rectangle(image, points[0], points[1], (255, 0, 0), 1)

        cv2.imwrite(os.path.join(self.output_dir, f"img_pose{i}.png"), image)


    
    def visualize_pose(self, image, w2c_tvec, obj_tvec, w2c_rvec):
        # world coord system
        image = cv2.drawFrameAxes(image, self.camera_matrix, self.distortion_coeffs, w2c_rvec, w2c_tvec, length=6, thickness=1)
        # object point
        image = cv2.drawFrameAxes(image, self.camera_matrix, self.distortion_coeffs, w2c_rvec, obj_tvec, length=3, thickness=1)

        # return output image with markers
        return image
    

    
    def visualize_aruco(self, image, corners, ids):

        # Verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            ids = ids.flatten()
            # loop over the detected ArUCo corners

            for (marker_corner, marker_id) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                top_right = (int(top_right[0]), int(top_right[1]))
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

                # draw the bounding box of the ArUCo detection
                cv2.line(image, top_left, top_right, (0, 255, 0), 2)
                cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)

                # compute and draw the center (x, y)-coordinates of the ArUco marker
                cX = int((top_left[0] + bottom_right[0]) / 2.0)
                cY = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(marker_id),
                    (top_left[0], top_left[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print("[INFO] ArUco marker ID: {}".format(marker_id))

                # save output image with markers
                cv2.imwrite(os.path.join(self.output_dir, "test.png"), image)


    def update_scene_gt(self, w2c_tvec, rotation_matrix, scene_id, obj_id):
        """
        Update dict that contains object pose data.

        """

        # Flatten numpy arrays and convert to dicts
        cam_t_m2c = w2c_tvec.flatten().tolist()
        cam_r_m2c = rotation_matrix.flatten().tolist()

        # Add translation, rotation to scene gt dict
        self.scene_gt_dict[scene_id] = []
        obj_dict = {
            'cam_R_m2c': cam_r_m2c,
            'cam_t_m2c': cam_t_m2c,
            'obj_id': obj_id,

        }
        self.scene_gt_dict[scene_id].append(obj_dict)

    def update_scene_gt_info(self, min_point, max_point, scene_id):
        """
        Update dict that contains bounding boxes.

        """
        xmin, ymin = min_point
        xmax, ymax = max_point
        width = xmax - xmin
        height = ymax - ymin

        # Add bb meta data to scene gt info dict
        self.scene_gt_info_dict[scene_id] = []

        obj_dict = {
            'bbox_obj': [int(xmin), int(ymin), int(width), int(height)],
            'bbox_visib': [int(xmin), int(ymin), int(width), int(height)],
            'px_count_all': int(width) * int(height), #TODO, NOT correct
            'px_count_valid': int(width) * int(height),
            'px_count_visib': int(width) * int(height),
            'visib_fract': 1.0,
        }
        self.scene_gt_info_dict[scene_id].append(obj_dict)

    
    def update_scene_camera(self, w2c_tvec, rotation_matrix, scene_id):

        # Flatten numpy arrays and convert to dicts
        cam_k = self.camera_matrix.flatten().tolist()
        cam_r_w2c = rotation_matrix.flatten().tolist()
        cam_t_w2c = w2c_tvec.flatten().tolist()

        self.scene_camera_dict[scene_id] = defaultdict(list)
        self.scene_camera_dict[scene_id]['cam_K'] = cam_k
        self.scene_camera_dict[scene_id]['cam_R_w2c'] = cam_r_w2c
        self.scene_camera_dict[scene_id]['cam_t_w2c'] = cam_t_w2c
        
        #TODO, fix depth scale param
        self.scene_camera_dict[scene_id]['depth_scale'] = 0.1

if __name__ == "__main__":
    
    # TODO create argparser
    root_dir = "./"

    # Create required dataset dirs if they don't exist
    if not os.path.exists(os.path.join(root_dir, "./hmi2")):
        os.makedirs(os.path.join(root_dir, "./hmi2"))

    if not os.path.exists(os.path.join(root_dir, "./hmi2/train")):
        os.makedirs(os.path.join(root_dir, "./hmi2/train"))

    if not os.path.exists(os.path.join(root_dir, "./hmi2/test")):
        os.makedirs(os.path.join(root_dir, "./hmi2/test"))

    if not os.path.exists(os.path.join(root_dir, "./hmi2/annotations")):
        os.makedirs(os.path.join(root_dir, "./hmi2/annotations"))

    if not os.path.exists(os.path.join(root_dir, "./hmi2/models_eval")):
        os.makedirs(os.path.join(root_dir, "./hmi2/models_eval"))

    # TODO implement function that creates classes and symm JSONs

    # Define input/output image paths
    image_dir = "./data/0/rgb"
    output_dir = "./output/0"

    pose_datagen = PoseDatasetGenerator(
        image_dir,
        output_dir,
        "DICT_6X6_250",
        board_size=(2,2), 
        marker_width=6.0,
        dist_between_markers=1.2,
    )

    # For each RGB image, get 6D pose, bounding box, camera params
    # If Aruco board was not detected, image is not added to dataset
    # Dump to dataset folder in BOP format
    i = 0
    for index, file in enumerate(os.listdir(image_dir)):
        image = cv2.imread(os.path.join(image_dir, file))
        # success, pose_image = aruco.estimate_pose_aruco(image, (20.9, 20.05, -12.5))

        obj_offset = (14.6, 15.6, -12.5)
        success, out = pose_datagen.estimate_pose_aruco(image, obj_offset)

        if success <= 0:
            # Aruco Pose not found - skip the image
            continue

        else:
            # Copy raw images to dataset folder
            shutil.copy(f"data/0/rgb/{index}.png", f"hmi2/train/0/rgb/{i}.png")
            shutil.copy(f"data/0/depth/{index}.png", f"hmi2/train/0/depth/{i}.png")

            w2c_tvec, rotation_matrix, obj_tvec, w2c_rvec = out
            pose_image = pose_datagen.visualize_pose(image, w2c_tvec, obj_tvec, w2c_rvec)
            
            # Add pose to scene gt dict
            pose_datagen.update_scene_gt(w2c_tvec, rotation_matrix, i)
            
            # Define box corner dims/coords
            # box_corners = [
            #     (0, 0, 0),
            #     (0, 8.9, 0),
            #     (12.6, 8.9, 0),
            #     (12.6, 0, 0),
            #     (0, 0, 12.5),
            #     (0, 8.9, 12.5),
            #     (12.6, 8.9, 12.5),
            #     (12.6, 0, 12.5),
            # ]
            
            # TODO, re-do offset and object measurements, dims temporarily padded
            box_corners = [
                (0, 0, 0),
                (0, 10.9, 0),
                (14.6, 10.9, 0),
                (14.6, 0, 0),
                (0, 0, 14.5),
                (0, 10.9, 14.5),
                (14.6, 10.9, 14.5),
                (14.6, 0, 14.5),
            ]

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
    with open(os.path.join('./hmi2/train/0', 'scene_gt.json'), 'w', encoding='utf-8') as f:
        json.dump(pose_datagen.scene_gt_dict, f)

    with open(os.path.join('./hmi2/train/0', 'scene_gt_info.json'), 'w', encoding='utf-8') as f:
        json.dump(pose_datagen.scene_gt_info_dict, f)

    with open(os.path.join('./hmi2/train/0', 'scene_camera.json'), 'w', encoding='utf-8') as f:
        json.dump(pose_datagen.scene_camera_dict, f)