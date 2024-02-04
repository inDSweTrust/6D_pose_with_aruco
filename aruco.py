import numpy as np
import cv2
print(cv2.__version__)

class Aruco():
    def __init__(self, dict_type) -> None:

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
        self.camera_matrix = np.array([[433.34295654296875, 0.0, 418.0463562011719], [0.0, 432.9460754394531, 244.36111450195312], [0.0, 0.0, 1.0]])
        self.distortion_coeffs = np.array([-0.053771115839481354, 0.05535884201526642, 0.0008205653284676373, 0.0009527259971946478, -0.018054140731692314])

    def create_aruco_marker(self):
        # Create an aruco marker
        marker_id = 23
        marker_size = 200
        # Generate aruco marker image
        marker_image = cv2.aruco.generateImageMarker(self.dictionary, marker_id, marker_size)

        # Save image in dir
        cv2.imwrite("./markers/marker.png",marker_image)

    def create_aruco_board(self):
        # Create an ArUco board
        board_size = (3, 4)
        board = cv2.aruco.GridBoard(board_size, 5.0, 1.0, self.dictionary)
        # Generate the board image
        board_image = board.generateImage((board_size[0]*250, board_size[1]*250), marginSize=10)
        # aruco_corners = np.array(board.getObjPoints())[:, :, :2]

        # Save images in dir
        cv2.imwrite("./boards/aruco_board.png", board_image)

    def detect_aruco(self, image):
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, aruco_params)
        (corners, ids, rejected) = aruco_detector.detectMarkers(image)

        return corners, ids

    def estimate_pose_aruco(self, image, transform=None):
        # create the board
        board = cv2.aruco.GridBoard((3, 4), 5.0, 1.0, self.dictionary)

        rvec = None
        tvec = None

        # detect markers
        corners, ids = self.detect_aruco(image)
        success, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, self.camera_matrix, self.distortion_coeffs, rvec, tvec)
        
        # If need to get pose of an object, transform coords
        # Only the translation coords need to be transformed

        print("ROTATION")
        print(rvec)
        print("TRANSLATION")
        print(tvec)

        if transform:
            print("TRANSFORM")
            tvec[0][0] -= transform[0] # x-coord
            tvec[1][0] -= transform[1] # y-coord
            tvec[2][0] -= transform[2] # z-coord

        print("ROTATION")
        print(rvec)
        print("TRANSLATION")
        print(tvec)

        if success > 0:
            image = cv2.drawFrameAxes(image, self.camera_matrix, self.distortion_coeffs, rvec, tvec, length=5, thickness=3)
                
        # save output image with markers
        cv2.imwrite("./output/test_pose.png", image)
    
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
                cv2.imwrite("./output/test.png", image)


if __name__ == "__main__":
    aruco = Aruco("DICT_6X6_250")
    # aruco.create_aruco_marker()
    # aruco.create_aruco_board()
    image_path = "./data/images/rgb_694800.png"
    image = cv2.imread(image_path)
    corners, ids = aruco.detect_aruco(image)
    # aruco.visualize_aruco(image, corners, ids)

    # aruco.estimate_pose_aruco(image)
    aruco.estimate_pose_aruco(image, (16.0, 10.0, 0.0))