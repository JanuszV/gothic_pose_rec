from PoseModule import poseDetector,handDetector,faceDetector
import cv2
import time
import mediapipe as mp

def visualize_points_in_3d(room_size, elements, room_image, connections):
    """
    Visualize a list of 3D points in a virtual room.

    Args:
        room_size (tuple): Dimensions of the virtual room (height, width, depth).
        elements (list): List of element lists where each element list contains [id, x, y, z] coordinates.

    Returns:
        numpy.ndarray: Image representing the virtual room with points.
    """
    room_height, room_width, room_depth = room_size

    # Create a black image representing the virtual room

    # Draw points for each element in the list
    for element in elements:
        id, x, y, z = element

        # Map real-world coordinates to virtual room coordinates
        virtual_x = int((x / room_width) * room_width)
        virtual_y = int((y / room_height) * room_height)
        virtual_z = int((z / room_depth) * room_depth)

        # Draw a point representing the element in the virtual room
        cv2.circle(room_image, (virtual_x, virtual_y), 5, (0, 255, 0), -1)
    
    for connection in connections:
        id1, id2 = connection

        # Find the coordinates of the connected points
        point1 = next((element for element in elements if element[0] == id1), None)
        point2 = next((element for element in elements if element[0] == id2), None)

        if point1 is not None and point2 is not None:
            # Map real-world coordinates to virtual room coordinates
            virtual_x1 = int((point1[1] / room_width) * room_width)
            virtual_y1 = int((point1[2] / room_height) * room_height)
            virtual_x2 = int((point2[1] / room_width) * room_width)
            virtual_y2 = int((point2[2] / room_height) * room_height)

            # Draw a line connecting the points
            cv2.line(room_image, (virtual_x1, virtual_y1), (virtual_x2, virtual_y2), (0, 255, 0), 2)

    return room_image
#implementation 
def main(cap = cv2.VideoCapture(0)):
    pose_detector = poseDetector()
    hand_detector = handDetector()
    face_detector = faceDetector()
    print_once = False
    while True:
        success, img = cap.read()
        if not success:
            break
        room_y, room_x, c= img.shape
        room_size = [room_y, room_x, room_x]
        pose_conns = pose_detector.findPose(img)
        hand_conns = hand_detector.findHand(img)
        face_conns = face_detector.findFace(img)
        pose_position = pose_detector.getPosition(img)
        hand_position = hand_detector.getPosition(img)
        face_position = face_detector.getPosition(img)
        img[:] = 0
        for i in range(len(hand_position)):
            img = visualize_points_in_3d(room_size, hand_position[i], img, hand_conns)
        img = visualize_points_in_3d(room_size, pose_position, img, pose_conns)
        img = visualize_points_in_3d(room_size, face_position, img, face_conns)
        cv2.imshow("Kamera na Beziego", img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()