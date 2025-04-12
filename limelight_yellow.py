import cv2
import numpy as np
import math

def drawDecorations(image, angle):
    cv2.putText(image,
                f'R {angle:.2f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

def runPipeline(image, llrobot):
    # Initialize variables
    largestContour = np.array([[]])
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]
    angle = 0
    longest_line = None
    
    # Step 1: Crop the image
    crop_x, crop_y = 0, 0
    crop_w, crop_h = 320, 240
    if llrobot[3] != 0:
        crop_x = int(llrobot[5])
        crop_y = int(llrobot[6])
        crop_w = int(llrobot[3])
        crop_h = int(llrobot[4])
    cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    # Step 2: Convert to HSV
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Step 3: Create a mask for yellow color (using two ranges)
    lower_yellow1 = np.array([16, 83, 101])
    upper_yellow1 = np.array([32, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)

    # Step 4: Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select the largest contour rather than concatenating all contours
        largest_contour = max(contours, key=cv2.contourArea)
        # Step 5: If touching the border, use Hough Transform to incorporate the second (partial) rectangle
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        longest_line_angle = None
        if lines is not None:
            longest_length = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw the Hough line in cyan
                cv2.line(cropped, (x1, y1), (x2, y2), (255, 255, 0), 2)
                # Compute the length of the line
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > longest_length:
                    longest_length = length
                    longest_line = line
                    # Calculate angle in degrees using atan2 (result in range -180 to 180)
                    longest_line_angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
            # If a longest Hough line was found, use its angle
            if longest_line_angle is not None:
                angle = longest_line_angle
            # Optionally, you could also compute a rectangle from all Hough endpoints if needed:
            pts = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                pts.append([x1, y1])
                pts.append([x2, y2])
            pts = np.array(pts)
            rect_hough = cv2.minAreaRect(pts)
            box_hough = cv2.boxPoints(rect_hough)
            box_hough = np.int0(box_hough)
            angle += 90
            llpython = [angle, 0, 0, 0, 0, 0, 0, 0]
        else:
            angle += 90
            llpython = [angle, 0, 0, 0, 0, 0, 0, 0]
        largestContour = largest_contour
    
    if longest_line is not None:
        cv2.line(cropped, (longest_line[0][0], longest_line[0][1]), (longest_line[0][2], longest_line[0][3]), (0, 0, 0), 2)

    # Step 6: Draw the crop rectangle on the original image
    cv2.rectangle(image, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (255, 0, 0), 2)
    # Place the processed cropped image back into the original image
    image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cropped
    # Draw angle decoration
    drawDecorations(image, angle)

    # Instead of returning multiple values, return the angle of the longest Hough line (or the contour's angle if Hough not used)
    return largestContour, image, llpython
