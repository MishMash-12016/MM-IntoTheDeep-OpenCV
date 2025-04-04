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

    
    #llrobot = [0, 0, 0, 44, 55, 130, 120, 0]

    # Step 1: Crop the image
    crop_x, crop_y = 0,0
    crop_w, crop_h = 320, 240
    if llrobot[3] != 0:
        crop_x = int(llrobot[5])
        crop_y = int(llrobot[6])
        crop_w = int(llrobot[3])
        crop_h = int(llrobot[4])
    cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    # Step 2: Convert to HSV
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Step 3: Create a mask for red color (using two ranges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Step 4: Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select the largest contour rather than concatenating all contours
        largest_contour = max(contours, key=cv2.contourArea)

        # Get rotated rectangle from the largest contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = rect[2]
        if angle < 0:
            angle += 90

        # Draw the rectangle and its center on the cropped image
        cv2.drawContours(cropped, [box], 0, (0, 255, 0), 2)
        cv2.circle(cropped, tuple(map(int, rect[0])), 5, (0, 0, 255), -1)



        # Step 5: If touching the border, use Hough Transform to incorporate the second (partial) rectangle
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        longest_line_angle = None
        if lines is not None :
            longest_length = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw the Hough line in cyan
                cv2.line(cropped, (x1, y1), (x2, y2), (255, 255, 0), 2)
                # Compute the length of the line
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > longest_length:
                    longest_length = length
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
            cv2.drawContours(cropped, [box_hough], 0, (255, 0, 255), 2)
            angle +=90
            llpython = [angle, 0, 0, 0, 0, 0, 0, 0]
        else:
            angle +=90
            llpython = [angle, 0, 0, 0, 0, 0, 0, 0]
        largestContour = largest_contour

    # Step 6: Draw the crop rectangle on the original image
    cv2.rectangle(image, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (255, 0, 0), 2)
    # Place the processed cropped image back into the original image
    image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cropped
    # Draw angle decoration
    drawDecorations(image, angle)

    # Instead of returning multiple values, return the angle of the longest Hough line (or the contour's angle if Hough not used)

    return largestContour, image, llpython
