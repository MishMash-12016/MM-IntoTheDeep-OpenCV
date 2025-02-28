import cv2
import numpy as np
import math

def calculate_angle(contour):
    if len(contour) < 5:
        return 0
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    return angle

def draw_info(image, color, angle, center, index, distance):
    cv2.putText(image, f"#{index}: {color}", (center[0] - 40, center[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, f"Angle: {angle:.2f}", (center[0] - 40, center[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, f"dis: {distance:.2f}", (center[0] - 40, center[1] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.circle(image, center, 5, (0, 255, 0), -1)
    cv2.line(image, center, (int(center[0] + 50 * math.cos(math.radians(90 - angle))),
                             int(center[1] - 50 * math.sin(math.radians(90 - angle)))), (0, 255, 0), 2)

def process_color(frame, mask, color_name, color_bgr):
    kernel = np.ones((5, 5), np.uint8)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    gray_boosted = cv2.addWeighted(gray_masked, 1.5, mask, 0.5, 0)
    blurred = cv2.GaussianBlur(gray_boosted, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
    edges = cv2.bitwise_not(edges)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy, gray_masked

def are_samples_close(center1, center2, threshold=100):
    return np.linalg.norm(np.array(center1) - np.array(center2)) < threshold


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_lowest_middle_point(contour):
    # Calculate the moments of the contour
    M = cv2.moments(contour)

    # Calculate the center of the contour using the moments
    if M["m00"] == 0:
        return float('inf')  # To avoid division by zero

    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    # The lowest middle point of the screen is (320, 480)
    lowest_point = (320, 480)

    # Calculate the Euclidean distance from the center of the contour to the lowest middle point
    distance = np.sqrt((center_x - lowest_point[0]) ** 2 + (center_y - lowest_point[1]) ** 2)

    return distance

def runPipeline(frame, llrobot):
    HSV_BLUE_RANGE = ([90, 120, 40], [140, 255, 255])
    HSV_RED_RANGE_1 = ([0, 120, 40], [10, 255, 255])
    HSV_RED_RANGE_2 = ([160, 120, 40], [180, 255, 255])
    HSV_YELLOW_RANGE = ([20, 120, 40], [40, 255, 255])
    SMALL_CONTOUR_AREA = 200
    MIN_BRIGHTNESS_THRESHOLD = 60

    frame_center_x = 640 / 2
    lowest_distance = 1000

    largest_contour = np.array([[]])
    largest_sample_angle = 0
    largest_sample_area = 0
    close_samples = []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_denoised = cv2.GaussianBlur(hsv, (5, 5), 0)

    blue_mask = cv2.inRange(hsv_denoised, np.array(HSV_BLUE_RANGE[0]), np.array(HSV_BLUE_RANGE[1]))
    red_mask1 = cv2.inRange(hsv_denoised, np.array(HSV_RED_RANGE_1[0]), np.array(HSV_RED_RANGE_1[1]))
    red_mask2 = cv2.inRange(hsv_denoised, np.array(HSV_RED_RANGE_2[0]), np.array(HSV_RED_RANGE_2[1]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv_denoised, np.array(HSV_YELLOW_RANGE[0]), np.array(HSV_YELLOW_RANGE[1]))

    all_contours = [
        (process_color(frame, blue_mask, "Blue", (255, 0, 0)), "Blue", (255, 0, 0)),
        (process_color(frame, red_mask, "Red", (0, 0, 255)), "Red", (0, 0, 255)),
        (process_color(frame, yellow_mask, "Yellow", (0, 255, 255)), "Yellow", (0, 255, 255))
    ]

    for (contours, hierarchy, gray), color, color_bgr in all_contours:
        for i, sample in enumerate(contours):
            if cv2.contourArea(sample) < SMALL_CONTOUR_AREA:
                continue

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [sample], -1, 255, -1)

            if cv2.mean(gray, mask=mask)[0] < MIN_BRIGHTNESS_THRESHOLD:
                continue

            M = cv2.moments(sample)
            if M["m00"] == 0:
                continue

            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            center_array = np.array(center)
            angle = calculate_angle(sample)
            area = cv2.contourArea(sample)
            distance = calculate_distance(center_array[0], center_array[1], 320, 480)

            cv2.drawContours(frame, [sample], 0, color_bgr, 2)
            draw_info(frame, color, angle, center, i + 1, distance)

            if distance < lowest_distance:
                largest_sample_area = area
                largest_sample_angle = angle
                largest_contour = sample
                lowest_distance = distance

            close_samples.append((center, color, angle, distance))

    # Check for close samples
    for i in range(len(close_samples)):
        for j in range(i + 1, len(close_samples)):
            if are_samples_close(close_samples[i][0], close_samples[j][0]):
                center1, color1, angle1, area1 = close_samples[i]
                center2, color2, angle2, area2 = close_samples[j]
                midpoint = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
                cv2.line(frame, center1, center2, (0, 255, 0), 2)
                cv2.putText(frame, "Close Samples", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    llpython = [largest_sample_angle, 1 if len(close_samples) > 1 else 0]

    return largest_contour, frame, llpython
