import cv2
import numpy as np
import math
import time
from collections import defaultdict

# Track OpenCV function calls and timing
# opencv_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})

# def track_opencv(func_name):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             start = time.time()
#             result = func(*args, **kwargs)
#             end = time.time()
#             opencv_stats[func_name]['count'] += 1
#             opencv_stats[func_name]['total_time'] += (end - start)
#             return result
#         return wrapper
#     return decorator

# Wrap commonly used OpenCV functions
# cv2.split = track_opencv('split')(cv2.split)
# cv2.cvtColor = track_opencv('cvtColor')(cv2.cvtColor)
# cv2.inRange = track_opencv('inRange')(cv2.inRange)
# cv2.bitwise_and = track_opencv('bitwise_and')(cv2.bitwise_and)
# cv2.bitwise_or = track_opencv('bitwise_or')(cv2.bitwise_or)
# cv2.bitwise_not = track_opencv('bitwise_not')(cv2.bitwise_not)
# cv2.morphologyEx = track_opencv('morphologyEx')(cv2.morphologyEx)
# cv2.GaussianBlur = track_opencv('GaussianBlur')(cv2.GaussianBlur)
# cv2.Sobel = track_opencv('Sobel')(cv2.Sobel)
# cv2.Canny = track_opencv('Canny')(cv2.Canny)
# cv2.findContours = track_opencv('findContours')(cv2.findContours)
# cv2.drawContours = track_opencv('drawContours')(cv2.drawContours)
# cv2.bilateralFilter = track_opencv('bilateralFilter')(cv2.bilateralFilter)
# cv2.normalize = track_opencv('normalize')(cv2.normalize)
# cv2.dilate = track_opencv('dilate')(cv2.dilate)
# cv2.contourArea = track_opencv('contourArea')(cv2.contourArea)

# Constants for filtering contours
SMALL_CONTOUR_AREA = 300
LARGEST_CONTOUR_AREA = 2000

# Minimum average brightness threshold (0-255)
MIN_BRIGHTNESS_THRESHOLD = 50

# Color detection ranges for yellow in HSV
HSV_BLUE_RANGE = ([90, 60, 100], [140, 255, 255])

# Edge detection parameters - initial values
BLUR_SIZE = 17
SOBEL_KERNEL = 3

# Aspect ratio range for contour filtering
MIN_ASPECT_RATIO = 1.5  # Minimum width/height ratio
MAX_ASPECT_RATIO = 6.0  # Maximum width/height ratio

# Vertical position threshold (in pixels from bottom)
VERTICAL_THRESHOLD = 145  # Adjust this value as needed
X_LIMIT_AUTO = 190  # Adjust this value as needed

tracked_contour = None
tracked_center = None
DISTANCE_THRESHOLD = 50  # pixels, adjust as needed


def calculate_angle(contour):
    if len(contour) < 5:
        return 0
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    return angle


def draw_info(image, color, angle, center, index, area):
    cv2.putText(image, f"#{index}: {color}", (center[0] - 40, center[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.putText(image, f"Angle: {angle:.2f}", (center[0] - 40, center[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (250, 255, 0), 2)
    cv2.putText(image, f"Area: {area:.2f}", (center[0] - 40, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.circle(image, center, 5, (0, 255, 0), -1)
    cv2.line(image, center, (int(center[0] + 50 * math.cos(math.radians(90 - angle))),
                             int(center[1] - 50 * math.sin(math.radians(90 - angle)))), (0, 255, 0), 2)


def separate_touching_contours(contour, min_area_ratio=0.15):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - [x, y]
    cv2.drawContours(mask, [shifted_contour], -1, 255, -1)

    original_area = cv2.contourArea(contour)
    max_contours = []
    max_count = 1

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    for threshold in np.linspace(0.1, 0.9, 9):
        _, thresh = cv2.threshold(dist_transform, threshold * dist_transform.max(), 255, 0)
        thresh = np.uint8(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > original_area * min_area_ratio]

        if len(valid_contours) > max_count:
            max_count = len(valid_contours)
            max_contours = valid_contours

    if max_contours:
        return [c + [x, y] for c in max_contours]
    return [contour]


def pipeline_debug_return(frame):
    return None, None, None, True, frame


def process_color(frame, mask):
    debug_info = None
    # return pipeline_debug_return(frame)
    kernel = np.ones((5, 5), np.uint8)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask) if 1 else frame
    # return pipeline_debug_return(masked_frame)
    gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY) if 1 else masked_frame
    # return pipeline_debug_return(gray_masked)
    gray_boosted = cv2.addWeighted(gray_masked, 1.5, mask, 0.5, 0) if 0 else gray_masked
    # return pipeline_debug_return(gray_boosted)
    blurred = cv2.GaussianBlur(gray_boosted, (3, 3), 0) if 1 else gray_boosted
    # return pipeline_debug_return(blurred)

    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=1)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))
    # return pipeline_debug_return(magnitude)

    _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY) if 1 else magnitude
    # return pipeline_debug_return(edges)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) if 1 else edges
    # return pipeline_debug_return(edges)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) if 1 else edges
    # return pipeline_debug_return(edges)
    edges = cv2.bitwise_not(edges) if 1 else edges
    # return pipeline_debug_return(edges)
    edges = cv2.bitwise_and(edges, edges, mask=mask) if 1 else edges
    # return pipeline_debug_return(edges)
    edges = cv2.GaussianBlur(edges, (3, 3), 0) if 1 else edges
    # return pipeline_debug_return(edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy, gray_masked, False, debug_info


def debug_return(frame):
    return np.array([[]]), frame, [0, 0, 0, 0, 0, 0, 0, 0]


def draw_threshold_blocks(image):
    height, width = image.shape[:2]

    # Draw upper block
    cv2.rectangle(image, (0, 0), (width, VERTICAL_THRESHOLD - 22), (0, 0, 0), -1)


def runPipeline(frame, llrobot):
    global tracked_contour, tracked_center

    try:
        # Initialize Limelight-style output
        llpython = [0, 0, 0, 0, 0, 0, 0, 0]
        closest_contour = np.array([[]])
        min_distance = float('inf')

        # Set the reference point to (320, 0)
        reference_point = (320, 0)

        # Convert to HSV and denoise
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_denoised = cv2.GaussianBlur(hsv, (5, 5), 0)

        # Create mask for blue
        blue_mask = cv2.inRange(hsv_denoised, np.array(HSV_BLUE_RANGE[0]), np.array(HSV_BLUE_RANGE[1]))
        blue_mask = cv2.erode(blue_mask, np.ones((3, 3), np.uint8))

        # Process yellow color
        blue_contours, blue_hierarchy, blue_gray, isDebug, debug_info = process_color(frame, blue_mask)
        if isDebug:
            return debug_return(debug_info)

        valid_contours = []
        for i, contour in enumerate(blue_contours):
            if cv2.contourArea(contour) < SMALL_CONTOUR_AREA or cv2.contourArea(contour) > LARGEST_CONTOUR_AREA:
                continue

            # Check aspect ratio using minAreaRect
            rect = cv2.minAreaRect(contour)
            width = max(rect[1])
            height = min(rect[1])
            if width == 0 or height == 0:
                continue

            aspect_ratio = width / height
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                continue

            for sep_contour in separate_touching_contours(contour):
                mask = np.zeros(blue_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [sep_contour], -1, 255, -1)

                if cv2.mean(blue_gray, mask=mask)[0] < MIN_BRIGHTNESS_THRESHOLD:
                    continue

                M = cv2.moments(sep_contour)
                if M["m00"] == 0:
                    continue

                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

              # Skip if contour is above vertical threshold (now measured from top)
                if center[1] < VERTICAL_THRESHOLD:
                    continue

                if center[0] < X_LIMIT_AUTO and llrobot[2] == 1:
                    continue

                angle = calculate_angle(sep_contour)
                area = cv2.contourArea(sep_contour)

                # Calculate distance to reference point (320, 0)
                distance = np.sqrt((center[0] - reference_point[0]) ** 2 + (center[1] - reference_point[1]) ** 2)

                # Store valid contour info
                valid_contours.append({
                    'contour': sep_contour,
                    'center': center,
                    'angle': angle,
                    'area': area,
                    'index': i,
                    'distance': distance
                })

        if llrobot[1] != 0:
            # Implement tracking logic
            if tracked_contour is None or tracked_center is None:
                # No contour is currently being tracked, find the closest one
                for contour_info in valid_contours:
                    if contour_info['distance'] < min_distance:
                        min_distance = contour_info['distance']
                        closest_contour = contour_info['contour']
                        tracked_contour = closest_contour
                        tracked_center = contour_info['center']
                        llpython = [1, tracked_center[0], tracked_center[1], contour_info['angle'], len(blue_contours),
                                    min_distance, 0, 0]
            else:
                # Check if the tracked contour is still visible
                tracked_contour_found = False
                for contour_info in valid_contours:
                    distance_to_tracked = np.sqrt((contour_info['center'][0] - tracked_center[0]) ** 2 +
                                                  (contour_info['center'][1] - tracked_center[1]) ** 2)
                    if distance_to_tracked < DISTANCE_THRESHOLD:
                        tracked_contour_found = True
                        tracked_contour = contour_info['contour']
                        tracked_center = contour_info['center']
                        llpython = [1, tracked_center[0], tracked_center[1], contour_info['angle'], len(blue_contours),
                                    distance_to_tracked, 0, 0]
                        break

                if not tracked_contour_found:
                    # Tracked contour lost, reset tracking
                    tracked_contour = None
                    tracked_center = None

        # Draw all valid contours and their info
        for contour_info in valid_contours:
            color = (0, 255, 0) if np.array_equal(contour_info['contour'], tracked_contour) else (0, 0, 255)
            cv2.drawContours(frame, [contour_info['contour']], -1, color, 2)
            draw_info(frame, "Blue", contour_info['angle'], contour_info['center'],
                      contour_info['index'] + 1, contour_info['area'])

        # Draw the reference point
        cv2.circle(frame, reference_point, 5, (255, 0, 0), -1)

        draw_threshold_blocks(frame)

        #draw_x_limit_auto(frame)

        return tracked_contour if tracked_contour is not None else np.array([[]]), frame, llpython

    except Exception as e:
        print(f"Error: {str(e)}")
        return np.array([[]]), frame, [0, 0, 0, 0, 0, 0, 0, 0]
