# READ ME
Floor plan identification on image / PDF file

The main idea of the algorithm is contour detection for both texts and rooms.

![image](https://github.com/ChingXuen/Automatic-Room-Segmentation-and-Labelling-in-Architectural-Floor-Plan-Images/assets/74391322/c331e66c-4f08-4a9b-bd35-2d17d8569d86)

Parameters explained in each function:

1. extract_room_labels(img)
a. Adjusting the width and height restrictions to obtain objects of size desired - Change to ensure all desired texts are bounded

2. fill_wall_gaps(img)
a. Harris Corner Detector - cv2.cornerHarris(img, blockSize (higher leads to more duplication of corners), ksize (must be odd number and <= 31), k (higher value, less false corners detected; smaller value, more false corners))
b. Draw straight lines from parallel corners (in x & y direction) if they are within defined distance apart

3. detect_rooms(gaps_filled_img, text_extracted_img)
a. Morphological Opening - To remove objects with the size of kernel defined
b. Conditions defined to filter the rooms (contours) - Area > 1000 AND area < img_size * 0.5 AND w * h < img_size

Limitations of algorithm:

1. Number of parameters that require tuning and understanding (not generalized for all floor plans)
2. Lack of symbol detection and removal (e.g. furniture)
3. Does not split large rooms that contain multiple labels

# HOW TO RUN ON GOOGLE COLAB
All you need to do is to run the cells inside the file AEFP.ipynb one by one.
