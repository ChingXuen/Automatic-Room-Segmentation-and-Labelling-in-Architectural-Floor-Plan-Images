import cv2
import numpy as np
import pytesseract
import random
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------------------------------------------------------------
def extract_room_labels(img):
    # Create mask of same size as image
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding 
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Find contours in binary image
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on mask for those within width and height boundaries
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 40 and 12 < h < 20:                       # Parameters
            cv2.drawContours(mask, [cnt], 0, (255,255,255), 1)
    
    # cv2.imshow("mask", mask)
    
    # Perform dilation on the contoured objects so words can be identified as one and repeat process as above   
    kernel = np.ones((4,4), np.uint8)                # Variable 
    dilation = cv2.dilate(mask, kernel, iterations = 1)
    gray_d = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)

    threshold_d = cv2.threshold(gray_d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours_d, hierarchy = cv2.findContours(threshold_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create copy to show image once texts are removed
    text_extracted_img = img.copy()
    bounded_text_img = img.copy()
    
    ROI = []
    coordinates = []
    
    # Draw rectangle on bounding boxes within width and height boundaries
    for cnt in contours_d:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 30 and h > 10:                          # Parameter
            cv2.rectangle(bounded_text_img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_c = bounded_text_img[y:y+h, x:x+w]
            coordinates.append((x, y, w, h))
            ROI.append(roi_c)
            text_extracted_img[y:y+h, x:x+w] = 255           # Replace the identified text with white color
    
    # plt.figure()
    # plt.imshow(bounded_text_img)
    cv2.imshow('labels detected', bounded_text_img)
    
    # Extract texts from bounding boxes
    room_labels = []
    for i, room in enumerate(ROI):
        text = pytesseract.image_to_string(room, lang="eng", config="--psm 6")   # config 
        text = text.replace("\n", ' ')
        room_labels.append((text, coordinates[i]))
        
    return room_labels, text_extracted_img

# -----------------------------------------------------------------------------------------------------------------------------------
def fill_wall_gaps(img):
    corners_detacted_img = img.copy()
    gaps_filled_img = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh= cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    thresh = np.float32(thresh)   # Converted to float32 type for Harris Corner function
    
    # Detect corners here 
    dst = cv2.cornerHarris(thresh, 3, 7, 0.07)   # Parameter    cv2.cornerHarris(img, blockSize, ksize, k)
    
    dst = cv2.dilate(dst, None)
    
    corners_detacted_img[dst > 0.1 * dst.max()]=[0, 255, 0]      # Show corners in green color 
    
    cv2.imshow("corners detected", corners_detacted_img)
    
    corners = dst > 0.1 * dst.max()      # Returns an array of image size with boolean values (true == corners)
    
    # Draw horizontal lines from parallel corners
    for y, row in enumerate(corners):
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):
            if abs(x2[0] - x1[0]) < 70:            # Parameter
                color = 0
                cv2.line(gaps_filled_img, (int(x1), int(y)), (int(x2), int(y)), color, 1)

    # Draw vertical lines from parallel corners
    for x, col in enumerate(corners.T):
        y_same_x = np.argwhere(col)
        for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
            if abs(y2[0] - y1[0]) < 70:            # Parameter
                color = 0
                cv2.line(gaps_filled_img, (int(x), int(y1)), (int(x), int(y2)), color, 1)
    
    return gaps_filled_img
    
# -----------------------------------------------------------------------------------------------------------------------------------
def detect_rooms(gaps_filled_img, text_extracted_img):
    segmented_rooms_img = text_extracted_img.copy()
    
    gray = cv2.cvtColor(gaps_filled_img, cv2.COLOR_RGB2GRAY)
    thresh= cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    mor_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=2)       # Parameter (to remove small objects)
    
    cv2.imshow("mor_img", mor_img)
    
    contours, hierarchy = cv2.findContours(mor_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort the contours in ascending order of area 
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)
    
    room_coordinates = []
    initial_contours = []
    
    img_size = gaps_filled_img.shape[0] * gaps_filled_img.shape[1]
    
    for c in sorted_contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c) # obtain coordinates of contours
        if area > 1000 and area < img_size * 0.5 and w * h < img_size :         # Parameter
            color = [random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)]
            cv2.fillPoly(segmented_rooms_img, [c], (color[0], color[1], color[2]))        # Show the segmented rooms in random colors 
            room_coordinates.append((x,y,w,h))
            initial_contours.append(c)
            
    return room_coordinates, initial_contours, segmented_rooms_img 

# -----------------------------------------------------------------------------------------------------------------------------------
def match_room_and_label(room_labels, initial_contours, img):
    final_segmented_rooms_img = img.copy()
    final_list = []
    
    # Match room labels to room coordinates 
    for c in initial_contours:
        for label in room_labels:
            x, y, w, h = cv2.boundingRect(c)
            label_midpoint = [((2*label[1][0]+label[1][2]) / 2), ((2*label[1][1]+label[1][3]) / 2)] # Calculate Midpoint (x, y)
            if cv2.pointPolygonTest(c, label_midpoint, False) == 1:           # If midpoint of label is within room boundaries
                final_list.append((label[0], (x, y, w, h)))
                color = [random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)]
                cv2.fillPoly(final_segmented_rooms_img, [c], (color[0], color[1], color[2]))        # Show final segmented rooms in random colors 

    return final_list, final_segmented_rooms_img

# -----------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    img = cv2.imread('./Images/01.png')
    cv2.imshow('original image', img)
    
    plt.figure()
    plt.imshow(img)
    
    room_labels, text_extracted_img = extract_room_labels(img)
    cv2.imshow('labels removed', text_extracted_img)
    print("room labels: " + str(room_labels))
    
    gaps_filled_img = fill_wall_gaps(text_extracted_img)
    cv2.imshow("lines drawn", gaps_filled_img)
    
    room_coordinates, initial_contours, segmented_rooms_img = detect_rooms(gaps_filled_img, text_extracted_img)
    cv2.imshow("rooms detected", segmented_rooms_img)
    print("\nrooms coordinates: " + str(room_coordinates))
    
    final_list, final_segmented_rooms_img = match_room_and_label(room_labels, initial_contours, text_extracted_img)
    cv2.imshow("final output", final_segmented_rooms_img)
    print("\nfinal output: " + str(final_list))

