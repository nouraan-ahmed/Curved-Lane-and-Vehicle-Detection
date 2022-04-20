import glob
import os
import sys

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import *

file_name = sys.argv[1]
input_video_path = sys.argv[2]
output_video_path = sys.argv[3]
debug = sys.argv[3]

objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)


def undistort_image(image):
    img_size = (image.shape[1], image.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)
    return cv2.undistort(image, mtx, dist, None, mtx)


# test for the function
# test_image = cv2.imread(images[0])
# cv2.imwrite(output_video_path +
#             '/output_images/chessboard_before_distortion.jpg', test_image)
# cv2.imwrite(output_video_path+'/output_images/chessboard_after_distortion.jpg',
#             undistort_image(test_image))

images = []
data_path = input_video_path+'test_images/'
out_path = output_video_path+'output_images/'
for file in os.listdir(data_path):
    if '.jpg' in file:
        image = mpimg.imread(data_path + file)
        images.append(image)


def showImages(process, num=len(images), show_gray=False):
    for i, image in enumerate(images[0:num]):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        ax1.set_title('Original Image')
        ax1.imshow(image)
        ax2.set_title('Processed Image')
        processed = process(image)
        if show_gray:
            ax2.imshow(processed, 'gray')
        else:
            ax2.imshow(processed)
        plt.show()


# showImages(undistort_image)


# Edge deection using sobel


def abs_sobel(image, direction='x', kernel_size=3, thresh_range=(0, 255)):

    # Convert to gray scale image
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient then take the absolute
    if direction == 'x':
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
        abs_sobel = np.absolute(sobel_x)
    if direction == 'y':
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
        abs_sobel = np.absolute(sobel_y)

    # Convert to 'cv2.CV_8U'.
    sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Take the same shape and with setting it's values to 0
    binary_img = np.zeros_like(sobel)
    binary_img[(sobel >= thresh_range[0]) & (sobel <= thresh_range[1])] = 1

    # Black and white output image
    return binary_img


def calc_mag_thresh(image, kernel_size=3, thresh_range=(0, 255)):

    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate Sobel x and y gradients
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # Calculate the gradient magnitude
    mag_grad = np.sqrt(sobel_x**2 + sobel_y**2)
    # Convert to 'cv2.CV_8U'.
    factor = np.max(mag_grad)/255
    mag_grad = (mag_grad/factor).astype(np.uint8)
    # Take the same shape and with setting it's values to 0
    binary_img = np.zeros_like(mag_grad)
    binary_img[(mag_grad >= thresh_range[0]) &
               (mag_grad <= thresh_range[1])] = 1

    # Black and white output image
    return binary_img

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_image_output = np.zeros_like(absgraddir)
    binary_image_output[(absgraddir >= thresh[0]) &
                        (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_image_output


def combined_thresholds(image, ksize=3):
    # Choose a Sobel kernel size
    # Apply each of the thresholding functions
    gradx = abs_sobel(image, direction='x',
                      kernel_size=ksize, thresh_range=(5, 100))
    mag_binary = calc_mag_thresh(
        image, kernel_size=ksize, thresh_range=(3, 255))
    dir_binary = dir_threshold(
        image, sobel_kernel=ksize, thresh=(45*np.pi/180, 75*np.pi/180))
    combined = np.zeros_like(dir_binary, np.uint8)
    combined[((gradx == 1) | (gradx == 1)) & (
        (mag_binary == 1) | (dir_binary == 1))] = 1
    return combined
#showImages(combined_thresholds, show_gray=True)

def Image_Filter(image):
    
    # Convert image from RGB to HLS 
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # get the HLS channels to HLS
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Choose a Sobel kernel size to detecte channel
    sxbinary = combined_thresholds(image)
    
    # Threshold color channel
    s_thresh_min = 80
    s_thresh_max = 255
    l_thresh_min = 190
    l_thresh_max = 255 
    
    #get the binary representation to this channels
    s_binary = np.zeros_like(s_channel)
    l_binary = np.zeros_like(s_channel)
    
    #Apple the theresould to this binary 
    s_binary[((s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)) ] = 1
    l_binary[((l_channel >= l_thresh_min) & (l_channel <= l_thresh_max))] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (sxbinary == 1)) | ((sxbinary == 1) & (l_binary == 1))] = 1
    
    return combined_binary
    
    
# warp and inwarp prespective

def warp(image):
    src = np.float32([[570, 460], [image.shape[1] - 573, 460],
                     [image.shape[1] - 150, image.shape[0]], [150, image.shape[0]]])
    dst = np.float32([[200, 0], [image.shape[1]-200, 0],
                     [image.shape[1]-200, image.shape[0]], [200, image.shape[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


def inwarp(image):
    src = np.float32([[570, 460], [image.shape[1] - 573, 460],
                     [image.shape[1] - 150, image.shape[0]], [150, image.shape[0]]])
    dst = np.float32([[200, 0], [image.shape[1]-200, 0],
                     [image.shape[1]-200, image.shape[0]], [200, image.shape[0]]])
    M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(
        image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


#land detection

class LaneDetector:

    def draw_lane(self, orignal_image, binary_warped, filtered_binary):

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50

        if self.detected:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(
                binary_warped[binary_warped.shape[0]//2:, :], axis=0)

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            nwindows = 9  # Choose the number of sliding windows
            # Set height of windows
            window_height = np.int((binary_warped.shape[0])/nwindows)

            leftx_current = leftx_base   # Current positions to be updated for each window
            rightx_current = rightx_base   # Set the width of the windows +/- margin

            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):

                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            self.left_lane_inds = np.concatenate(left_lane_inds)
            self.right_lane_inds = np.concatenate(right_lane_inds)
        else:
            self.left_lane_inds = ((nonzerox > (self.best_fit[0][0]*(nonzeroy**2) + self.best_fit[0][1]*nonzeroy +
                                    self.best_fit[0][2] - margin)) & (nonzerox < (self.best_fit[0][0]*(nonzeroy**2) +
                                                                                  self.best_fit[0][1]*nonzeroy + self.best_fit[0][2] + margin)))
            self.right_lane_inds = ((nonzerox > (self.best_fit[1][0]*(nonzeroy**2) + self.best_fit[1][1]*nonzeroy +
                                    self.best_fit[1][2] - margin)) & (nonzerox < (self.best_fit[1][0]*(nonzeroy**2) +
                                                                                  self.best_fit[1][1]*nonzeroy + self.best_fit[1][2] + margin)))
  
        
    # Again, extract left and right line pixel positions
        leftx = nonzerox[self.left_lane_inds]
        lefty = nonzeroy[self.left_lane_inds]
        rightx = nonzerox[self.right_lane_inds]
        righty = nonzeroy[self.right_lane_inds]

        ploty = np.linspace(
            0, binary_warped.shape[0]-1, binary_warped.shape[0])

        # Fit a second order polynomial to each
        if lefty.shape[0] >= 400 and righty.shape[0] >= 400 and leftx.shape[0] >= 400 and rightx.shape[0] >= 400:
            self.detected = False
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

            if len(self.recent_coefficients) >= self.n_frames:
                self.recent_coefficients.pop(0)
            self.recent_coefficients.append([self.left_fit, self.right_fit])

            self.best_fit = [0, 0, 0]
            for coefficient in self.recent_coefficients:
                self.best_fit[0] = self.best_fit[0] + coefficient[0]
                self.best_fit[1] = self.best_fit[1] + coefficient[1]

            self.best_fit[0] = self.best_fit[0]/len(self.recent_coefficients)
            self.best_fit[1] = self.best_fit[1]/len(self.recent_coefficients)

            # Generate x and y values for plotting
            left_fitx = self.best_fit[0][0]*ploty**2 + \
                self.best_fit[0][1]*ploty + self.best_fit[0][2]
            right_fitx = self.best_fit[1][0]*ploty**2 + \
                self.best_fit[1][1]*ploty + self.best_fit[1][2]

            if len(self.recent_xfitted) >= self.n_frames:
                self.recent_xfitted.pop(0)

            self.recent_xfitted.append([left_fitx, right_fitx])

            self.bestx = [np.zeros_like(
                720, np.float32), np.zeros_like(720, np.float32)]
            for fit in self.recent_xfitted:
                self.bestx[0] = self.bestx[0] + fit[0]
                self.bestx[1] = self.bestx[1] + fit[1]

            self.bestx[0] = self.bestx[0]/len(self.recent_xfitted)
            self.bestx[1] = self.bestx[1]/len(self.recent_xfitted)
            
            
            
#process image function

lane_detector = LaneDetector()
draw_lane = lane_detector.draw_lane
def process_image(image):
    undistorted = undistort_image(image)
    filtered_binary = Image_Filter(undistorted)
    binary_warped = warp(filtered_binary)
    binary_inwarped = inwarp(binary_warped)
    final_image = draw_lane(image, binary_warped, filtered_binary)
    sobel_image =combined_thresholds(image, ksize = 3)
    img1= cv2.resize(final_image,(0, 0), None,0.5, 0.5)
    img2 = cv2.resize(undistorted, (0, 0), None, 0.5, 0.5)
    img3 = np.dstack((filtered_binary, filtered_binary, filtered_binary)) * 255
    img3 = cv2.resize(img3, (0, 0), None, 0.5, 0.5)
    img4 = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    img4 = cv2.resize(img4, (0, 0), None, 0.5, 0.5)
    img5 = np.dstack((sobel_image, sobel_image, sobel_image)) * 255
    img5 = cv2.resize(img5, (0, 0), None, 0.5, 0.5)
    img6 = np.dstack((binary_inwarped, binary_inwarped, binary_inwarped)) * 255
    img6 = cv2.resize(img6, (0, 0), None, 0.5, 0.5)

    f_image1 = cv2.vconcat([img1, img2])
    f_image2 = cv2.vconcat([img3, img4])
    f_image3 = cv2.vconcat([img5, img6])
    f_image = np.concatenate((f_image1, f_image2, f_image3), axis=1)
    return f_image


showImages(process_image,show_gray=True)
            
            
   
