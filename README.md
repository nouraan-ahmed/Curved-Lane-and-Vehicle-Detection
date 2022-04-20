# CURVED LANE DETECTION

## SUMMARY

In this Lane Detection project, we apply computer vision techniques to detect the lanes, highlight them using a fixed color, and pain the
area between them in green

## STEPS

1. camera calibration given a set of chessboard images.
2. Apply a distortion correction to images.
3. Use Sobel to detect edges.
4. Apply Filter.
5. Apply a perspective transform to rectify binary image (“birds-eye view”).
6. Detect lane pixels and fit to find the lane boundary.
7. Determine the curvature of the lane and vehicle position with respect to center.
8. Warp the detected lane boundaries back onto the original image.
9. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_before_distortion.jpg "Chessboard"
[image2]: ./output_images/chessboard_after_distortion.jpg "Chessboard Undistorted"
[image3]: ./output_images/before_undistort_image.jpg "Orignal Image"
[image4]: ./output_images/after_undistort_image.jpg "Undistorted"
[image5]: ./output_images/before_combined_thresholds.jpg "Before Gradient"
[image6]: ./output_images/after_combined_thresholds.jpg "After Gradient"
[image7]: ./output_images/before_filter.jpg "Before Color filter"
[image8]: ./output_images/after_filter.jpg "After Color filter"
[image9]: ./output_images/before_warp.jpg "Before warp"
[image10]: ./output_images/after_warp.jpg "After warp"
[image11]: ./examples/color_fit_lines.jpg "Identify poly"
[image12]: ./output_images/before_process_image.jpg "Before Process"
[image13]: ./output_images/after_process_image.jpg "After process"


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "Lane Detection.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Before Undestortion
![alt text][image1]

After Undestortion
![alt text][image2]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Before 
![alt text][image3]

After
![alt text][image4]

#### 2. Perspective Transform from Camera Angle to Bird's Eye View

To calucluate curvature, the ideal perspective is a bird's eye view. This means that the road is perceived from above, instead of at an angle through the vehicle's windshield.

This perspective transform is computed using a straight lane scenario and prior common knowledge that the lane lines are in fact parallel. Source and destination points are identified directly from the image for the perspective transform.

OpenCV provides perspective transform functions to calculate the transformation matrix for the images given the source and destination points. Using warpPerspective function, the bird's eye view perspective transform is performed.

Before Warp Perspective
![alt text][image9]

After  Warp Perspective
![alt text][image10]


