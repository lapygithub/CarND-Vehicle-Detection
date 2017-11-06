## CarND - Vehicle Detection Writeup
---
** Vehicle Detection Project **

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car_mrl.png
[image2]: ./examples/HOG_example_mrl.png
[image3]: ./examples/sliding_windows_mrl.png
[image4]: ./examples/bboxes_and_heat_mrlp9.png
[image5]: ./examples/bboxes_and_heat_mrl.png
[image50]: ./examples/bboxes_and_heat_mrl1.png
[image51]: ./examples/bboxes_and_heat_mrl2.png
[image52]: ./examples/bboxes_and_heat_mrl3.png
[image53]: ./examples/bboxes_and_heat_mrl4.png
[image54]: ./examples/bboxes_and_heat_mrl5.png
[image55]: ./examples/bboxes_and_heat_mrl6.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  The code for this step is contained in the code cell of the IPython notebook labeled `1) Car Not-Car Training Data`).  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

In IPython notebook cell labeled `2a) Params and Color Spaces Example`, I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` and `LUV` color spaces with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and color spaces settling on the default starting parameters and the `LUV` color space to begin work on the classifier. The HOG level of detail was indistinguishable between `YCrCb` and `LUV` to my naked eye anyway.  I may change the color space performance warrants during classification experimentation.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In cell `3) Train SVM` of `P5_Vechicle_Detection`, the classifier is trained using SCIKIT Image linear SVM and a combination of color, histogram and HOG (all channels) features.  Training and testing of classification was randomized and split.  Output shows a test accuracty of 98%.

```
16.96 Seconds to extract HOG features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
1.0 Seconds to train SVC...
Test Accuracy of SVC =  0.98
My SVC predicts:      [ 1.  1.  0.  0.  1.  1.  1.  0.  1.  1.]
For these 10 labels:  [ 1.  1.  0.  0.  1.  0.  1.  0.  1.  1.]
0.00665 Seconds to predict 10 labels with SVC
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at 3 scales relating to vehicles in the distance for smaller images, midway for medium scale and finally closer for larger scale vechicle images.  Scaling directions for each image can be found in cell `8) Process Image` of the project Jupyter notebook in the function process_image() which uses the function slide_window() defined in cell `5) Sliding Window & Draw Boxes`.  Overlap was initially started and 0.5 and increased with the heat map implementation to 0.75.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using spatially binned color and histograms of color plus LUV 3-channel HOG features in the feature vector. The results had lots of false positives as shown below. Ultimately, the left side of the image was ignored even after trying to classify out the guardrail by adding many, many frames to the non-vechicle training set.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project5_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video, and for each frame, applied thresholding and clipped the remaining values to 1 to remove multiple detections. Up to 5 frames are stored in a Python collection deque object. From the sum of the up to 5 queued positive detections I created a heatmap and then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding labels and heatmaps:

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I had with this projects feature based supervised learning is working out the false positives.  Guardrails look to be specifically hard to distinguish from vechicles.  In addition, the heatmaps are just a bandaid for removing false postives and do not remove false positives that exist over a number of frames.  If I were to pursue this project further, I would try training a CNN instead of using HOG. 

