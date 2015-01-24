import cv2
import numpy as np
import datetime

# the index for the camera (0 should be integrated laptop cam)
camera_port = 0

# the number of frames to discard before saving
ramp_frames = 10

camera = cv2.VideoCapture(camera_port)
#set the width and height
# reference: http://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
camera.set(3,1280)
camera.set(4,720)

# the number of images to capture before blending
frame_buffer_size = 1
# a list to store the captured images in
frame_buffer = []

def get_image():
	retval, im = camera.read()
	
	# we need 32-bit floating point images ( CV_32F), so convert it before returning
	# reference: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
	# reference: http://stackoverflow.com/questions/8976502/convert-8-bit-image-to-32-bit-in-opencv-2
	#float_im = cv2.cvtColor(im, cv2.CV_32F)

	# NB: I could not get the OpenCV functions to work as expected	
	# so instead I treated the images as numpy arrays of floats
	float_im = np.array(im, dtype=np.float32) 
	
	return float_im

# the blend function merges all the images from the image buffer into one single image
# the objective here is to reduce noise and flicker, and add in motion-blur on movement 
def blend():
	result = frame_buffer[0]
	
	print("Image color depth =" + str(result.dtype))
	
	# used as a multiplyer to scale the averaged image
	alpha = 1.0 / frame_buffer_size
	
	for i in range(1, frame_buffer_size):
		# usage note from the documentation (http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?highlight=addweighted#addweighted)
		# cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) → dst
		
		#cv2.addWeighted(result, alpha, frame_buffer[i], beta, 0.0, result)
		# the above takes the result frame (starts as frame 0), and blends it 50/50 with the 
		# next frame in the frame_buffer. It saves the blended image back into result, 
		# ready to be blended with the next frame
		
		# sum the frames together
		# reference: http://docs.opencv.org/trunk/modules/imgproc/doc/motion_analysis_and_object_tracking.html

		# NB: I could not get the OpenCV functions to work as expected
		# so instead I treated the images as numpy arrays of floats
		# nd used numpy to sum then all up here
		result = result + frame_buffer[i]
		
	
	# average the summed images 
	# reference: http://docs.opencv.org/modules/core/doc/old_basic_structures.html#cv.ConvertScale
	# reference: http://stackoverflow.com/questions/18461623/average-values-in-two-numpy-arrays

	# treating the image as a numpy array, multiply it by alpha
	# (or devide it by frame_buffer_length) here
	result = result * alpha
	return result

def save_tmp(image, index):
	file = "../frames/tmp/" + str(index) + ".png"
	cv2.imwrite(file, image)
	
	
def add_overlay(image):
	color = [0.2, 0.2, 0.2]
	today = datetime.datetime.now()
	
	day_text = today.strftime("DAY %j")
	date_text = today.strftime("%B %d, %Y %H:%M")
	
	x = 10 #position of text
	y = 680 #position of text
	
	# reference; http://www.cplusplus.com/forum/beginner/65318/
	# draw the DAY XXX data
	#font = cv2.initFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.1, 1.1, 0, 3, 8) #Creates a font
	cv2.putText(image, day_text, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) #Draw the text
	
	# draw the date in smaller text
	#font = cv2.initFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8) #Creates a font
	cv2.putText(image, date_text, (x,y+25),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2) #Draw the text
	return image

	
for i in xrange(ramp_frames):
		temp = get_image()
		
for c in xrange(frame_buffer_size):		
	print("getting image #" + str(c))
		
	print("adding image to frame buffer...")
	image = get_image()

	# I commented this out. If you include it, it stores all the
	# recorded frames in the frames/tmp/ directory for debugging
	#save_tmp(image, c)
	
	frame_buffer.append(image)
	
	
	
file = "../frames/test_blend_image.png"

blended_frame = blend();

blended_frame = add_overlay(blended_frame)
# write the rsult of the blend() function to the file
cv2.imwrite(file, blended_frame)

del(camera)
del(frame_buffer)
