import cv2

# the index for the camera (0 should be integrated laptop cam)
camera_port = 0

# the number of frames to discard before saving
ramp_frames = 30

camera = cv2.VideoCapture(camera_port)

def get_image():
	retval, im = camera.read()
	return im
	
print("Skipping ramp frames...")
for i in xrange(ramp_frames):
	temp = get_image()
	
print("Taking Image...")

camera_capture = get_image()

file = "../frames/test_image.png"

cv2.imwrite(file, camera_capture)

del(camera)
