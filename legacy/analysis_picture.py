import cv2
from analysis.image_analysis import PendantDropAnalysis

analyzer = PendantDropAnalysis()
img = analyzer.select_image()
st, analyzed_img = analyzer.image2st(img=img)

# Resize the analyzed image to a smaller size
scale_percent = 30  # Percent of original size
width = int(analyzed_img.shape[1] * scale_percent / 100)
height = int(analyzed_img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
resized_img = cv2.resize(analyzed_img, dim, interpolation=cv2.INTER_AREA)

# Display the resized analyzed image using OpenCV
cv2.imshow("Analyzed Image", resized_img)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close the window


