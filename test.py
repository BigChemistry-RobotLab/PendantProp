from analysis.image_analysis import PendantDropAnalysis
import cv2

img = cv2.imread("test_image.png")
analyzer = PendantDropAnalysis()
analyzer.load_raw_image(file_path="test_image.png")
analyzer.process_image()
st = analyzer.analyse()
print(analyzer.check_wortington(vol_droplet=11, st=st))