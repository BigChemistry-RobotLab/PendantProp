
import cv2
from pendantprop.analysis.image_analysis import PendantDropAnalysis
from opentrons_api.load_save_functions import load_settings

settings = load_settings(file_path="config/settings.json")
analyzer = PendantDropAnalysis(settings=settings)

# img = cv2.imread("docs/example_drop.png")
img = analyzer.select_image()
st, wo, analysis_img = analyzer.analyse_image(img=img, vol_droplet=10)
print(f"Surface Tension: {st} mN/m")
print(f"Worthington Number: {wo}")

# Resize to 50% for display
width = int(analysis_img.shape[1] * 0.5)
height = int(analysis_img.shape[0] * 0.5)
resized = cv2.resize(analysis_img, (width, height), interpolation=cv2.INTER_AREA)

cv2.imshow("Analysis Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()