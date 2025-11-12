
from pendantprop.analysis.image_analysis import PendantDropAnalysis
from opentrons_api.load_save_functions import load_settings

settings = load_settings(file_path="config/settings.json")
analyzer = PendantDropAnalysis(settings=settings)

analyzer.load_raw_image(file_path="docs/example_drop.png")
analyzer.process_image()

st = analyzer.analyse()
print(f"Calculated surface tension: {st} mN/m")

print(f"Needle diameter within tolerance: {analyzer.check_diameter()}")
print(f"measured needle diameter (px): {analyzer.needle_diameter_px_measured}")
print(f"given needle diameter (px): {analyzer.needle_diameter_px}")

wo = analyzer.img2wo(img=analyzer.raw_image, vol_droplet=8)
print(f"Calculated Wo number: {wo}")

# analyzer.show_analysis_image()