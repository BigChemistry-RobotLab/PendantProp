import cv2
from analysis.image_analysis import PendantDropAnalysis
import pandas as pd

drop_volumes = [1, 2, 3, 4, 5, 6, 7, 8]
df = pd.DataFrame()
for drop_volume in drop_volumes:
    filename = f"data/worthington_dropvolume/pictures_paper/{drop_volume}.png"
    img = cv2.imread(filename=filename)
    analyzer = PendantDropAnalysis()
    wo = analyzer.image2wortington(img=img, vol_droplet=drop_volume)
    st, img_ana = analyzer.image2st(img=img)

    row = pd.DataFrame(
        [{"drop_volume": drop_volume, "worthington_output": wo, "surface_tension": st}]
    )
    df = pd.concat([df, row], ignore_index=True)

df.to_csv("data/worthington_dropvolume/picture_data.csv")