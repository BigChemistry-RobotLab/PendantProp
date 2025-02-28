import cv2
import imutils
import numpy as np
import itertools
from tkinter import Tk
from scipy.spatial.distance import euclidean
from tkinter.filedialog import askopenfilename
import numpy as np
from scipy.spatial import KDTree
import cv2
import itertools
from scipy.spatial.distance import euclidean
import os

from utils.load_save_functions import load_settings
from utils.logger import Logger


class PendantDropAnalysis:
    def __init__(self):
        self.settings = load_settings()
        self.density = float(self.settings["DENSITY"])
        self.scale = float(self.settings["SCALE"])
        self.gravity_constant = 9.80665
        self.file_path = None
        self.raw_image = None
        self.processed_image = None
        self.analysis_image = None

    def init_logger(self):
        self.logger = Logger(
            name="analysis",
            file_path=f'experiments/{self.settings["EXPERIMENT_NAME"]}/meta_data',
        )

    def select_image(self):
        # Create Tkinter root window
        root = Tk()
        root.withdraw()  # Hide the root window

        # Prompt the user to select an image file
        self.file_path = askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )

        # Read the selected image file
        self.raw_image = cv2.imread(self.file_path)
        return self.raw_image

    def load_raw_image(self, file_path: str):
        self.file_path = file_path
        self.raw_image = cv2.imread(self.file_path)

    def process_image(self):

        blur = cv2.GaussianBlur(self.raw_image, (9, 9), 0)
        canny = cv2.Canny(blur, 10, 10)
        edged = cv2.dilate(canny, None, iterations=1)
        self.processed_image = cv2.erode(edged, None, iterations=1)

    def analyse(self):

        # some constants
        text_y_offset = 100

        # Find contours on processed image
        contours = imutils.grab_contours(
            cv2.findContours(
                image=self.processed_image.copy(),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )
        )

        # Sort contours by the width of the bounding box in descending order
        # Keep only the broadest contour which should be the droplet
        longest_contour = sorted(
            contours, key=lambda c: cv2.boundingRect(c)[2], reverse=True
        )[0]

        # Draw the longest contour on the original image
        self.analysis_image = self.raw_image.copy()
        cv2.drawContours(
            image=self.analysis_image,
            contours=[longest_contour],
            contourIdx=-1,
            color=(252, 3, 103),
            thickness=10,
        )

        # Find the bounding rectangle for the contour + De calculated
        x, y, w, h = cv2.boundingRect(longest_contour)
        de = w  #! important for calculation st
        self.de = de

        # Draw arrowline + De
        touching_points_left = []
        for point in longest_contour:
            px, py = point[0]
            if px == x:
                touching_points_left.append(point[0])
        left_pt_de_line = (touching_points_left[0][0], touching_points_left[0][1])
        right_pt_de_line = (touching_points_left[0][0] + w, touching_points_left[0][1])
        cv2.arrowedLine(
            img=self.analysis_image,
            pt1=left_pt_de_line,
            pt2=right_pt_de_line,
            color=(255, 127, 14),
            thickness=10,
        )
        cv2.arrowedLine(
            img=self.analysis_image,
            pt1=right_pt_de_line,
            pt2=left_pt_de_line,
            color=(255, 127, 14),
            thickness=10,
        )
        cv2.putText(
            img=self.analysis_image,
            text=f"de={de:.0f}px",
            org=(left_pt_de_line[0], left_pt_de_line[1] + text_y_offset),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 127, 14),
            thickness=3,
        )
        # Compute the coordinates of the rectangle corners
        top_left = (x, y)
        top_right = (x + w, y)
        bottom_right = (x + w, y + h)
        bottom_left = (x, y + h)

        # Create new blank image to redraw biggest contour and crop above the ds
        cropped_image = np.zeros_like(self.raw_image)
        cv2.drawContours(cropped_image, [longest_contour], -1, (0, 255, 0), thickness=10)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_image = cropped_image[
            int(top_left[1]) : int(bottom_left[1] - (de)),
            int(top_left[0]) : int(top_right[0]),
        ]
        # find new contours in cropped image
        cnts_2, _ = cv2.findContours(
            cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Assuming cnts_2 is available and contains the contours
        contourright, contourleft = max(
            itertools.combinations(cnts_2, 2),
            key=lambda pair: euclidean(
                cv2.minEnclosingCircle(pair[0])[0], cv2.minEnclosingCircle(pair[1])[0]
            ),
        )

        # Calculate the horizontal distance between the two farthest points of the contours
        Lx, Ly, Lw, Lh = cv2.boundingRect(contourleft)
        Rx, Ry, Rw, Rh = cv2.boundingRect(contourright)
        ds = Rx + Rw - Lx  # This assumes the contours are ordered left to right

        # Draw the line for the maximum distance
        Lx_adjusted, Ly_adjusted = Lx + top_left[0], Ly + top_left[1]
        Rx_adjusted, Ry_adjusted = Rx + top_left[0], Ry + top_left[1]

        # Calculate a new Y-coordinate for drawing the max_distance line and text
        new_y_left = Ly + Lh  # Bottom of the left contour
        new_y_right = Ry + Rh  # Bottom of the right contour
        new_y_position = max(
            new_y_left, new_y_right
        )  # Use the lower of the two for drawing

        # # Adjust the drawing of the maximum distance line and text with the new Y-coordinate
        cv2.arrowedLine(
            self.analysis_image,
            (Lx_adjusted, new_y_position),  # Use new Y-coordinate for left contour
            (
                Rx_adjusted + Rw,
                new_y_position,
            ),  # Use new Y-coordinate for right contour, ensuring line is horizontal
            (31, 119, 180),  # Color for distinction
            thickness=10,
        )
        cv2.arrowedLine(
            self.analysis_image,
            (
                Rx_adjusted + Rw,
                new_y_position,
            ),
            (Lx_adjusted, new_y_position),  # Use new Y-coordinate for left contour
            (31, 119, 180),  # Color for distinction
            thickness=10,
        )

        # # Adjust the text position to be slightly above or below the line
        cv2.putText(
            self.analysis_image,
            f"ds={ds:.0f}px",
            (Lx_adjusted, new_y_position + text_y_offset),  # Adjusted for visibility
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(31, 119, 180),  # Color for distinction
            thickness=3,
        )

        left_visual = contourleft.copy()
        left_visual[:, :, 0] += x

        right_visual = contourright.copy()
        right_visual[:, :, 0] += x

        S = ds / de
        Hin = self._calculate_Hin(S)
        self.Hin = Hin
        de_scaled = de * self.scale # pixels -> mm
        surface_tension = self.density * self.gravity_constant * (de_scaled**2) * Hin

        # Draw the surface_tension on the visual_image
        cv2.putText(
            self.analysis_image,
            f"surface_tension: {surface_tension:.2f} mN/m",
            (10, self.analysis_image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            2,
        )

        return surface_tension

    def _calculate_Hin(self, S):
        if not (0.3 < S < 1):
            # self.logger.error("analysis: shape factor S is out of bounds")
            pass

        # find value for 1/H for different values of S
        if (S >= 0.3) and (S <= 0.4):
            Hin = (
                (0.34074 / (S**2.52303))
                + (123.9495 * (S**5))
                - (72.82991 * (S**4))
                + (0.01320 * (S**3))
                - (3.38210 * (S**2))
                + (5.52969 * (S))
                - 1.07260
            )
        if (S > 0.4) and (S <= 0.46):
            Hin = (
                (0.32720 / (S**2.56651))
                - (0.97553 * (S**2))
                + (0.84059 * S)
                - (0.18069)
            )
        if (S > 0.46) and (S <= 0.59):
            Hin = (
                (0.31968 / (S**2.59725))
                - (0.46898 * (S**2))
                + (0.50059 * S)
                - (0.13261)
            )
        if (S > 0.59) and (S <= 0.68):
            Hin = (
                (0.31522 / (S**2.62435))
                - (0.11714 * (S**2))
                + (0.15756 * S)
                - (0.05285)
            )
        if (S > 0.68) and (S <= 0.9):
            Hin = (
                (0.31345 / (S**2.64267))
                - (0.09155 * (S**2))
                + (0.14701 * S)
                - (0.05877)
            )
        if (S > 0.9) and (S <= 1):
            Hin = (
                (0.30715 / (S**2.84636))
                - (0.69116 * (S**3))
                + (1.08315 * (S**2))
                - (0.18341 * S)
                - (0.20970)
            )
        return Hin

    def image2st(self, img):
        self.init_logger()
        self.raw_image = img    
        self.process_image()
        st = self.analyse()
        return st, self.analysis_image

    def image2scale(self, img):
        # surface_tension = self.density * self.gravity_constant * (de_scaled**2) * Hin
        self.init_logger()
        self.raw_image = img    
        self.process_image()
        st = self.analyse()
        st_water = (
            72.37  # mN/m, 22.5 degrees C, see https://srd.nist.gov/JPCRD/jpcrd231.pdf
        )
        de_scaled = np.sqrt(st_water / (self.density * self.gravity_constant * self.Hin))
        scale = de_scaled / self.de
        return scale

    def show_raw_image(self):
        cv2.imshow(winname=self.file_path, mat=self.raw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_processed_image(self):
        cv2.imshow(winname=self.file_path, mat=self.processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_analysis_image(self):
        cv2.imshow(winname=self.file_path, mat=self.analysis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_analysis_image(self, file_path = None):
        if file_path == None:
            file_path = f"experiments/{self.settings['EXPERIMENT_NAME']}/data/analysis.jpg"
        cv2.imwrite(file_path, self.analysis_image)
