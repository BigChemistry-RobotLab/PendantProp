import cv2
import imutils
import numpy as np
import itertools
from tkinter import Tk
from scipy.spatial.distance import euclidean
from tkinter.filedialog import askopenfilename

from utils.load_save_functions import load_settings


class PendantDropAnalysis:
    def __init__(self):
        self.settings = load_settings()
        self.density = float(self.settings["DENSITY"])
        self.needle_diameter_mm = self.settings["DIAMETER_NEEDLE_MM"]
        self.needle_diameter_px = None
        # self.scale = float(self.settings["SCALE"])
        self.gravity_constant = 9.80665
        self.file_path = None
        self.raw_image = None
        self.processed_image = None
        self.analysis_image = None

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
        overlay = self.analysis_image.copy()
        cv2.drawContours(
            image=self.analysis_image,
            contours=[longest_contour],
            contourIdx=-1,
            color=(40, 39, 150),
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
        self._draw_double_arrow_line(
            image=self.analysis_image,
            point1=left_pt_de_line,
            point2=right_pt_de_line,
            color=(255, 127, 14),  # Orange color
            thickness=10,
            tip_length=0.05,
            text=f"de={de:.0f}px",  # Add the text for the line
        )

        # Compute the coordinates of the rectangle corners
        top_left = (x, y)
        top_right = (x + w, y)
        bottom_left = (x, y + h)

        # Create new blank image to redraw biggest contour and crop above the ds
        cropped_image = np.zeros_like(self.raw_image)
        cv2.drawContours(cropped_image, [longest_contour], -1, (0, 255, 0), thickness=2)
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

        right_point_needle = contourright[-1][0]
        left_point_needle = contourleft[0][0]
        self.needle_diameter_px = right_point_needle[0] - left_point_needle[0]
        self.scale = self.needle_diameter_mm / self.needle_diameter_px
        # Adjust the coordinates of the needle points to the original image

        offset_y = 40

        right_point_needle_transposed = (
            right_point_needle[0] + top_left[0],
            right_point_needle[1] + top_left[1] + offset_y,
        )
        left_point_needle_transposed = (
            left_point_needle[0] + top_left[0],
            left_point_needle[1] + top_left[1] + offset_y,
        )

        # Draw the double-arrow line for the needle diameter on the analysis image
        self._draw_double_arrow_line(
            image=self.analysis_image,
            point1=left_point_needle_transposed,
            point2=right_point_needle_transposed,
            color=(44, 160, 44),
            thickness=5,
            tip_length=0.05,
            fontscale=0.5,
            text=f"Needle D: {self.needle_diameter_px}px",
            thickness_font=2,
        )

        # Calculate the horizontal distance between the two farthest points of the contours
        Lx, Ly, Lw, Lh = cv2.boundingRect(contourleft)
        Rx, Ry, Rw, Rh = cv2.boundingRect(contourright)
        ds = Rx + Rw - Lx  # This assumes the contours are ordered left to right

        # Draw the line for the maximum distance
        Lx_adjusted, _ = Lx + top_left[0], Ly + top_left[1]
        Rx_adjusted, _ = Rx + top_left[0], Ry + top_left[1]

        # Calculate a new Y-coordinate for drawing the max_distance line and text
        new_y_left = Ly + Lh  # Bottom of the left contour
        new_y_right = Ry + Rh  # Bottom of the right contour
        new_y_position = max(
            new_y_left, new_y_right
        )  # Use the lower of the two for drawing

        self._draw_double_arrow_line(
            image=self.analysis_image,
            point1=(Lx_adjusted, new_y_position),  # Left contour point
            point2=(Rx_adjusted + Rw, new_y_position),  # Right contour point
            color=(31, 119, 180),  # Color for distinction
            thickness=10,
            tip_length=0.05,
            text=f"ds={ds:.0f}px",  # Add the text for the line
        )

        S = ds / de
        Hin = self._calculate_Hin(S)
        self.Hin = Hin
        de_scaled = de * self.scale  # mm -> pixels
        surface_tension = self.density * self.gravity_constant * (de_scaled**2) * Hin

        alpha = 0  # Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)
        cv2.addWeighted(
            overlay, alpha, self.analysis_image, 1 - alpha, 0, self.analysis_image
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

    def _draw_double_arrow_line(
        self,
        image,
        point1,
        point2,
        color=(0, 255, 255),
        thickness=2,
        tip_length=0.05,
        text=None,
        fontscale=2,
        thickness_font=3,
    ):
        """
        Draws a double-arrow line between two points on the given image.

        Args:
            image (numpy.ndarray): The image on which to draw the line.
            point1 (tuple): The starting point of the line (x, y).
            point2 (tuple): The ending point of the line (x, y).
            color (tuple): The color of the arrow line in BGR format. Default is yellow.
            thickness (int): The thickness of the arrow line. Default is 2.
            tip_length (float): The length of the arrow tip relative to the arrow length. Default is 0.05.
            text (str): Optional text to display at the midpoint of the line.
        """
        # Draw arrow from point1 to point2
        cv2.arrowedLine(
            image,
            tuple(point1),
            tuple(point2),
            color,
            thickness=thickness,
            tipLength=tip_length,
        )

        # Draw arrow from point2 to point1
        cv2.arrowedLine(
            image,
            tuple(point2),
            tuple(point1),
            color,
            thickness=thickness,
            tipLength=tip_length,
        )

        # Add optional text at the midpoint of the line
        if text:
            mid_x = (point1[0] + point2[0]) // 2
            mid_y = (point1[1] + point2[1]) // 2

            # Get the text size
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, thickness=1
            )[0]
            text_width, text_height = text_size

            # Adjust the text position to center it and move it slightly above the line
            text_x = mid_x - (text_width // 2)
            text_y = mid_y - (text_height // 2) - 5  # Move text slightly above the line

            # Draw the text
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontscale,
                color=color,
                thickness=thickness_font,
            )

    def _calculate_wortington(self, vol_droplet, st):
        """
        taken from https://doi.org/10.1016/j.jcis.2015.05.012.
        units cancel out if st given in mN/m and needle_diameter in mm and vol droplet in uL
        """

        Wo = (self.density * self.gravity_constant * vol_droplet) / (
            np.pi * st * self.needle_diameter_mm
        )
        return Wo

    def check_diameter(self):
        diameter_needle_px_given = self.settings["NEEDLE_DIAMETER_PX"]
        if (
            0.95 * diameter_needle_px_given
            < self.needle_diameter_px
            < 1.05 * diameter_needle_px_given
        ):
            return True
        else:
            # print(f"too large of diameter ({self.needle_diameter_px} px), droplet probably sticking to needle.")
            return False

    def image2wortington(self, img, vol_droplet):
        self.raw_image = img
        self.process_image()
        st = self.analyse()
        if st > 20:
            if self.check_diameter():
                return self._calculate_wortington(vol_droplet=vol_droplet, st=st)
            else:
                return 0
        else:
            return 0

    def image2st(self, img):
        self.raw_image = img
        self.process_image()
        st = self.analyse()
        return st, self.analysis_image

    def image2scale(self, img):
        "legacy"
        # surface_tension = self.density * self.gravity_constant * (de_scaled**2) * Hin
        self.raw_image = img
        self.process_image()
        _ = self.analyse()
        st_water = (
            72.37  # mN/m, 22.5 degrees C, see https://srd.nist.gov/JPCRD/jpcrd231.pdf
        )
        de_scaled = np.sqrt(
            st_water / (self.density * self.gravity_constant * self.Hin)
        )
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

    def save_analysis_image(self, file_path=None, experiment_dir="experiments"):
        if file_path is None:
            file_path = (
                f"{experiment_dir}/{self.settings['EXPERIMENT_NAME']}/data/analysis.jpg"
            )
        cv2.imwrite(file_path, self.analysis_image)
