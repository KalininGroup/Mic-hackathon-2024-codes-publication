import scipy.io
from pathlib import Path
from tqdm import tqdm
from skimage.measure import regionprops
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd


class ContourAnalyzer:
    def __init__(self, output_shape=(512, 512)):
        """
        Initialize ContourAnalyzer with output image shape.

        Parameters:
        output_shape (tuple): Desired shape (height, width) for output visualizations
        """
        self.output_shape = output_shape
        self.all_contours = []
        self.filenames = []
        self.all_properties = []

    def _get_default_properties(self):
        """Return default properties for invalid contours."""
        return {
            'area': 0,
            'perimeter': 0,
            'centroid_y': 0,
            'centroid_x': 0,
            'major_axis_length': 0,
            'minor_axis_length': 0,
            'aspect_ratio': float('inf'),
            'solidity': 0,
            'circularity': 0,
            'first_hu_moment': 1.0,
            'convexity': 1.0
        }

    def analyze_single_contour(self, boundary_points):
        """
        Analyze a single contour and return its geometric properties.

        Parameters:
        boundary_points: numpy array of shape (N, 2) containing (x, y) coordinates

        Returns:
        dict: Dictionary containing geometric properties of the contour
        """
        try:
            # Input validation
            if not isinstance(boundary_points, np.ndarray):
                boundary_points = np.array(boundary_points)

            if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
                return self._get_default_properties()

            if len(boundary_points) < 3:
                return self._get_default_properties()

            # Create a binary mask from the contour points
            x_max, y_max = np.max(boundary_points, axis=0) + 1
            if x_max <= 0 or y_max <= 0:
                return self._get_default_properties()

            mask = np.zeros((int(y_max), int(x_max)), dtype=np.uint8)
            points = boundary_points.astype(np.int32)
            cv2.fillPoly(mask, [points], 1)

            # Check if mask is empty
            if np.sum(mask) == 0:
                return self._get_default_properties()

            # Calculate properties using regionprops
            props_list = regionprops(mask)
            if not props_list:
                return self._get_default_properties()

            props = props_list[0]

            # Calculate perimeter using contour length
            try:
                perimeter = cv2.arcLength(points.reshape(-1, 1, 2), closed=True)
            except:
                perimeter = 0

            # Calculate convex hull perimeter
            try:
                hull = cv2.convexHull(points.reshape(-1, 1, 2))
                convex_perimeter = cv2.arcLength(hull, closed=True)
            except:
                convex_perimeter = 0

            # Calculate Hu Moments
            try:
                moments = cv2.moments(mask)
                hu_moments = cv2.HuMoments(moments)
                first_hu_moment = hu_moments[0][0]
            except:
                first_hu_moment = 0

            return {
                'area': props.area,
                'perimeter': perimeter,
                'centroid_y': props.centroid[0],
                'centroid_x': props.centroid[1],
                'eccentricity': props.eccentricity,
                'major_axis_length': props.major_axis_length,
                'minor_axis_length': props.minor_axis_length,
                'aspect_ratio': props.major_axis_length / props.minor_axis_length if props.minor_axis_length > 0 else float(
                    'inf'),
                'solidity': props.solidity,
                'circularity': 4 * np.pi * props.area / (perimeter ** 2) if perimeter > 0 else 0,
                'first_hu_moment': 1/first_hu_moment,
                'convexity': convex_perimeter / perimeter if perimeter > 0 else 1.0
            }
        except Exception as e:
            print(f"Error processing contour: {str(e)}")
            return self._get_default_properties()

    def save_visualizations(self, save_path):
        """
        Save visualizations of all contours.

        Parameters:
        save_path (Path or str): Path where visualizations should be saved
        """
        if not self.all_contours:
            return

        save_path = Path(save_path)
        shape = self.output_shape
        dpi = 100
        figsize = (shape[1] / dpi, shape[0] / dpi)

        # Generate random colors for each contour
        colors = [(random.random(), random.random(), random.random())
                  for _ in range(len(self.all_contours))]

        try:
            # First visualization (original)
            plt.figure(figsize=figsize, dpi=dpi)
            img1 = np.ones(shape + (3,), dtype=np.uint8) * 255

            for idx, (contour, color) in enumerate(zip(self.all_contours, colors)):
                contour = self._scale_contour(contour)
                cv2.fillPoly(img1, [contour.astype(np.int32)],
                             (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
                plt.fill(0, 0, color=color, label=Path(self.filenames[idx]).stem)

            plt.imshow(img1)
            plt.axis('off')
            plt.savefig(save_path, bbox_inches=None, pad_inches=0, dpi=dpi)
            plt.close()

            # Second visualization (with labels)
            plt.figure(figsize=figsize, dpi=dpi)
            img2 = np.ones(shape + (3,), dtype=np.uint8) * 255

            for idx, (contour, color) in enumerate(zip(self.all_contours, colors)):
                contour = self._scale_contour(contour)
                cv2.fillPoly(img2, [contour.astype(np.int32)],
                             (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))

                self._add_label_to_contour(img2, contour, self.filenames[idx])

            plt.imshow(img2)
            plt.axis('off')
            labeled_save_path = str(save_path).replace('.tiff', '_labeled.tiff')
            plt.savefig(labeled_save_path, bbox_inches=None, pad_inches=0, dpi=dpi)
            plt.close()
        except Exception as e:
            print(f"Error saving visualizations: {str(e)}")

    def _scale_contour(self, contour):
        """Scale and center a contour within the output shape."""
        try:
            contour = contour.astype(np.float32)
            if contour.size == 0:
                return np.zeros((0, 2), dtype=np.float32)

            scale_x = self.output_shape[1] / np.max(contour[:, 0]) * 0.8 if np.max(contour[:, 0]) > 0 else 1
            scale_y = self.output_shape[0] / np.max(contour[:, 1]) * 0.8 if np.max(contour[:, 1]) > 0 else 1
            scale = min(scale_x, scale_y)

            contour[:, 0] = contour[:, 0] * scale + self.output_shape[1] * 0.1
            contour[:, 1] = contour[:, 1] * scale + self.output_shape[0] * 0.1

            return contour
        except Exception:
            return np.zeros((0, 2), dtype=np.float32)

    def _add_label_to_contour(self, img, contour, filename):
        """Add a label to a contour in the image."""
        try:
            M = cv2.moments(contour.astype(np.int32))
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                label = Path(filename).stem
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

                text_x = cx - text_width // 2
                text_y = cy + text_height // 2

                cv2.putText(img, label, (text_x, text_y), font, font_scale,
                            (255, 255, 255), thickness + 1)
                cv2.putText(img, label, (text_x, text_y), font, font_scale,
                            (0, 0, 0), thickness)
        except Exception as e:
            print(f"Error adding label: {str(e)}")

    def process_mat_files(self, mat_folder, vis_folder=None):
        """
        Process all .mat files in a folder and generate analysis.

        Parameters:
        mat_folder (str or Path): Path to folder containing .mat files
        vis_folder (str or Path, optional): Path to save visualizations. If None,
                                          creates 'visualizations' subfolder in mat_folder

        Returns:
        pd.DataFrame: DataFrame containing geometric properties of all contours
        """
        mat_folder = Path(mat_folder)
        if vis_folder is None:
            vis_folder = mat_folder / 'visualizations'
        vis_folder = Path(vis_folder)
        vis_folder.mkdir(exist_ok=True)

        mat_files = list(mat_folder.glob('*.mat'))

        # Reset stored data
        self.all_contours = []
        self.filenames = []
        self.all_properties = []

        # First pass to collect all contours
        for mat_file in mat_files:
            try:
                mat = scipy.io.loadmat(mat_file)
                cont_points = mat['output'][0][0][0]
                self.all_contours.append(cont_points)
                self.filenames.append(mat_file.name)
            except Exception as e:
                print(f"Error loading {mat_file}: {str(e)}")
                continue

        # Analyze each contour
        for mat_file in tqdm(mat_files):
            try:
                mat = scipy.io.loadmat(mat_file)
                cont_points = mat['output'][0][0][0]
                properties = self.analyze_single_contour(cont_points)
                properties['mat_file_name'] = mat_file.stem
                self.all_properties.append(properties)
            except Exception as e:
                print(f"Error processing {mat_file}: {str(e)}")
                properties = self._get_default_properties()
                properties['mat_file_name'] = mat_file.stem
                self.all_properties.append(properties)

        try:
            # Create and save visualizations
            self.save_visualizations(vis_folder / 'all_contours.tiff')

            # Create DataFrame
            df = pd.DataFrame(self.all_properties)

            # Reorder columns to put identifier first
            cols = ['mat_file_name'] + [col for col in df.columns if col != 'mat_file_name']
            df = df[cols]

            # Save DataFrame to CSV
            csv_path = vis_folder / 'geometric_properties.csv'
            df.to_csv(csv_path, index=False)

            return df
        except Exception as e:
            print(f"Error creating output: {str(e)}")
            return pd.DataFrame(self.all_properties)


if __name__ == "__main__":
    analyzer = ContourAnalyzer(output_shape=(512, 512))
    base_folder = '/Users/vnpawan/Downloads/Hackathon'
    results_df = analyzer.process_mat_files(base_folder)
