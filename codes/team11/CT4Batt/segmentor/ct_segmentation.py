# Use PySimpleGUIWx for dialogs
import PySimpleGUIWx as sg

# use these settings for auto-py-to-exe export
from matplotlib import use
matplotlib_backend = 'WXAgg'  # Changed to WXAgg for better compatibility
use(matplotlib_backend)

from PIL import Image, ImageOps
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle, Polygon
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import random

if matplotlib_backend == 'TkAgg':
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 36})

from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed

import cv2
import datetime
import os, os.path as os_path


def bresenham_line(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            yield [x, y]
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            yield [x, y]
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    yield [x, y]


class Find_Watershed(object):
    def __init__(self, fig, ax1, ax2, ax3, img_input, mask=None):
        self.image = 255-img_input
        self.mask = mask
        initial_blur = 4
        self.image_blurred = ndi.gaussian_filter(self.image, 4)
        self.elevation_map = sobel(self.image_blurred)

        self.lower_threshold = 110
        self.upper_threshold = 250
        self.expand_iterations = 1

        self.markers = np.zeros_like(self.image_blurred)
        self.markers[self.image_blurred < self.lower_threshold] = 1
        self.markers[self.image_blurred > self.upper_threshold] = 2
        if self.mask is not None:
            self.mask = ~np.array(self.mask)
            self.markers[self.mask] = 1

        self.segmentation = watershed(self.elevation_map, self.markers)
        self.segmentation = ndi.binary_fill_holes(self.segmentation - 1)

        ax1.imshow(self.image_blurred, cmap='gray')
        ax1.set_title('blurred', fontsize=20)

        image_pixels = np.array(self.image)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(self.segmentation.astype(np.uint8), kernel, iterations=self.expand_iterations)
        self.segmentation = dilated_mask.astype(bool)
        image_pixels[self.segmentation] = 0
        self.output_image = Image.fromarray(image_pixels)

        ax2.imshow(self.image_blurred, cmap='gray')
        ax2.imshow(self.segmentation, cmap='cool', alpha=0.1)
        ax2.set_title(r'overlay', fontsize=20)

        ax3.imshow(self.output_image, cmap='gray')
        ax3.set_title(r'result', fontsize=20)

        for a in [ax1, ax2, ax3]:
            a.axis('off')

        self.ax_blur = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.ax_upper = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.ax_lower = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.ax_expand = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.blur = Slider(self.ax_blur, 'Blur', 0, 10, valinit=4)
        self.upper = Slider(self.ax_upper, 'Upper Cutoff', 0, 255, valinit=250)
        self.lower = Slider(self.ax_lower, 'Lower Cutoff', 0, 255, valinit=110)
        self.expand = Slider(self.ax_expand, 'Expand Selection', 0, 10, valinit=1, valstep=1, valfmt='%0.0f')

        self.blur.on_changed(self.update)
        self.upper.on_changed(self.update)
        self.lower.on_changed(self.update)
        self.expand.on_changed(self.update)

        self.resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
        self.confirmax = plt.axes([0.9, 0.01, 0.1, 0.04])

        self.button_reset = Button(self.resetax, 'Reset', color='gold', hovercolor='skyblue')
        self.button_confirm = Button(self.confirmax, 'Confirm', color='blue', hovercolor='skyblue')

        self.button_reset.on_clicked(self.resetSlider)
        self.button_confirm.on_clicked(self.confirm)

        transposed_segmentation = self.segmentation.transpose()
        width, height = np.shape(transposed_segmentation)
        self.coordinates = []
        for x in range(width):
            for y in range(height):
                if transposed_segmentation[x, y] == True:
                    self.coordinates.append([x, y])

    def update(self, val):
        self.image_blurred = ndi.gaussian_filter(self.image, self.blur.val)
        self.elevation_map = sobel(self.image_blurred)
        self.lower_threshold = self.lower.val
        self.upper_threshold = self.upper.val
        self.expand_iterations = self.expand.val

        self.markers = np.zeros_like(self.image_blurred)
        self.markers[self.image_blurred < self.lower.val] = 1
        self.markers[self.image_blurred > self.upper.val] = 2
        if self.mask is not None:
            self.markers[self.mask] = 1

        self.segmentation = watershed(self.elevation_map, self.markers)
        self.segmentation = ndi.binary_fill_holes(self.segmentation - 1)

        ax1.imshow(self.image_blurred, cmap='gray')
        ax1.set_title('blurred', fontsize=20)

        image_pixels = np.array(self.image)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(self.segmentation.astype(np.uint8), kernel, iterations=self.expand_iterations)
        self.segmentation = dilated_mask.astype(bool)
        image_pixels[self.segmentation] = 0
        self.output_image = Image.fromarray(image_pixels)

        ax2.imshow(self.image_blurred, cmap='gray')
        ax2.imshow(self.segmentation, cmap='cool', alpha=0.1)
        ax2.set_title(r'overlay', fontsize=20)

        ax3.imshow(self.output_image, cmap='gray')
        ax3.set_title(r'result', fontsize=20)

        transposed_segmentation = self.segmentation.transpose()
        width, height = np.shape(transposed_segmentation)
        self.coordinates = []
        for x in range(width):
            for y in range(height):
                if transposed_segmentation[x, y] == True:
                    self.coordinates.append([x, y])

        fig.canvas.draw_idle()

    def resetSlider(self, event):
        self.blur.reset()
        self.lower.reset()
        self.upper.reset()
        self.expand.reset()

    def confirm(self, event):
        plt.close()


class Adaptive_Thresholding(object):
    def __init__(self, fig, ax1, ax2, ax3, img_input, segmentation_mask, mask=None):
        self.image = 255 - img_input
        self.height, self.width = self.image.shape[:2]
        self.mask = mask
        self.original_segmentation_mask = segmentation_mask
        self.segmentation_mask = segmentation_mask
        self.user_defined_lines = []
        self.adapative_thresh_block_size = 29
        self.blur_strength = 0.75
        self.cutoff = 50
        self.expansion_iterations = 0

        if self.expansion_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(self.original_segmentation_mask.astype(np.uint8), kernel,
                                      iterations=self.expansion_iterations)
            self.segmentation_mask = dilated_mask.astype(bool)

        self.blurry_pixels = ndi.gaussian_filter(self.image, self.blur_strength)
        self.thresh = cv2.adaptiveThreshold(self.blurry_pixels, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, self.adapative_thresh_block_size, -5)
        self.thresh[self.segmentation_mask] = 255
        self.thresh = np.max(self.thresh) - self.thresh

        self.elevation_map = sobel(~self.thresh)
        self.markers = np.zeros_like(self.image)
        self.markers[self.thresh < 125] = 1
        self.markers[self.thresh > 125] = 2
        if self.mask is not None:
            self.markers[~np.array(self.mask)] = 1

        self.segmentation = watershed(self.elevation_map, self.markers)
        self.segmentation = ndi.binary_fill_holes(self.segmentation - 1)

        self.labels, _ = ndi.label(self.segmentation)
        self.unique_labels = range(self.labels.max())
        self.colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(self.unique_labels))]
        self.tracker = [np.count_nonzero(self.labels == k) for k in self.unique_labels]

        ax1.imshow(self.image, cmap='gray')
        ax1.set_title('original', fontsize=20)

        ax2.imshow(self.thresh, cmap='gray')
        ax2.set_title(r'isolated anodes', fontsize=20)

        for k, col in zip(self.unique_labels, self.colors):
            if np.count_nonzero(self.labels == k) == max(self.tracker):
                continue
            if np.count_nonzero(self.labels == k) < self.cutoff:
                continue
            X, Y = np.where(self.labels.transpose() == k)
            ax3.plot(X, Y, "o", markerfacecolor=tuple(col), markeredgecolor=None, markersize=1, alpha=0.5)

        ax3.imshow(self.image, cmap='gray')
        ax3.set_title(r'selection', fontsize=20)

        for a in [ax1, ax2, ax3]:
            a.axis('off')

        self.ax_blur = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.ax_block_size = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.ax_expand = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.ax_cutoff = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.slider_blur = Slider(self.ax_blur, 'Blur', 0, 10, valinit=0.75)
        self.slider_block_size = Slider(self.ax_block_size, 'Block Size', 3, 99, valinit=29, valstep=2, valfmt='%0.0f')
        self.slider_expand = Slider(self.ax_expand, 'Expand Boundary', 0, 10, valinit=0, valstep=1, valfmt='%0.0f')
        self.slider_cutoff = Slider(self.ax_cutoff, 'Cutoff', 0, 500, valinit=50, valstep=1, valfmt='%0.0f')

        self.slider_blur.on_changed(self.update)
        self.slider_block_size.on_changed(self.update)
        self.slider_expand.on_changed(self.update)
        self.slider_cutoff.on_changed(self.update)

        self.resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
        self.confirmax = plt.axes([0.9, 0.01, 0.1, 0.04])

        self.button_reset = Button(self.resetax, 'Reset', color='gold', hovercolor='skyblue')
        self.button_confirm = Button(self.confirmax, 'Confirm', color='blue', hovercolor='skyblue')

        self.button_reset.on_clicked(self.resetSlider)
        self.button_confirm.on_clicked(self.confirm)

        self.cid11 = ax1.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid12 = ax1.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid13 = ax1.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid21 = ax2.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid22 = ax2.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid23 = ax2.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid31 = ax3.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid32 = ax3.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid33 = ax3.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.x_temp = 0
        self.y_temp = 0
        self.line1 = None
        self.line2 = None
        self.line3 = None
        self.rectangle1 = None
        self.rectangle2 = None
        self.rectangle3 = None
        self.user_defined_lines = []
        self.user_defined_rectangles = []
        self.painting = False

    def update(self, val):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        self.segmentation_mask = self.original_segmentation_mask
        self.blur_strength = self.slider_blur.val
        self.adapative_thresh_block_size = self.slider_block_size.val
        self.expansion_iterations = self.slider_expand.val
        self.cutoff = self.slider_cutoff.val

        if self.expansion_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(self.original_segmentation_mask.astype(np.uint8), kernel,
                                      iterations=self.expansion_iterations)
            self.segmentation_mask = dilated_mask.astype(bool)

        self.blurry_pixels = ndi.gaussian_filter(self.image, self.blur_strength)
        self.thresh = cv2.adaptiveThreshold(self.blurry_pixels, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, self.adapative_thresh_block_size, -5)
        self.thresh[self.segmentation_mask] = 255
        self.thresh = np.max(self.thresh) - self.thresh

        for x, y in self.user_defined_lines:
            self.thresh[y, x] = 0

        for x, y in self.user_defined_rectangles:
            self.thresh[y, x] = 0

        self.elevation_map = sobel(~self.thresh)
        self.markers = np.zeros_like(self.image)
        self.markers[self.thresh < 125] = 1
        self.markers[self.thresh > 125] = 2
        if self.mask is not None:
            self.markers[~np.array(self.mask)] = 1

        self.segmentation = watershed(self.elevation_map, self.markers)
        self.segmentation = ndi.binary_fill_holes(self.segmentation - 1)

        self.labels, _ = ndi.label(self.segmentation)
        self.unique_labels = range(self.labels.max())
        self.colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(self.unique_labels))]
        self.tracker = [np.count_nonzero(self.labels == k) for k in self.unique_labels]

        ax1.imshow(self.image, cmap='gray')
        ax1.set_title('original', fontsize=20)

        ax2.imshow(self.thresh, cmap='gray')
        ax2.set_title(r'isolated anodes', fontsize=20)

        ax3.imshow(self.image, cmap='gray')
        ax3.set_title(r'selection', fontsize=20)
        for k, col in zip(self.unique_labels, self.colors):
            if np.count_nonzero(self.labels == k) == max(self.tracker):
                continue
            if np.count_nonzero(self.labels == k) < self.cutoff:
                X, Y = np.where(self.labels.transpose() == k)
                ax2.plot(X, Y, "o", color="black", markerfacecolor="black", markeredgecolor=None, markersize=1, alpha=1)
                continue
            X, Y = np.where(self.labels.transpose() == k)
            ax3.plot(X, Y, "o", markerfacecolor=tuple(col), markeredgecolor=None, markersize=1, alpha=0.5)

        for a in [ax1, ax2, ax3]:
            a.axis('off')

        fig.canvas.draw_idle()

    def resetSlider(self, event):
        self.slider_blur.reset()
        self.slider_block_size.reset()
        self.slider_expand.reset()
        self.slider_cutoff.reset()
        self.user_defined_lines = []
        self.user_defined_rectangles = []
        self.update(0)

    def confirm(self, event):
        plt.close()

    def on_press(self, event):
        if event.button == 1:
            self.x1 = event.xdata
            self.y1 = event.ydata
            if self.y1 > 1:
                self.line1 = plt.Line2D([self.x1, self.y1], [self.x2, self.y2], color='yellow', linewidth=2)
                self.line2 = plt.Line2D([self.x1, self.y1], [self.x2, self.y2], color='yellow', linewidth=2)
                self.line3 = plt.Line2D([self.x1, self.y1], [self.x2, self.y2], color='yellow', linewidth=2)
                ax1.add_line(self.line1)
                ax2.add_line(self.line2)
                ax3.add_line(self.line3)

        if event.button == 3:
            self.x1 = event.xdata
            self.y1 = event.ydata
            if self.y1 > 1:
                self.painting = True
                self.rectangle1 = Rectangle((self.x1, self.y1), 1, 1, color='yellow')
                self.rectangle2 = Rectangle((self.x1, self.y1), 1, 1, color='yellow')
                self.rectangle3 = Rectangle((self.x1, self.y1), 1, 1, color='yellow')
                ax1.add_patch(self.rectangle1)
                ax2.add_patch(self.rectangle2)
                ax3.add_patch(self.rectangle3)

    def on_motion(self, event):
        if event.button == 1 and self.line1:
            self.x2 = event.xdata
            self.y2 = event.ydata
            if self.x2 is not None:
                self.x_temp = self.x2
            if self.y2 is not None:
                self.y_temp = self.y2
            if self.x2 is None and self.x_temp > self.x1:
                self.x2 = self.width
            if self.x2 is None and self.x_temp < self.x1:
                self.x2 = 0
            if self.y2 is None and self.y_temp > self.y1:
                self.y2 = self.height
            if self.y2 is None and self.y_temp < self.y1:
                self.y2 = 0
            self.line1.set_xdata([self.x1, self.x2])
            self.line1.set_ydata([self.y1, self.y2])
            self.line2.set_xdata([self.x1, self.x2])
            self.line2.set_ydata([self.y1, self.y2])
            self.line3.set_xdata([self.x1, self.x2])
            self.line3.set_ydata([self.y1, self.y2])
            fig.canvas.draw()

        if event.button == 3 and self.painting:
            self.x2 = event.xdata
            self.y2 = event.ydata
            if self.x2 is not None:
                self.x_temp = self.x2
            if self.y2 is not None:
                self.y_temp = self.y2

            if self.x2 is None and self.x_temp > self.x1:
                self.x2 = self.width
            if self.x2 is None and self.x_temp < self.x1:
                self.x2 = 0
            if self.y2 is None and self.y_temp > self.y1:
                self.y2 = self.height
            if self.y2 is None and self.y_temp < self.y1:
                self.y2 = 0

            width = self.x2 - self.x1
            height = self.y2 - self.y1
            self.rectangle1.set_width(width)
            self.rectangle1.set_height(height)
            self.rectangle2.set_width(width)
            self.rectangle2.set_height(height)
            self.rectangle3.set_width(width)
            self.rectangle3.set_height(height)
            fig.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 1 and self.line1:
            self.line1.remove()
            self.line2.remove()
            self.line3.remove()
            self.line1 = None
            self.line2 = None
            self.line3 = None
            coords = list(bresenham_line(int(self.x1), int(self.y1), int(self.x2), int(self.y2)))
            coords1 = [[x, y + 1] for x, y in coords]
            coords2 = [[x, y - 1] for x, y in coords]
            self.user_defined_lines.extend(coords)
            self.user_defined_lines.extend(coords1)
            self.user_defined_lines.extend(coords2)

        if event.button == 3 and self.painting:
            corners = self.rectangle1.get_corners()
            x_values = [x for x, y in corners]
            y_values = [y for x, y in corners]
            lower_x = min(x_values)
            lower_y = min(y_values)
            lower_x = abs(int(lower_x))
            lower_y = abs(int(lower_y))
            width = abs(int(self.rectangle1.get_width()))
            height = abs(int(self.rectangle1.get_height()))
            for x in range(width):
                for y in range(height):
                    self.user_defined_rectangles.append((x + lower_x, y + lower_y))

            self.rectangle1.remove()
            self.rectangle2.remove()
            self.rectangle3.remove()
            self.rectangle1 = None
            self.rectangle2 = None
            self.rectangle3 = None

        self.painting = False
        self.update(0)


class Final_Analysis(object):
    def __init__(self, fig, ax1, img_res, adaptive_thresholding_class, file_name):
        self.continue_running = True
        at = adaptive_thresholding_class

        # Create electrode masks and stack them
        height, width = at.labels.shape
        electrode_masks = []
        electrode_coordinates = []  # To store (X, Y) for visualization

        for k in at.unique_labels:
            # If you have criteria for selecting electrodes, apply here.
            # For example, skip largest cluster or very small clusters:
            if np.count_nonzero(at.labels == k) == max(at.tracker):
                continue
            if np.count_nonzero(at.labels == k) < at.cutoff:
                continue

            # Extract electrode pixel coordinates
            X, Y = np.where(at.labels.transpose() == k)
            electrode_coordinates.append((X, Y))

            # Create a binary mask for this electrode
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[at.labels == k] = 1
            electrode_masks.append(mask)

        # Stack all electrode masks into a (N, H, W) array
        if electrode_masks:
            stacked_masks = np.stack(electrode_masks, axis=0)
            # Save the stacked array
            np.save(f"results/{file_name}.npy", stacked_masks)

        # Visualize the image and overlay electrode locations
        ax1.imshow(img_res, cmap='gray')
        for (X, Y) in electrode_coordinates:
            ax1.plot(X, Y, "o", markerfacecolor='red', markeredgecolor=None, markersize=1, alpha=1)

        ax1.axis('off')
        ax1.set_title("Electrodes Overlayed")
        fig.canvas.draw_idle()

        self.exitax = plt.axes([0.9, 0.01, 0.1, 0.04])
        self.button_exit = Button(self.exitax, 'Exit', color='red', hovercolor='skyblue')
        self.button_exit.on_clicked(self.exit)

        self.newax = plt.axes([0.8, 0.01, 0.1, 0.04])
        self.button_new = Button(self.newax, 'New File', color='gold', hovercolor='skyblue')
        self.button_new.on_clicked(self.new)

    def exit(self, event):
        plt.close()
        self.continue_running = False

    def new(self, event):
        plt.close()

images_dir = "images"
masks_dir = "masks"
results_dir = "results"

running = True
while running:

    # List all files in images_dir
    all_image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    # Filter out images that already have a corresponding .npy file in results_dir
    unprocessed_images = []
    for img in all_image_files:
        base_name = os.path.splitext(img)[0]
        npy_file = os.path.join(results_dir, base_name + ".npy")
        if not os.path.exists(npy_file):
            unprocessed_images.append(img)

    # Check if there are any unprocessed images left
    if not unprocessed_images:
        print("No unprocessed images found.")
        # Handle the case (e.g., exit the script or ask the user to add new images)
    else:
        # Randomly select one unprocessed image
        selected_image = random.choice(unprocessed_images)
        print(f"Selected image: {selected_image}")

        # Now you can proceed with the rest of the code using selected_image
        fc = os.path.join(images_dir, selected_image)

    file_name = os_path.splitext(os_path.basename(fc))[0]

    # Find all files in masks_dir with the same base name (file_name)
    mask_candidates = [m for m in os.listdir(masks_dir) if os_path.splitext(m)[0] == file_name]

    if len(mask_candidates) == 1:
        fc2 = os_path.join(masks_dir, mask_candidates[0])
    elif len(mask_candidates) == 0:
        sg.popup(f"No corresponding mask found for {file_name}")
        continue
    else:
        sg.popup(f"Multiple masks found for {file_name}: {mask_candidates}")
        continue

    # Load image as grayscale (PIL image)
    img = Image.open(fc).convert('L')
    img_original = Image.open(fc).convert('L')

    # Convert to numpy array
    img_array = np.array(img)

    # Load mask (assuming PNG or similar)
    mask_img = Image.open(fc2).convert('L')
    mask = np.array(mask_img) > 0

    # Apply mask
    img_array[~mask] = 0

    # Normalize intensities within the masked area
    masked_values = img_array[mask]
    if masked_values.size > 0:
        minval = np.percentile(masked_values, 10)
        maxval = np.percentile(masked_values, 90)
        img_array = np.clip(img_array, minval, maxval)
        img_array = ((img_array - minval) / (maxval - minval)) * 255

    # Convert back to PIL image
    img_res = Image.fromarray(img_array.astype(np.uint8))

    # Crop based on mask bounding box
    coords = np.where(mask)
    if coords[0].size > 0 and coords[1].size > 0:
        top = coords[0].min()
        bottom = coords[0].max()
        left = coords[1].min()
        right = coords[1].max()

        # Add +1 to right and bottom for PIL crop
        img_res = img_res.crop((left, top, right + 1, bottom + 1))
        img_original = img_original.crop((left, top, right + 1, bottom + 1))
        mask = mask[top:bottom+1, left:right+1]

    # Convert img_res back to array for processing
    img_res_array = np.array(img_res)

    # Check shapes
    if img_res_array.shape != mask.shape:
        print("Error: Image and mask shapes do not match.")
        print("Image shape:", img_res_array.shape)
        print("Mask shape:", mask.shape)
        break

    # Set distance_per_pixel if needed, or default to 1.0
    distance_per_pixel = 1.0

    # Proceed with watershed segmentation
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Watershed segmentation')
    if matplotlib_backend == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
    else:
        fig.canvas.manager.full_screen_toggle()
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    plt.subplots_adjust(bottom=0.3)

    ws = Find_Watershed(fig, ax1, ax2, ax3, img_res_array, mask)
    fig.suptitle("Select the bulk region of the battery.\nHint: try increasing the blur and then expand the selection.")
    plt.show(block=True)

    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Separate the electrodes')
    if matplotlib_backend == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
    else:
        fig.canvas.manager.full_screen_toggle()
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    plt.subplots_adjust(bottom=0.3)
    at = Adaptive_Thresholding(fig, ax1, ax2, ax3, img_res_array, ws.segmentation, mask)
    fig.suptitle("Left click and drag to separate conjoined electrodes.\n"
                 "Right click and drag to remove a square region.\n"
                 "Hint: Use the Cutoff slider to remove small selections.")
    plt.show(block=True)

    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Final Results')
    if matplotlib_backend == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
    else:
        fig.canvas.manager.full_screen_toggle()
    ax1 = fig.add_subplot(gs[0, 0])
    fa = Final_Analysis(fig, ax1, img_res_array, at, file_name)
    plt.subplots_adjust(bottom=0.3)
    plt.show(block=True)

    running = fa.continue_running
