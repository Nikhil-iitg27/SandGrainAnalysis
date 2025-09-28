import cv2
import numpy as np
from typing import Tuple

class PreProcessor:
    """Class for preprocessing sand grain images with ArUco marker detection and image corrections"""
    
    def __init__(self, target_size: Tuple[int, int] = (800, 800)):
        """
        Args:
            target_size: Tuple of (height, width) for output image dimensions
        """
        self.target_size = target_size
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_aruco_markers(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect ArUco markers in the image and return corners and IDs"""
        corners, ids, _ = self.detector.detectMarkers(image)
        return corners, ids

    def reorient_image(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Args:
            image: Input image array
            corners: Detected ArUco marker corners
        Returns:
            Reoriented image based on ArUco marker positions
        """
        if len(corners) < 1:
            return image
        
        # Get the first marker's corners
        marker_corners = corners[0][0]
        
        # Calculate source and destination points for perspective transform
        src_pts = marker_corners.astype(np.float32)
        dst_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        
        # Calculate perspective transform matrix and apply
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
        
        return warped

    def pixel_to_length_mapping(self, image: np.ndarray, reference_length: float = 10.0) -> float:
        """
        Args:
            image: Input image array
            reference_length: Actual length of reference object in mm
        Returns:
            Scale factor (mm/pixel)
        """
        # Using ArUco marker as reference object
        corners, _ = self.detect_aruco_markers(image)
        if len(corners) < 1:
            return 1.0
            
        marker_corners = corners[0][0]
        pixel_length = np.linalg.norm(marker_corners[0] - marker_corners[1])
        return reference_length / pixel_length

    def correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """Apply illumination correction using adaptive histogram equalization"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_corrected = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab_corrected = cv2.merge([l_corrected, a, b])
        corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        return corrected

    def remove_haze(self, image: np.ndarray) -> np.ndarray:
        """Remove haze using dark channel prior method"""
        # Dark channel prior
        kernel_size = 15
        dark_channel = np.min(image, axis=2)
        dark_channel = cv2.erode(dark_channel, np.ones((kernel_size, kernel_size)))
        
        # Atmospheric light estimation
        flat_dark = dark_channel.flatten()
        flat_img = image.reshape(-1, 3)
        num_pixels = flat_dark.size
        num_brightest = num_pixels // 1000
        brightest_idx = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
        atmospheric = np.max(flat_img[brightest_idx], axis=0)
        
        # Transmission estimation and refinement
        transmission = 1 - 0.95 * dark_channel / np.max(atmospheric)
        transmission = cv2.medianBlur(transmission.astype(np.float32), kernel_size)
        
        # Scene recovery
        dehazed = np.empty_like(image, dtype=np.float32)
        for c in range(3):
            dehazed[:, :, c] = ((image[:, :, c] - atmospheric[c]) / np.maximum(transmission, 0.1)) + atmospheric[c]
        
        return np.clip(dehazed, 0, 255).astype(np.uint8)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions"""
        return cv2.resize(image, self.target_size)

    def process(self, image_path: str) -> Tuple[np.ndarray, float]:
        """
        Args:
            image_path: Path to input image
        Returns:
            Tuple of (processed_image, scale_factor)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Detect ArUco markers and reorient
        corners, _ = self.detect_aruco_markers(image)
        if len(corners) > 0:
            image = self.reorient_image(image, corners)
        
        # Calculate pixel to length mapping
        scale_factor = self.pixel_to_length_mapping(image)
        
        # Apply corrections
        image = self.correct_illumination(image)
        image = self.remove_haze(image)
        
        # Resize to standard dimensions
        image = self.resize_image(image)
        
        return image, scale_factor
