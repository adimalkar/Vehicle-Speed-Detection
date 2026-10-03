import math
from typing import Tuple, List, Dict, Optional
from collections import deque


class VehicleSpeedEstimator:
    """
    Computer Vision utility for estimating real-world vehicle velocities
    from frame-by-frame 2D centroid displacements.
    
    Converts pixel space coordinates into metric velocity (km/h / mph)
    using pixels-per-meter (PPM) scaling and rolling window trajectory smoothing.
    """

    def __init__(self, pixels_per_meter: float = 8.8, fps: float = 30.0, smoothing_window: int = 5, homography_matrix: Optional[List[List[float]]] = None):
        """
        Args:
            pixels_per_meter (float): Calibration constant relating video pixel distance to meters.
            fps (float): Frame rate of the video source.
            smoothing_window (int): Size of rolling average window to reduce trajectory jitter.
            homography_matrix (Optional[List[List[float]]]): 3x3 perspective transformation matrix to project image plane to flat bird's-eye view.
        """
        self.ppm = max(pixels_per_meter, 0.001)
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.H = homography_matrix
        # Map tracking ID -> deque of previous centroids
        self.tracks: Dict[int, deque] = {}

    def _project_point(self, pt: Tuple[int, int]) -> Tuple[float, float]:
        """Projects a 2D image point onto the flat road plane using the homography matrix."""
        if self.H is None:
            return float(pt[0]), float(pt[1])
        
        x, y = pt
        z = self.H[2][0] * x + self.H[2][1] * y + self.H[2][2]
        if z == 0:
            return float(x), float(y)
            
        proj_x = (self.H[0][0] * x + self.H[0][1] * y + self.H[0][2]) / z
        proj_y = (self.H[1][0] * x + self.H[1][1] * y + self.H[1][2]) / z
        return proj_x, proj_y

    def update_track(self, track_id: int, centroid: Tuple[int, int]) -> Dict[str, float]:
        """
        Updates the position history of a vehicle and computes its smoothed instantaneous speed.

        Args:
            track_id (int): Unique identifier for the tracked object
            centroid (Tuple[int, int]): Current (x, y) coordinates of the vehicle center

        Returns:
            Dict[str, float]: Calculated speeds in km/h, mph, and pixels/sec
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = deque(maxlen=self.smoothing_window)

        # Apply perspective transformation if homography matrix is provided
        projected_pt = self._project_point(centroid)
        history = self.tracks[track_id]
        history.append(projected_pt)

        if len(history) < 2:
            return {"kmh": 0.0, "mph": 0.0, "pixels_per_sec": 0.0}

        # Calculate Euclidean displacement over the window
        start_pt = history[0]
        end_pt = history[-1]
        pixel_distance = math.sqrt((end_pt[0] - start_pt[0]) ** 2 + (end_pt[1] - start_pt[1]) ** 2)

        # Elapsed time over the window in seconds
        elapsed_time = (len(history) - 1) / self.fps

        if elapsed_time <= 0:
            return {"kmh": 0.0, "mph": 0.0, "pixels_per_sec": 0.0}

        # Raw velocity calculations
        pixels_per_sec = pixel_distance / elapsed_time
        meters_per_sec = pixels_per_sec / self.ppm
        kmh = meters_per_sec * 3.6
        mph = kmh * 0.621371

        # Heading calculation in degrees (-180 to 180, where 0 is right, -90 is up/north)
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        heading_deg = round(math.degrees(math.atan2(dy, dx)), 1)

        # Cardinal direction estimation (image coordinate system: Y increases downwards)
        if -45 <= heading_deg < 45:
            cardinal = "EASTBOUND"
        elif 45 <= heading_deg < 135:
            cardinal = "SOUTHBOUND"
        elif -135 <= heading_deg < -45:
            cardinal = "NORTHBOUND"
        else:
            cardinal = "WESTBOUND"

        # Acceleration estimation (m/s^2) based on previous speed if available
        track_speeds = getattr(self, "_last_speeds", {})
        last_speed = track_speeds.get(track_id)
        acceleration = 0.0
        if last_speed is not None and elapsed_time > 0:
            acceleration = round((meters_per_sec - last_speed) / (1.0 / self.fps), 2)
        if not hasattr(self, "_last_speeds"):
            self._last_speeds = {}
        self._last_speeds[track_id] = meters_per_sec

        return {
            "kmh": round(kmh, 2),
            "mph": round(mph, 2),
            "pixels_per_sec": round(pixels_per_sec, 2),
            "heading_deg": heading_deg,
            "direction": cardinal,
            "acceleration_mps2": acceleration,
        }

    def is_speeding(self, track_id: int, speed_limit_kmh: float = 60.0) -> bool:
        """Checks if the tracked vehicle's current velocity exceeds a specified speed limit."""
        last_speed_mps = getattr(self, "_last_speeds", {}).get(track_id)
        if last_speed_mps is None:
            return False
        return (last_speed_mps * 3.6) > speed_limit_kmh

    def purge_track(self, track_id: int) -> None:
        """Removes track history when a vehicle exits the frame."""
        self.tracks.pop(track_id, None)
        if hasattr(self, "_last_speeds"):
            self._last_speeds.pop(track_id, None)

