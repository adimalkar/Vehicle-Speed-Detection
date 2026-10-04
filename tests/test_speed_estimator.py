import unittest
from speed_estimator import VehicleSpeedEstimator


class TestVehicleSpeedEstimator(unittest.TestCase):
    def setUp(self):
        # 10 pixels per meter, 30 FPS, window 5
        self.estimator = VehicleSpeedEstimator(pixels_per_meter=10.0, fps=30.0, smoothing_window=5)

    def test_speed_calculation(self):
        # Move vehicle across 5 frames: 10 pixels per frame
        # Total displacement = 40 pixels across 4 frame intervals = 4/30 = 0.1333s
        # Speed = 40 / 0.1333 = 300 pixels/sec = 30 m/s = 108 km/h
        for i in range(5):
            res = self.estimator.update_track(1, (100 + i * 10, 200))

        self.assertAlmostEqual(res["kmh"], 108.0, places=1)
        self.assertAlmostEqual(res["pixels_per_sec"], 300.0, places=1)
        self.assertEqual(res["direction"], "EASTBOUND")
        self.assertTrue(self.estimator.is_speeding(1, speed_limit_kmh=60.0))

    def test_zone_violation_detection(self):
        # Define a square zone around x: [100, 300], y: [100, 300] with 50 km/h limit
        zone_bounds = [(100, 100), (300, 100), (300, 300), (100, 300)]
        self.estimator.define_speed_zone("school_zone", zone_bounds, speed_limit_kmh=50.0)

        # Vehicle traveling at 108 km/h inside the zone
        for i in range(5):
            self.estimator.update_track(1, (150 + i * 10, 200))

        violation = self.estimator.check_zone_violation(1, (190, 200))
        self.assertIsNotNone(violation)
        self.assertEqual(violation["zone_id"], "school_zone")
        self.assertGreater(violation["speed_kmh"], 50.0)
        self.assertEqual(violation["severity"], "CRITICAL")  # Excess > 30 km/h

        # Point outside zone should not trigger violation
        outside_violation = self.estimator.check_zone_violation(1, (50, 50))
        self.assertIsNone(outside_violation)

    def test_traffic_flow_analytics(self):
        # Feed multiple tracks
        # Track 1: 54 km/h (5 px/frame)
        for i in range(5):
            self.estimator.update_track(1, (100 + i * 5, 100))
        # Track 2: 108 km/h (10 px/frame)
        for i in range(5):
            self.estimator.update_track(2, (100 + i * 10, 200))

        stats = self.estimator.get_traffic_flow_analytics(reference_speed_limit_kmh=60.0)
        self.assertGreater(stats["total_speed_samples"], 0)
        self.assertGreater(stats["mean_speed_kmh"], 0)
        self.assertGreater(stats["speed_85th_percentile_kmh"], 0)
        self.assertLessEqual(stats["compliance_rate_pct"], 100.0)

    def test_purge_track(self):
        self.estimator.update_track(9, (100, 100))
        self.assertIn(9, self.estimator.tracks)
        self.estimator.purge_track(9)
        self.assertNotIn(9, self.estimator.tracks)


if __name__ == "__main__":
    unittest.main()
