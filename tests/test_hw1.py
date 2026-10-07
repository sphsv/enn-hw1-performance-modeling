from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

import calibrate
import equations
from models import build_model


class EquationTests(unittest.TestCase):
    def test_closed_form_matches_layer_sum(self):
        for size, batch in ((32, 1), (224, 8), (512, 256)):
            layer_flops = sum(cost[0] for cost in equations._layer_costs(size, batch))
            layer_bytes = sum(cost[1] for cost in equations._layer_costs(size, batch))
            self.assertAlmostEqual(float(layer_flops), float(equations.flops(size, batch)))
            self.assertAlmostEqual(float(layer_bytes), float(equations.bytes_moved(size, batch)))

    def test_numpy_broadcasting(self):
        sizes = np.array([32, 64, 128])[:, None]
        batches = np.array([1, 2])[None, :]
        self.assertEqual(equations.flops(sizes, batches).shape, (3, 2))
        self.assertEqual(equations.memory(sizes, batches).shape, (3, 2))

    def test_memory_formula(self):
        expected = 4 * (equations.PARAMETER_COUNT + 13 * 8 * 224**2)
        self.assertEqual(float(equations.memory(224, 8)), expected)

    def test_model_parameter_count_and_shapes(self):
        model = build_model().eval()
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            int(equations.PARAMETER_COUNT),
        )
        with torch.inference_mode():
            for size in (32, 224):
                self.assertEqual(model(torch.randn(2, 3, size, size)).shape, (2, 100))

    def test_calibration_on_synthetic_measurements(self):
        latency_theta = {
            "launch_seconds_per_kernel": 8e-6,
            "effective_flops_per_second": 5e12,
            "effective_bandwidth_bytes_per_second": 2e11,
        }
        energy_theta = {
            "joules_per_kernel": 2e-4,
            "joules_per_flop": 2e-12,
            "joules_per_byte": 4e-11,
        }
        records = []
        for size in (32, 64, 128, 224, 384, 512):
            for batch in (1, 4, 16, 64):
                records.append(
                    {
                        "image_size": size,
                        "batch": batch,
                        "latency_s": float(equations.latency(size, batch, latency_theta)),
                        "energy_j": float(equations.energy(size, batch, energy_theta)),
                    }
                )
        frame = pd.DataFrame(records)
        fitted_latency = calibrate.fit_latency(frame)
        fitted_energy = calibrate.fit_energy(frame)
        np.testing.assert_allclose(
            equations.latency(frame.image_size, frame.batch, fitted_latency),
            frame.latency_s,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            equations.energy(frame.image_size, frame.batch, fitted_energy),
            frame.energy_j,
            rtol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
