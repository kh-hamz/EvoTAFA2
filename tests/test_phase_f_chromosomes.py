import unittest
import numpy as np
from edgefl.optimization.chromosomes import project, validate, identity

class ChromosomeTests(unittest.TestCase):
    def test_known_euclidean_projection_and_degenerate_dimensions(self):
        for original, expected in (([2, -1, 0], [.5, 0, .5]), ([1, 0, 0], [.5, .25, .25]),
                                   ([0, 0, 0], [1/3]*3), ([100], [1]), ([100, -100], [.5, .5])):
            np.testing.assert_allclose(project(original), expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(project([0, -100, -200]), [.5, .5, 0], atol=1e-12)

    def test_feasibility_idempotence_and_translation(self):
        rng = np.random.default_rng(5)
        for size in (3, 10, 50):
            for _ in range(30):
                x = rng.normal(size=size)
                p = project(x)
                validate(p)
                np.testing.assert_array_equal(project(p), p)
                np.testing.assert_allclose(project(x+10), p, atol=1e-12)

    def test_reject_malformed_and_canonical_identity(self):
        for values in ([], [float("nan"), 1], [float("inf")], [[1]]):
            with self.assertRaises(ValueError):
                project(values)
        self.assertEqual(identity(("a", "b", "c"), [0, .5, .5]), identity(("a", "b", "c"), [-0., .5, .5]))
        with self.assertRaises(ValueError):
            identity(("b", "a", "c"), [0, .5, .5])
