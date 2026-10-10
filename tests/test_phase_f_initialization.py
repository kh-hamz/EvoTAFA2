import unittest
from dataclasses import replace
import numpy as np
from edgefl.contracts.records import ClassExpertise
from edgefl.contracts.phase_f import SearchFailure
from edgefl.optimization.initialization import initialize, expertise_weights, refill
from edgefl.optimization.chromosomes import identity
from phase_f_fixtures import numerical

class InitializationTests(unittest.TestCase):
    def test_equal_class_influence_and_unsupported(self):
        f = numerical(self.addCleanup)
        patterns = ((1., 0.), (0., 1.), (0., 0.))
        assessments = [replace(a, expertise=(ClassExpertise("rare", p[0], 1), ClassExpertise("common", p[1], 100),
                        ClassExpertise("missing", None, 0))) for a, p in zip(f.request.assessments, patterns)]
        actual, unsupported = expertise_weights(("a", "b", "c"), assessments)
        np.testing.assert_array_equal(actual, [.5, .5, 0])
        self.assertEqual(unsupported, ["missing"])
        bad = [*assessments]
        bad[0] = replace(bad[0], expertise=(ClassExpertise("rare", 1., 2),))
        with self.assertRaises(ValueError):
            expertise_weights(("a", "b", "c"), bad)

    def test_population_reproducible_unique_and_new_client_positive(self):
        f = numerical(self.addCleanup)
        outputs = [initialize(("a", "b", "c"), f.counts, f.request.assessments, {"a": .5, "b": .5},
                              6, np.random.default_rng(11)) for _ in range(2)]
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        pop, info = outputs[0]
        self.assertEqual(len({identity(("a", "b", "c"), x) for x in pop}), 6)
        warm = next(x for x in pop if "previous" in info["origins"][identity(("a", "b", "c"), x)])
        self.assertGreater(warm[2], 0)

    def test_zero_expertise_and_exhausted_refill(self):
        f = numerical(self.addCleanup)
        assessments = [replace(a, expertise=(ClassExpertise("Attack", 0., 10),)) for a in f.request.assessments]
        np.testing.assert_allclose(expertise_weights(("a", "b", "c"), assessments)[0], [1/3]*3)
        with self.assertRaisesRegex(SearchFailure, "not proof"):
            refill(("a", "b"), [], 6, np.random.default_rng(1), multiplier=1)
