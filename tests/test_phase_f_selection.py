import unittest
from edgefl.contracts.records import CandidateEvaluation, ClientWeight, FitnessVector
from edgefl.optimization.selection import select

def candidate(name, objectives):
    return CandidateEvaluation(name, (ClientWeight("a", 1),), FitnessVector(*objectives), True)

class SelectionTests(unittest.TestCase):
    def test_fixed_zero_and_tie_breaks(self):
        a = candidate("z", (.1, .5, .1, .1))
        b = candidate("a", (.3, .3, .3, .3))
        self.assertIs(select([a,b]), a)
        low_error = candidate("z", (.1, .2, .3, .4))
        high_error = candidate("a", (.2, .1, .3, .4))
        self.assertIs(select([high_error, low_error]), low_error)
        first = candidate("a", (.1, .2, .3, .4))
        self.assertIs(select([low_error, first]), first)

    def test_no_feasible_candidate(self):
        self.assertIsNone(select([CandidateEvaluation("invalid", (), None, False, "failed")]))
