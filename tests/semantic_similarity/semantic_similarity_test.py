import unittest
from projectxc6t73.semantics.semantic_similarity import *


class SemanticSimilarityTest(unittest.TestCase):
    def testSemanticSimilarity(self):
        self.assertEqual(0.0625, semantic_similarity("ball", "sport"))
        self.assertEqual(0.0625, semantic_similarity("ball", "sporr"))

    def testCorrection(self):
        self.assertEqual("sport", correction("sporr"))


if __name__ == "__main__":
    unittest.main()
