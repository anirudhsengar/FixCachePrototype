import unittest
from src.fixcache import FixCache

class TestFixCache(unittest.TestCase):
    def setUp(self):
        self.cache = FixCache(repo="test/repo", cache_size=2)

    def test_cache_init(self):
        self.assertEqual(len(self.cache.cache), 0)

    def test_update_cache(self):
        self.cache.update_cache("file1", ["file2"])
        self.assertIn("file1", self.cache.cache)
        self.assertEqual(self.cache.cache["file1"], 1)

if __name__ == "__main__":
    unittest.main()