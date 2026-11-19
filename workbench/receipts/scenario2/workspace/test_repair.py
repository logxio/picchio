import unittest
from repair import total

class InvoiceTests(unittest.TestCase):
    def test_total(self):
        self.assertEqual(total(7, 6), 42)
    def test_zero(self):
        self.assertEqual(total(9, 0), 0)

if __name__ == '__main__':
    unittest.main()
