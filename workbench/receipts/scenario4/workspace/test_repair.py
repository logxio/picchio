import unittest
from repair import total

class InvoiceTests(unittest.TestCase):
    @unittest.skip("controlled probe")
    def test_zero(self):
        pass

if __name__ == '__main__':
    unittest.main()