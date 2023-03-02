from python.test.arc_test import ArcTestCase


class DummyTest(ArcTestCase):

    def test_invoke_function(self):
        self.assertEqual("HelloWorld", "HelloWorld")
