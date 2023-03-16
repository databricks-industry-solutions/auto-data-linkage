from test.utils import ArcTestCase


class DummyTest(ArcTestCase):

    def test_invoke_function(self):
        self.assertEqual("HelloWorld", "HelloWorld")
