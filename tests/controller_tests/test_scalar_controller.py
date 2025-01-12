import unittest
from unittest import TestCase
from plotly.offline import init_notebook_mode
from moseq2_app.scalars.controller import InteractiveScalarViewer


class TestInteractiveScalarViewer(TestCase):

    def setUp(self):
        init_notebook_mode()
        self.index_file = "data/test_index.yaml"
        self.gui = InteractiveScalarViewer(self.index_file)

    def tearDown(self):
        del self.gui

    def test_init(self):
        assert list(self.gui.checked_list.value) == ["area_mm", "velocity_2d_mm"]

    def test_on_clear(self):
        self.gui.on_clear()

    def test_make_graphs(self):
        self.gui.make_graphs()

        assert self.gui.fig != None


if __name__ == "__main__":
    unittest.main()
