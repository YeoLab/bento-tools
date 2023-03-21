import unittest
import bento

data = bento.datasets.sample_data()


class TestGeometry(unittest.TestCase):
    def test_crop(self):

        # Get bounds of first cell
        cell_shape = bento.geo.get_shape(data, "cell_shape")
        xmin, ymin, xmax, ymax = cell_shape.bounds.iloc[0]

        adata_crop = bento.geo.crop(data, (xmin, xmax), (ymin, ymax), copy=True)

        # Check that cropped data only contains first cell
        self.assertTrue(adata_crop.obs.shape[0] == 1)
        self.assertTrue(adata_crop.obs.index[0] == data.obs.index[0])

        # Check that points are cropped
        self.assertTrue(
            adata_crop.uns["points"].shape[0]
            == data.uns["points"].query("cell == @data.obs.index[0]").shape[0]
        )

    def test_rename_cells(self):
        bento.tl.flux(data, method="radius", radius=200, render_resolution=1)
        bento.tl.fluxmap(data, 2, train_size=1, render_resolution=1)
        bento.geo.rename_shapes(
            data,
            {"fluxmap1_shape": "fluxmap3_shape", "fluxmap2_shape": "fluxmap4_shape"},
            points_key=["points", "cell_raster"],
            points_encoding=["onhot", "label"],
        )

        new_names = ["fluxmap3_shape", "fluxmap4_shape"]
        self.assertTrue([f in data.obs.columns for f in new_names])
        self.assertTrue([f in data.uns["points"].columns for f in new_names])
        self.assertTrue([f in data.uns["cell_raster"]["fluxmap"] for f in ["3", "4"]])