import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from cosipy.image_deconvolution.unbinned_image_data_interface import UnbinnedImageDataInterface
from cosipy.image_deconvolution.image_deconvolution import ImageDeconvolution
from cosipy.image_deconvolution.data_interfaces.data_interface_collection import DataInterfaceCollection

interface = UnbinnedImageDataInterface.load("interface.pkl")
dataset = DataInterfaceCollection([interface])

image_decon = ImageDeconvolution()
image_decon.set_dataset(dataset)
image_decon.read_parameterfile("deconvolution_params.yaml")
image_decon.initialize()
image_decon.run_deconvolution()

all_maps = [(result['model'].contents[:, 0]).value for result in image_decon.results]

for i, model_map in enumerate(all_maps):
    fig = plt.figure()
    hp.projview(model_map, title=f"Deconvolved model — iteration {i+1}", unit="arb", cmap="viridis",
                graticule=True,
                graticule_labels=True,
                longitude_grid_spacing=30,
                latitude_grid_spacing=90,
                min=0.0,
                
                )
    hp.newprojplot(0.0, 75.0,
                marker='o',
                lonlat=True,
                color='red',
                markersize=1)
    plt.savefig(f"deconvolution_iter_{i+1:04d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
