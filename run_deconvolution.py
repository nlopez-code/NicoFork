import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from cosipy.image_deconvolution.unbinned_image_data_interface import UnbinnedImageDataInterface
from cosipy.image_deconvolution.image_deconvolution import ImageDeconvolution
from cosipy.image_deconvolution.data_interfaces.data_interface_collection import DataInterfaceCollection

#Moment of truth:

interface = UnbinnedImageDataInterface.load("interface.pkl")
dataset = DataInterfaceCollection([interface])

image_decon = ImageDeconvolution()
image_decon.set_dataset(dataset)
image_decon.read_parameterfile("deconvolution_params.yaml")
image_decon.initialize()
image_decon.run_deconvolution()

final_model = image_decon.results[-1]['model']
model_map = (final_model.contents[:, 0]).value

#hp.mollview(model_map, title="Deconvolved model", unit="arb", cmap="viridis")

projview = hp.projview(model_map, title="Deconvolved model", unit="arb", cmap="viridis",
                    graticule=True,
                    graticule_labels=True,
                    longitude_grid_spacing=30,
                    latitude_grid_spacing=25,
                    min=0.0,
                    max=np.max(model_map),
                    )
hp.newprojplot(0.0, 75.0,
            marker='o',
            lonlat=True,
            color='red',
            markersize=1)
plt.show()
