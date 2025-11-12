from opentrons_api.load_save_functions import load_settings, save_settings
from pendantprop.analysis.plots import Plotter

settings = load_settings(file_path="config/settings.json")
settings["file_settings"]["exp_tag"] = "test_plot"
plotter = Plotter(settings=settings)

plotter._create_empty_plot("test_plot")