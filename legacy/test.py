from utils.load_save_functions import load_settings

settings = load_settings()
print(type(settings["EQUILIBRATION_TIME"]))
