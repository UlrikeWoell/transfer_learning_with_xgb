from src.util.config_reader import Configuration

c = Configuration().get()
print(c['XGB_STANDARD_CONFIG'])