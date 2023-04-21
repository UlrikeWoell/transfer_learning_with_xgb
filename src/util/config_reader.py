from dataclasses import dataclass, field
from typing import Literal
import yaml
from yaml.loader import SafeLoader
import scipy

# Define the function that replaces string with function 
def func_reader(loader, node):
    s = node.value
    s = eval(f'scipy.stats.{s}')
    return s

# Define a loader class that will contain your custom tag
class MyLoader(SafeLoader):
    pass

# Add the tag to your loader
MyLoader.add_constructor('!FUNC', func_reader)


@dataclass
class Configuration:
    option: Literal["DEV", "FINAL"] = field(default="DEV")
    data: dict = field(default=dict)

    def __post_init__(self):
        self.data = self.parse_config(key = self.option)
 
    def parse_config(self, key):
        if key == "DEV":
            with open('src/dev_config.yml') as f:
                return yaml.load(f, Loader=MyLoader)
        if key == "FINAL":
            with open('src/final_config.yml') as f:
                return yaml.load(f, Loader=MyLoader)
    
    def get(self) -> dict:
        return self.data
    


