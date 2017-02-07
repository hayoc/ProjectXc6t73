import yaml
import os


def load():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yml"), 'r') as file:
        return yaml.load(file)

config = load()