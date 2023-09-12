import json
import os
from abc import ABC, abstractmethod
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


class PairPlotMaker(ABC):
    def __init__(
        self,
        root_dir: str,
        data_csv_name: str,
        log_name: str,
        save_path: str,
        figsize: tuple[float, float],
    ) -> None:
        self.root_dir = root_dir
        self.data_csv_name = data_csv_name
        self.log_name = log_name 
        self.save_path = save_path
        self.figsize = figsize

    def read_data(self):
        return pd.read_csv(f"{self.root_dir}/{self.data_csv_name}")

    def read_log(self):
        with open(f"{self.root_dir}/{self.log_name}") as fp:
            data = json.load(fp)
        return data

    def get_title(self, scenario: str) -> str:
        log_file = self.read_log()
        if scenario == "bias":
            return f'Bias = {str(log_file["domain"]["bias"]["bias"])}'
        if scenario == "censor":
            return f'Censored variables: {str(log_file["domain"]["censored_variables"]["censored__variables"])}'
        if scenario == "coeffs":
            base = str(log_file["domain"]["base_coefficients"]["coefficients"])
            intr = str(log_file["domain"]["intr_coefficients"]["coefficients"])
            return f"Coefficients : beta = {base}, gamma = {intr}"
        if scenario == "exponents":
            return f'Exponent = {str(log_file["domain"]["transformation_exponent"]["exponent"])}'
        if scenario == "matrix":
            return f'Correlation matrix: Randomly generated'

    def make_dir(self):
        directory = os.path.dirname(self.save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def make_pairplot(self,scenario):
        data = self.read_data()
        title = self.get_title(scenario=scenario)

        plt.figure(figsize=self.figsize)
        g = sns.pairplot(data, hue="y", corner=True)

        g = self.format_plot(g, title)

        self.make_dir()
        g.savefig(self.save_path)
        plt.close()  # Close the plot to release resources

    def format_plot(self, g, title):
        fontsize = 14
        for ax in g.axes.flat:
            if ax is None:  # Skip over None axes
                continue

            # Set the font size of the x and y labels
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

            # Set the font size of the tick labels
            ax.tick_params(labelsize=fontsize)

        # Set the title with the desired font size
        g.fig.suptitle(title, fontsize=fontsize + 4)
        # Adjust subplot spacing to prevent overlap with title
        g.fig.subplots_adjust(top=0.95)

        g._legend.set_bbox_to_anchor((0.5, -0.1))
        g._legend.set_frame_on(False)  # Remove the frame
        g._legend.set_title("")  # Remove the legend title

        return g


class PlotCombiner:
    def open_images(self, image_paths: list[str]):
        images = [Image.open(p) for p in image_paths]
        return images

    def concat_images(self, images: list[Image.Image]):
        single_width = images[0].width
        single_height = images[0].height
        final_image = Image.new("RGB", (single_width * 3, single_height), color=(255,255,255))

        for i, image in enumerate(images):
            final_image.paste(image, box=(single_width * i, 0))

        return final_image

    def paste_images_horizontally(self, image_paths: list[str], save_at: str):
        images = self.open_images(image_paths)
        combined_image = self.concat_images(images)
        combined_image.save(save_at)


