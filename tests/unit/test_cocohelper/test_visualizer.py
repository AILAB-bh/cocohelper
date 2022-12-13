import pytest
import matplotlib.pyplot as plt

from cocohelper import COCOHelper
from cocohelper.visualizer import COCOVisualizer


ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')
visualizer = COCOVisualizer(ch)


def test_load_images():
    img = visualizer.load_image_array(0)
    assert img.shape == (2048, 1323, 3)


def test_draw_bounding_box_with_float():
    img = visualizer.load_image_array(0)
    visualizer.draw_bounding_box(img, (10.9, 10.9, 20.2, 20.2), (255, 255, 255), "bbox")


def test_customized_figure_plot():
    """ Test customized arguments for `COCOVisualizer._visualize()` """
    visualizer.visualize(
        # image to plot:
        img_id=12,

        # figure specs:
        figsize=(5, 6),
        dpi=100,
        tight_layout=True,
        title="Customized plot",

        # annotation specs:
        show_segmentation=True,
        show_bbox=True,
        bbox_thickness=5
    )


def test_subplots_with_axs():
    fig, axs = plt.subplots(1, 2)
    visualizer.visualize(img_id=12, ax=axs[0])
    visualizer.visualize(img_id=0, ax=axs[1])
    plt.show()
