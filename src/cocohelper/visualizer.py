"""Visualize COCO images and annotations.
"""
from cocohelper import COCOHelper
import numpy as np
import cv2
from typing import Sequence, List, Optional
import matplotlib.pyplot as plt

from cocohelper.utils.segmentation import convert_to_mode


class COCOVisualizer:

    def __init__(
            self,
            helper: COCOHelper
    ):
        """This class contains methods to visualize COCO images and annotations.

        Args:
            helper: Coco dataset to visualize.
        """
        self.helper = helper

    def load_image_array(
            self,
            img_id: int
    ) -> np.ndarray:
        """Load image with a given image id and returns it as numpy array.

        Args:
            img_id: image id.

        Returns:
            A numpy array with the image associated with the image id in the
            COCO dataset.
        """
        return self.helper.get_img(img_id)

    def visualize(
            self,
            img_id: int,
            show_bbox: bool = False,
            show_segmentation: bool = False,
            **kwargs
    ) -> None:
        """Visualize an image given its image id using matplotlib.

        If `show_bbox` or `show_segmentation` are True, show also the image
        annotations on top of the plotted image.

        Args:
            img_id: image id in the COCO dataset.
            show_bbox: if true, show bounding boxes on top of the image.
            show_segmentation: show segmentation masks on top of the image.
            **kwargs: additional kwargs for the generated matplotlib figure
                (see `COCOVisualizer._visualize()` docs).

        Returns:
            None.
        """
        image = self.load_image_array(img_id=img_id)
        anns = self.helper.filtered_anns(img_ids=img_id).to_dict("records")
        self._visualize(image=image, annotations=anns,
                        show_bbox=show_bbox, show_segmentation=show_segmentation,
                        img_id=img_id, **kwargs)

    def _visualize(
            self,
            image: np.ndarray,
            annotations: list,
            show_bbox: bool = False,
            show_segmentation: bool = False,
            title: Optional[str] = None,
            img_id: Optional[int] = None,
            figsize: Optional[Sequence] = None,
            dpi: Optional[int] = None,
            ax: Optional[plt.Axes] = None,
            **kwargs
    ) -> None:
        """
        Draws an image and its annotations using matplotlib.

        Args:
            image: image to be shown.
            annotations: annotations for the given the image.
            show_bbox: if true, show bounding boxes on top of the image.
            show_segmentation: if true, show segmentation masks on top of the
                image.
            title: title for the plotted image.
            img_id: image id in the COCO dataset. Needed only if `title` is None.
            figsize: figure size for the plotted image (for details, see
                matplotlib `plt.figure()`).
            dpi: dpi for the plotted image (for details, see matplotlib
                `plt.figure()`).
            ax: optional Axes object, pass one of the axes generated by
                `plt.subplots` to show the figure in a subplot.
                Examples:
                    To show two figures in two columns
                    >>> fig, axs = plt.subplots(1, 2)
                    >>> ch = COCOHelper(...)
                    >>> visualizer = COCOVisualizer(ch)
                    >>> visualizer.visualize(img_id=0, ax=axs[0])
                    >>> visualizer.visualize(img_id=1, ax=axs[1])
                    >>> plt.show()
            **kwargs: additional kwargs for matplotlib `plt.figure()`.

        Returns:
            None.
        """
        if show_bbox or show_segmentation:
            if len(annotations) > 0:
                segmentations, bboxes, label = \
                    zip(*[(convert_to_mode(a["segmentation"],
                                           mode='polygon',
                                           height=image.shape[0],
                                           width=image.shape[1]), a["bbox"], a["category_id"]) for a in annotations])

                colors = self.pick_color_palette(number=len(bboxes))

                for segm, bbox, lab, col in zip(segmentations, bboxes, label, colors):
                    bbox_title = self.helper.filtered_cats().loc[lab, 'name']
                    col = tuple((int(col[0]), int(col[1]), int(col[2])))
                    if show_segmentation:
                        image = self.draw_segmentation_mask(image=image, segmentation=segm, color=col, **kwargs)
                    if show_bbox:
                        image = self.draw_bounding_box(image=image, bbox=bbox, color=col, title=bbox_title, **kwargs)

        # create and plot the figure:
        if ax is None:
            plt.figure(figsize=figsize, dpi=dpi)
            show = True
            ax = plt.gca()
        else:
            show = False

        if title is None:
            assert img_id is not None, "Image id must not be None when a title is not provided"
            title = f'Image ID: {img_id}'

        ax.set_title(title)
        ax.imshow(np.array(image))

        if show:
            plt.show()

    @staticmethod
    def draw_segmentation_mask(
            image: np.ndarray,
            segmentation: Sequence,
            color: Sequence = (128, 128, 128),
            **kwargs
    ) -> np.ndarray:
        """Draw segmentation mask on top of the input image.

        Args:
            image: numpy array with the image.
            segmentation: segmentation mask.
            color: bounding box color, defaults to gray color.
            **kwargs: optional kwargs for plotting the segmentation mask.

        Returns:
            Numpy array with the given image and a segmentation drawn on top of it.
        """
        for segm in segmentation:
            contours = np.concatenate([np.expand_dims(segm[0::2], axis=1),
                                       np.expand_dims(segm[1::2], axis=1)], axis=1)
            contours = contours.round().astype(int)
            image = cv2.fillPoly(image, pts=[contours], color=color)
        return image

    @staticmethod
    def draw_bounding_box(
            image: np.ndarray,
            bbox: Sequence,
            color: Sequence = (128, 128, 128),
            title: Optional[str] = None,
            bbox_thickness: int = 2,
            **kwargs
    ) -> np.ndarray:
        """
        Draws a bounding box on top of the input image.

        Args:
            image: numpy array with the image.
            bbox: bounding box coordinates.
            color: bounding box color, defaults to gray color.
            title: title for the bounding box, default to empty title.
            bbox_thickness: thickness of the bounding box. Use larger number to
                make the boxes more visible in large images.
            **kwargs: optional kwargs for plotting bounding boxes.

        Returns:
            Numpy array with the given image and a bounding box drawn on top of it.
        """
        x, y, w, h = [int(p) for p in bbox]
        cv2.rectangle(image,
                      pt1=(x, y), pt2=(x + w, y + h),
                      color=color, thickness=bbox_thickness)

        if title is not None:
            # Bounding box title parameters:
            font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, bbox_thickness / 4, int(bbox_thickness / 2)
            text_size, _ = cv2.getTextSize(title, font, font_scale, font_thickness)
            text_w, text_h = text_size

            xo = x
            yo = (y - (text_h + bbox_thickness)) \
                if (y >= (2 * text_h) + bbox_thickness) \
                else (y + h + bbox_thickness)

            # Draw black background rectangle and add text to the bounding box:
            cv2.rectangle(image, pt1=(xo, yo), pt2=(xo + text_w, yo + text_h), color=color, thickness=-1)
            cv2.putText(image, title,
                        org=(xo, yo + text_h),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=font_thickness)

        return image

    def pick_color_palette(self, number: int) -> List:
        """Returns standard RGB colors palette (+ random colors, if needed).

        Args:
            number: the number of RGB color triplets to fetch.

        Returns:
            A list of triplets with a `number` RGB colors. The first twelve
            colors follow a standard palette, the remaining colors will be
            randomly picked.
        """
        palette = [(166, 206, 227),
                   (253, 191, 111),
                   (178, 223, 138),
                   (255, 255, 153),
                   (251, 154, 153),
                   (31, 120, 180),
                   (51, 160, 44),
                   (227, 26, 28),
                   (255, 127, 0),
                   (202, 178, 214),
                   (106, 61, 154),
                   (177, 89, 40)]

        if number <= len(palette):
            return palette[:number]
        else:
            extra_colors = self.pick_random_colors(number - len(palette))
            palette.extend(extra_colors)
            return palette

    @staticmethod
    def pick_random_colors(number: int) -> List:
        """
        Generate random RGB colors.

        Args:
            number: the number of RGB color triplets to fetch

        Returns:
            A list of triplets containing a `number` RGB colors. The colors are randomly picked.
        """
        r = list(np.random.randint(low=0, high=255, size=number))
        g = list(np.random.randint(low=0, high=255, size=number))
        b = list(np.random.randint(low=0, high=255, size=number))
        colors = list(zip(*[r, g, b]))
        return colors
