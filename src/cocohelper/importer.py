"""
Generate COCOHelper objects from a generic dataset interface.
"""
import datetime as dt
from pathlib import Path
from typing import Optional, Union

from PIL import Image
from tqdm import tqdm

from cocohelper import COCOHelper, COCOHelperPaths
from cocohelper.adapters.dataset_adapter import DatasetAdapter


class Importer:

    def __init__(
            self,
            adapter: DatasetAdapter
    ):
        """
        Importer can generate COCOHelper objects from a generic interface.

        We delegate to DatasetAdapter hierarchy the strategy used to load data
        from another dataset. Importer expect that a DatasetAdapter is provided
        and use it to load data in an understandable format, generating a new
        COCOHelper object.

        This can be used to convert from an arbitrary dataset format to the
        COCO format.

        Args:
            adapter: DatasetAdapter to use.
        """
        self._adapter = adapter
        self.img_dir = "images"

    def create(
            self,
            out_coco_dir: Union[str, Path],
            ann_dir: Union[str, Path] = COCOHelperPaths.ann_dir,
            img_dir: Union[str, Path] = COCOHelperPaths.img_dir,
            save_images: bool = False
    ) -> COCOHelper:
        """
        Generate the new COCOHelper, and optionally save it.

        Args:
            out_coco_dir: Root path to save the dataset.
            ann_dir: Annotation directory (relative to out_coco_path).
            img_dir: Image directory (relative to out_coco_path).
            save_images: If True saves the image to out_coco_path.

        Returns:
            A new COCOHelper.
        """
        if save_images and out_coco_dir is None:
            raise ValueError("Importer.create(): `save_images` can be true only if `out_coco_dir` is provided.")

        json_data = _get_empty_json()
        json_data['categories'] = self._adapter.get_categories()

        for idx, (image, annotations) in enumerate(tqdm(self._adapter)):
            if save_images:
                img_array = self._adapter.read_image(idx)
                image_name = Path(image['file_name']).name
                image_fname = Path(out_coco_dir) / Path(img_dir) / image_name
                image_fname.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img_array).save(image_fname)
                image['file_name'] = image_name

            json_data['images'].append(image)
            json_data['annotations'] += annotations

        return COCOHelper.load_data(json_data,
                                    coco_dir=str(out_coco_dir),
                                    ann_dir=str(ann_dir),
                                    img_dir=str(img_dir))


def _get_empty_json() -> dict:
    """
    Get an empty COCO dataset as a dict.

    Returns:
        A dict with the required fields for a COCO json file.
    """
    return {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {
            "contributor": "COCO Helpers Generator",
            "date_created": dt.datetime.now().astimezone(dt.timezone.utc).strftime("%Y-%b-%d, %I:%M:%S (%Z)"),
            "url": "",
            "version": "1.0",
            "year": int(dt.datetime.now().astimezone(dt.timezone.utc).strftime("%Y"))
        },
        "licenses": []
    }
