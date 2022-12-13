from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from cocohelper import COCOHelper, COCOHelperPaths
from os.path import join
from pathlib import Path
from PIL import Image
import numpy as np
import json


# TODO: clean code in this file (there is a small margin of improvement)

class Transform(ABC):

    def transform_dataset(
            self,
            coco: COCOHelper,
            out_dir: Union[str, Path]
    ) -> COCOHelper:
        """Apply an abstract transformation on the whole dataset.

        TODO: should we apply on the whole dataset eagerly or use a lazy execution when the data is obtained?
              - In the first case, *apply* takes a COCODataset and returns a new modified COCODataset.
              - In the second case probably COCODataset should have a reference to a Transform and apply just-in-time
                when an element is retrieved.
        """
        if type(out_dir) == str:
            out_dir = Path(out_dir)

        json_fname = join(str(out_dir), COCOHelperPaths.ann_dir, COCOHelperPaths.ann_fname)
        images_dir = join(str(out_dir), COCOHelperPaths.img_dir)

        json_dataset = coco.to_json_dataset()
        del json_dataset['paths']
        images = []
        annotations = []
        for image in json_dataset['images']:
            # transforms example
            image_data, anns = coco.get_img_sample(img_id=image['id'])
            tr_image, tr_anns = self.apply(image_data['image'], anns)

            # save image
            image_fname = images_dir / image['file_name']
            image_fname.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(tr_image).save(image_fname)

            # change height/width
            h, w, _ = tr_image.shape
            image['height'] = h
            image['width'] = w

            images.append(image)
            annotations += tr_anns

        json_dataset['images'] = images
        json_dataset['annotations'] = annotations

        Path(json_fname).parent.mkdir(parents=True, exist_ok=True)

        with open(json_fname, 'w') as f:
            json.dump(json_dataset, f)

        return COCOHelper.load_json(json_fname)

    @abstractmethod
    def apply(
            self,
            img: np.ndarray,
            anns: List[dict]
    ) -> Tuple[np.ndarray, List[dict]]:
        """Apply the transformation to the image array and its annotations.

        Args:
            img: image array
            anns: annotations for this image

        Returns:
            Transformed image array and annotations
        """
        pass

    @staticmethod
    def compute_bbox_area(
            bbox: List[int]
    ) -> int:
        """Compute area from a bounding box.

        Args:
            bbox: bounding box.

        Returns:
            The area inside the given bounding box.
        """
        _, _, w, h = bbox
        return w * h
