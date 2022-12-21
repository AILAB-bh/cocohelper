"""
Check COCO dataset validity based on data ids and directory tree.
"""
from typing import Dict, List, Optional, Type, Union, Sequence, Any, Tuple
from functools import partial
import logging
import os
from cocohelper import COCOHelper


class COCOValidator:

    def __init__(
            self,
            coco_helper: COCOHelper
    ):
        """This class validates COCO datasets."""
        self.helper = coco_helper

    def validate_dir(
            self,
            dataset_dir: str,
            json_fname: str = 'coco.json'
    ) -> bool:
        """Checks the COCO dataset validity based on the dir structure."""
        # TODO: throw specific exception instead of returning or using asserts
        logging.info("\n")
        logging.info("Checking COCO dataset validity...")
        fail_message = " | Test failed: this is not a valid COCO dataset:"

        # check folder tree:
        if not self._has_valid_dataset_tree():
            logging.error(f"{fail_message} Folders are not organised as expected by a COCO dataset.")
            return False

        # make sure json_fname is a valid name for the json file:
        suffix = json_fname.rsplit(os.sep)[-1]
        fname = os.path.join(os.path.join(dataset_dir, "annotations"), suffix)
        if not fname.endswith(".json"):
            raise ValueError("The annotation file name must end with the extension '.json'")

        logging.info(" | Test passed.")
        return True

    def validate_dataset(self) -> bool:
        """
        Check if this is a valid COCO dataset.

        Returns:
            True if this is a valida dataset.
        """
        logging.info("\n")
        logging.info("Checking COCO dataset validity...")
        fail_message = " | Test failed: this is not a valid COCO dataset:"

        # check folder tree:
        if not self._has_valid_dataset_tree():
            logging.error(f"{fail_message} Folders are not organised as expected by a COCO dataset.")
            return False

        # start verifying dataset validity
        data = self.helper.to_json_dataset()
        if not self._json_has_mandatory_keys(data):
            logging.error(f"{fail_message} There are missing mandatory keys in the json file.")
            return False
        if not self._categories_have_mandatory_keys(data):
            logging.error(f"{fail_message} There are missing mandatory keys in the COCO categories.")
            return False
        if not self._images_have_mandatory_keys(data):
            logging.error(f"{fail_message} There are missing mandatory keys in the COCO images.")
            return False
        if not self._annotations_have_mandatory_keys(data):
            logging.error(f"{fail_message} There are missing mandatory keys in the COCO annotations.")
            return False
        if not self._category_ids_are_unique(data):
            logging.error(f"{fail_message} There are duplicated category ids.")
            return False
        if not self._licenses_ids_are_unique(data):
            logging.error(f"{fail_message} There are duplicated licenses ids.")
            return False
        if not self._image_ids_are_unique(data):
            logging.error(f"{fail_message} There are duplicated image ids.")
            return False
        if not self._annotation_ids_are_unique(data):
            logging.error(f"{fail_message} There are duplicated annotation ids.")
            return False
        if not self._annotations_have_valid_image_id(data):
            logging.error(f"{fail_message} There are annotations with invalid image id.")
            return False
        if not self._annotations_have_valid_category_id(data):
            logging.error(f"{fail_message} There are annotations with invalid category id.")
            return False
        logging.info(" | Test passed.")
        return True

    def _has_valid_dataset_tree(self) -> bool:
        """
        Check dataset directory tree validity

        Returns:
            True if the dataset tree is valid, False otherwise
        """
        # 1) there exist a folder named annotations:
        cond1 = os.path.exists(os.path.join(self.helper.root_path, "annotations"))

        # TODO: 2) all the data is under a separate sub-folder. Consider adding checks here
        # folder_content = glob(os.path.join(self.dataset_dir, "*.*"))
        # cond2 = len(folder_content) == 0
        cond2 = True

        has_valid_tree = cond1 and cond2
        return has_valid_tree

    @staticmethod
    def _json_has_mandatory_keys(
            json_data: Dict,
            mandatory_keys: Optional[List] = None
    ) -> bool:
        """
        Check if the input COCO annotation json file has the mandatory keys.

        Correct COCO annotation json files must have the following keys: "images",
        "annotations", "categories". There are some exceptions: for example, the
        "categories" field does not exist for caption annotations. In these cases,
        you can explicitly feed the mandatory_keys to the function.

        Args:
            json_data: data from a coco json file.
            mandatory_keys: mandatory keys for a valid COCO json file.

        Returns:
            True if the json file has mandatory keys, False otherwise.
        """
        if mandatory_keys is None:
            mandatory_keys = ["images", "annotations", "categories"]
        for k in mandatory_keys:
            if k not in json_data.keys():
                return False
        return True

    @staticmethod
    def _images_have_mandatory_keys(
            json_data: Dict,
            mandatory_keys: Optional[List] = None
    ) -> bool:
        """
        Check if images have the mandatory keys.

        Args:
            json_data: data from a coco json file.
            mandatory_keys: mandatory keys for a valid COCO json file.

        Returns:
            True if the dictionary has mandatory keys, False otherwise.
        """
        if mandatory_keys is None:
            mandatory_keys = ["id", "width", "height", "file_name"]
        fail_message = " -- Error for image data having id: "

        images = json_data["images"]
        for img in images:
            keys = img.keys()
            img_id = img["id"]
            for k in mandatory_keys:
                if k not in keys:
                    raise ValueError(f"{fail_message}{img_id}. Missing mandatory key {k}.")

            assert_type = partial(_assert_dict_value_type, dictionary=img, msg_header=f"{fail_message} {img_id}")
            assert_type(key="id", expected_types=[int])
            assert_type(key="width", expected_types=[int])
            assert_type(key="height", expected_types=[int])
            assert_type(key="file_name", expected_types=[str])

        return True

    @staticmethod
    def _annotations_have_mandatory_keys(
            json_data: Dict,
            mandatory_keys: Optional[List] = None
    ) -> bool:
        """
        Check if annotations have the mandatory keys.

        Args:
            json_data: data from a coco json file.
            mandatory_keys: mandatory keys for a valid COCO json file.

        Returns:
            True if the dictionary has mandatory keys, False otherwise.
        """
        if mandatory_keys is None:
            mandatory_keys = ["id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd"]
        fail_message = " -- Error for annotation data having id: "

        annotations = json_data["annotations"]
        for ann in annotations:
            keys = ann.keys()
            ann_id = ann["id"]
            for k in mandatory_keys:
                if k not in keys:
                    raise ValueError(f"{fail_message}{ann_id}. Missing mandatory key {k}.")

            assert_type = partial(_assert_dict_value_type, dictionary=ann, msg_header=f"{fail_message} {ann_id}")
            assert_type(key="id", expected_types=[int])
            assert_type(key="image_id", expected_types=[int])
            assert_type(key="category_id", expected_types=[int])
            assert_type(key="area", expected_types=[float, int])
            assert_type(key="bbox", expected_types=[list])

            key = "iscrowd"
            if ann[key] not in [0, 1]:
                raise ValueError(f"{fail_message} -- The value of '{key}' must be in [0, 1].")

        return True

    @staticmethod
    def _categories_have_mandatory_keys(
            json_data: Dict,
            mandatory_keys: Optional[List] = None,
            recommended_keys: Optional[List] = None
    ) -> bool:
        """
        Check if categories have the mandatory keys.
        
        Args:
            json_data: data from a coco json file.
            mandatory_keys: mandatory keys for a valid COCO json file.
                If missing, validation fails.
            recommended_keys: recommended keys for a valid COCO json file.
                If missing, generates a warning. The check on these keys is
                run only after checking the mandatory keys.

        Returns:
            True if the dictionary has mandatory keys, False otherwise.
        """
        if mandatory_keys is None:
            mandatory_keys = ["id", "name"]
        if recommended_keys is None:
            recommended_keys = ["supercategory"]
        fail_message = " -- Error for category data having id: "
        warn_message = " -- Warning for category data having id: "

        categories = json_data["categories"]
        for cat in categories:
            keys = cat.keys()
            cat_id = cat["id"]
            for k in mandatory_keys:
                if k not in keys:
                    _id = cat["id"]
                    logging.error(f"{fail_message} {cat_id}. Missing mandatory key {k}.")
                    return False
            for k in recommended_keys:
                if k not in keys:
                    _id = cat["id"]
                    logging.warning(f"{warn_message} {cat_id}. Missing recommended key {k}.")

            assert_type = partial(_assert_dict_value_type, dictionary=cat, msg_header=f"{fail_message} {cat_id}")
            assert_type(key="id", expected_types=[int])
            assert_type(key="name", expected_types=[str])
            assert_type(key="supercategory", expected_types=[str])

        return True

    @staticmethod
    def _annotations_have_valid_category_id(
            json_data: Dict
    ) -> bool:
        """
        Check that annotations have a valid category id.

        Args:
            json_data: data from a coco json file.
        """
        categories = [c["id"] for c in json_data["categories"]]
        labels = json_data["annotations"]
        for lbl in labels:
            if lbl["category_id"] not in categories:
                return False
        return True

    @staticmethod
    def _annotations_have_valid_image_id(
            json_data: Dict
    ) -> bool:
        """
        Check that annotations have a valid image id.

        Args:
            json_data: data from a coco json file.
        """
        image_ids = [img["id"] for img in json_data["images"]]
        labels = json_data["annotations"]
        for lbl in labels:
            if lbl["image_id"] not in image_ids:
                return False
        return True

    @staticmethod
    def _category_ids_are_unique(
            json_data: Dict
    ) -> bool:
        """
        Check if there are duplicated category ids.

        Args:
            json_data: data from a coco json file.
        """
        id_list = [el["id"] for el in json_data["categories"]]
        if len(id_list) == len(set(id_list)):
            return True
        return False

    @staticmethod
    def _licenses_ids_are_unique(
            json_data: Dict
    ) -> bool:
        """
        Check if there are duplicated licenses ids.

        Args:
            json_data: data from a coco json file.
        """
        id_list = [el["id"] for el in json_data["licenses"]]
        if len(id_list) == len(set(id_list)):
            return True
        return False

    @staticmethod
    def _image_ids_are_unique(
            json_data: Dict
    ) -> bool:
        """
        Check if there are duplicated image ids.

        Args:
            json_data: data from a coco json file.
        """
        id_list = [el["id"] for el in json_data["images"]]
        if len(id_list) == len(set(id_list)):
            return True
        return False

    @staticmethod
    def _annotation_ids_are_unique(
            json_data: Dict
    ) -> bool:
        """
        Check if there are duplicated annotations ids.

        Args:
            json_data: data from a coco json file.

        Returns:
            True if the annotation ids are unique, False otherwise.
        """
        id_list = [el["id"] for el in json_data["annotations"]]
        if len(id_list) == len(set(id_list)):
            return True
        return False


def _assert_dict_value_type(
        dictionary: Dict,
        key: Any,
        expected_types: List[Type],
        msg_header=""
) -> None:
    """
    Check if a dictionary item has the expected type under the given key.

    Args:
        dictionary: a dictionary containing the pairs (key, value) to be tested
        key: the key of the dictionary whose value must be tested
        expected_types: a list of expected types for the value
        msg_header: a header string for the error message delivered for check
            fails.

    Raises:
        TypeError if the item in the dictionary is not of the expected type.
    """
    if key not in dictionary.keys():
        raise KeyError(f"{msg_header} -- Missing dictionary key.")
    type_ok = False
    for exp_type in expected_types:
        if isinstance(dictionary[key], exp_type):
            type_ok = True
            break
    if not type_ok:
        raise TypeError(f"{msg_header} -- Type of '{key}' must be in {expected_types}.")
