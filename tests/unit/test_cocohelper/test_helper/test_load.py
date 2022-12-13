import json
import os
from os.path import join, dirname

import pytest

from cocohelper import COCOHelper


class TestFilteredImgsGetter:

    @pytest.fixture
    def coco_dir(self) -> str:
        return 'tests/data/coco_dataset'

    @pytest.fixture
    def ann_dir_name(self) -> str:
        return 'annotations'

    @pytest.fixture
    def ann_file_name(self) -> str:
        return 'coco.json'

    @pytest.fixture
    def json_file_path(self, coco_dir, ann_dir_name, ann_file_name) -> str:
        return os.path.join(coco_dir, ann_dir_name, ann_file_name)

    @pytest.fixture
    def json_data(self, json_file_path) -> str:
        with open(json_file_path, 'r') as f:
            return f.read()

    @pytest.fixture
    def dict_data(self, json_data) -> dict:
        return json.loads(json_data)

    def test_load_from_json_file_path(self, json_file_path):
        coco = COCOHelper.load_json(json_file_path)
        assert coco is not None

    def test_load_from_json_string(self, json_data, coco_dir, ann_dir_name):
        annotations = json.loads(json_data)
        coco = COCOHelper.load_data(annotations, coco_dir=dirname(coco_dir), ann_dir=ann_dir_name)
        assert coco is not None

    def test_load_from_json_dict(self, dict_data, coco_dir, ann_dir_name):
        coco = COCOHelper.load_data(dict_data, coco_dir=dirname(coco_dir), ann_dir=ann_dir_name)
        assert coco is not None
