from cocohelper import COCOHelper
import pytest
from cocohelper.dataframe import COCODataFrame




class TestCocoDataframe:

    @pytest.fixture
    def df_imgs(self) -> COCODataFrame:
        # Arrange
        ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')
        yield ch.imgs


    def test_to_dict_not_having_index_column(self, df_imgs):
        # Act:
        imgs_dict = df_imgs.to_dict()

        # Assert:
        for column in df_imgs.columns:
            assert column in imgs_dict.keys()
        assert 'image_id' not in imgs_dict.keys()


    def test_to_dict_include_index_column(self, df_imgs):
        # Act:
        imgs_dict = df_imgs.to_dict(include_index=True)

        # Assert
        for column in df_imgs.columns:
            assert column in imgs_dict.keys()
        assert 'image_id' in imgs_dict.keys()
