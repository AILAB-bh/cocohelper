"""
Access to different joins of a COCO dataset tables.
"""
from typing import TYPE_CHECKING
from pandas import DataFrame
from cocohelper.dataframe import COCODataFrame


if TYPE_CHECKING:
    from cocohelper import COCOHelper


class COCOJoins:

    def __init__(self, coco_helper: "COCOHelper"):
        """
        Enable easy access to different joins of a COCO dataset tables.

        Args:
            coco_helper: the COCOHelper object representing a COCO dataset.
        """
        self._ch = coco_helper

    @property
    def anns_imgs(self) -> COCODataFrame:
        """Returns a left join between anns and imgs."""
        return self._ch.anns.cocojoin(self._ch.imgs).auto_reset_index().set_index('image_id')

    @property
    def imgs_anns(self) -> COCODataFrame:
        """Returns a left join between imgs and anns."""
        return self._ch.imgs.cocojoin(self._ch.anns).auto_reset_index().set_index('image_id')

    @property
    def anns_cats(self) -> COCODataFrame:
        """Returns a left join between anns and cats."""
        data = self._ch.anns.cocojoin(self._ch.cats).auto_reset_index().set_index('annotation_id')
        data["category_name"] = data["name"]  # remove "name" ambiguity adding a new column "category_name"
        return data

    @property
    def cats_anns(self) -> COCODataFrame:
        """Returns a left join between cats and anns."""
        data = self._ch.cats.cocojoin(self._ch.anns).auto_reset_index().set_index('category_id')
        data["category_name"] = data["name"]  # remove "name" ambiguity adding a new column "category_name"
        return data

    @property
    def anns_cats_imgs(self) -> COCODataFrame:
        """Returns a left join between anns, cats and imgs."""
        return self.anns_cats.cocojoin(self._ch.imgs).auto_reset_index().set_index('annotation_id')

    @property
    def anns_imgs_cats(self) -> COCODataFrame:
        """Returns a left join between anns, imgs and cats."""
        return self.anns_imgs.cocojoin(self._ch.cats).auto_reset_index().set_index('annotation_id')

    @property
    def imgs_anns_cats(self) -> COCODataFrame:
        """Returns a left join between imgs, anns and cats."""
        return self._ch.imgs.cocojoin(self.anns_cats).auto_reset_index().set_index('image_id')

    @property
    def imgs_cats_anns(self) -> COCODataFrame:
        """Returns a left join between imgs, cats and anns."""
        return self._ch.imgs.cocojoin(self.cats_anns).auto_reset_index().set_index('image_id')

    @property
    def cats_anns_imgs(self) -> COCODataFrame:
        """Returns a left join between cats, anns and imgs."""
        return self.cats_anns.cocojoin(self._ch.imgs)

    @property
    def cats_imgs_anns(self) -> COCODataFrame:
        """Returns a left join between imgs, anns and cats."""
        return self._ch.cats.cocojoin(self.imgs_anns).auto_reset_index().set_index('category_id')

    def extract_cats(
            self,
            joined_cats: DataFrame
    ) -> DataFrame:
        """
        Extract cats view from a cats dataframe merged with other dataframes.

        Useful to extract columns compatible with standard coco categories and
        merge/assign to COCOHelper cats property.

        Args:
            joined_cats: a dataframe that contains at least the standard coco
                categories columns.

        Returns:
            A dataframe containing only the standard coco categories columns.
        """
        try:
            joined_cats = joined_cats.auto_reset_index().set_index('category_id', drop=False)
        except KeyError:
            assert joined_cats.index.name == 'category_id'
        joined_cats = joined_cats[self._ch.cats.columns]

        # Faster than simply drop_duplicate():
        # https://stackoverflow.com/questions/13035764/remove-pandas-rows-with-duplicate-indices
        return joined_cats[~joined_cats.index.duplicated()]

    def extract_imgs(
            self,
            joined_imgs: DataFrame
    ) -> DataFrame:
        """
        Extract imgs view from an imgs dataframe merged with other dataframes.

        Useful to extract columns compatible with standard coco images and
        merge/assign to COCOHelper imgs property.

        Args:
            joined_imgs: a dataframe that contains at least the standard coco
                images columns.

        Returns:
            A dataframe containing only the standard coco images columns.
        """
        try:
            joined_imgs = joined_imgs.reset_index().set_index('image_id', drop=True)
        except KeyError:
            assert joined_imgs.index.name == 'image_id'
        joined_imgs = joined_imgs[self._ch.imgs.columns]

        # Faster than simply drop_duplicate():
        # https://stackoverflow.com/questions/13035764/remove-pandas-rows-with-duplicate-indices
        return joined_imgs[~joined_imgs.index.duplicated()]

    def extract_anns(
            self,
            joined_anns: DataFrame
    ) -> DataFrame:
        """
        Get annotation view from an annotation dataframe merged with other
        dataframes.

        Useful to extract columns compatible with standard coco annotations and
        merge/assign to COCOHelper annotations property.

        Args:
            joined_anns: a dataframe that contains at least the standard coco
                annotations columns.

        Returns:
            A dataframe containing only the standard coco annotations columns.
        """
        try:
            joined_anns = joined_anns.reset_index().set_index('annotation_id', drop=True)
        except KeyError:
            assert joined_anns.index.name == 'annotation_id'
        joined_anns = joined_anns[self._ch.anns.columns]

        # Faster than simply drop_duplicate():
        # https://stackoverflow.com/questions/13035764/remove-pandas-rows-with-duplicate-indices
        return joined_anns[~joined_anns.index.duplicated()]
