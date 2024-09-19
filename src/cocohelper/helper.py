"""
Represent a dataset in the COCO format.
"""
from __future__ import annotations
from typing import Union, Optional, Tuple, Type, Dict, Sequence, TYPE_CHECKING
from pycocotools.coco import COCO
from json import JSONDecodeError
from pandas import DataFrame
from pathlib import Path
from PIL import Image
import datetime as dt
import pandas as pd
import numpy as np
import dataclasses
import logging
import copy
import json
import os
from cocohelper.utils.dataframe import df_to_records, drop_duplicate_rows, fix_fk_after_drop_duplicate
from cocohelper.errors.not_found_error import COCOImageNotFoundError, COCOAnnotationNotFoundError
from cocohelper.filters.filter import Filter, AndFilter, NotFilter, ComposeFilter
from cocohelper.errors.validation_error import COCOValidationError
from cocohelper.utils.colmapper import ColMap, ColsMapper
from cocohelper.filters import cocofilters as cfilters
from cocohelper.joins import COCOJoins, COCODataFrame
from cocohelper.utils.timer import Timer
from cocohelper.utils.types._types import IDXSelector
from cocohelper.validator import COCOValidator


# IMPORTS FOR TYPE-CHECKING ONLY
if TYPE_CHECKING:
    from cocohelper.transforms import Transform


@dataclasses.dataclass(frozen=True)
class COCOColsMapper(ColsMapper):
    """(DEPRECATED) Enable the possibility to remap specific columns in COCOHelper"""

    cat: ColMap = ColMap(orig='id', new='category_id')
    img: ColMap = ColMap(orig='id', new='image_id')
    ann: ColMap = ColMap(orig='id', new='annotation_id')
    lic: ColMap = ColMap(orig='id', new='license_id')


@dataclasses.dataclass
class COCOHelperPaths:
    """Information about folder and file organization for a COCO dataset."""
    ann_fname: str = 'coco.json'
    ann_dir: str = 'annotations/'
    img_dir: str = 'images/'


class COCOHelper:

    def __init__(
            self,
            img_df: DataFrame,
            ann_df: DataFrame,
            cat_df: DataFrame,
            lic_df: Optional[DataFrame] = None,
            info: Optional[dict] = None,
            coco_dir: Union[str, Path] = './',
            paths: Optional[COCOHelperPaths] = None,
            validate: bool = True
    ) -> None:
        """
        Represent a dataset in the COCO format.

        To create an instance of COCOHelper is advisable to use the `load` methods.

        Args:
            img_df: DataFrame of images.
            ann_df: DataFrame of annotations.
            cat_df: DataFrame of categories, optional.
            lic_df: DataFrame of licenses, optional.
            info: Info dict, optional.
            coco_dir: Root directory of the dataset, optional.
            paths: COCOHelperPaths, used to customize directory structure.
            validate: If True, validate the COCO dataset and raise an error if
              invalid.

        Raises:
            COCOValidationError if the input COCO dataset is not valid. This
            check is performed only if `validate=True`.
        """
        if lic_df is None:
            lic_df = DataFrame(columns=['id', 'name', 'url'])

        self._root_path = coco_dir
        self._paths = paths if paths is not None else COCOHelperPaths()
        self._imgs = COCODataFrame(img_df, 'image')
        self._anns = COCODataFrame(ann_df, 'annotation')
        self._cats = COCODataFrame(cat_df, 'category')  # if cat_df is not None else None
        self._lics = COCODataFrame(lic_df, 'license') if cat_df is not None else None
        self._info = info if info is not None else COCOHelper.new_info_dict()
        self._colmaps: COCOColsMapper = COCOColsMapper()

        if validate:
            is_valid = self.validator.validate_dataset()
            if not is_valid:
                raise COCOValidationError()

    def copy(
            self,
            cat_df: Optional[DataFrame] = None,
            img_df: Optional[DataFrame] = None,
            ann_df: Optional[DataFrame] = None,
            lic_df: Optional[DataFrame] = None,
            info: Optional[dict] = None,
            validate: bool = True,
    ) -> COCOHelper:
        """
        Copy the dataset and optionally change some dataframes.
        
        When changing categories or images, annotations that
        result as invalid will be removed.

        Args:
            cat_df: New category dataframe, optional
            img_df: New image dataframe, optional
            ann_df: New annotation dataframe, optional
            lic_df: New license dataframe, optional
            info: New info dict, optional
            validate: If True, validate the COCO dataset and raise an error if
              invalid

        Returns:
            A new `COCOHelper` object.
        """
        helper = copy.deepcopy(self)
        if cat_df is not None:
            helper._cats = COCODataFrame(pd.DataFrame(cat_df), 'category')
        if img_df is not None:
            helper._imgs = COCODataFrame(pd.DataFrame(img_df), 'image')
        if ann_df is not None:
            helper._anns = COCODataFrame(pd.DataFrame(ann_df), 'annotation')
        if lic_df is not None:
            helper._lics = COCODataFrame(pd.DataFrame(lic_df), 'license')
        if info is not None:
            helper._info = info

        helper._remove_unlinked_anns()

        # validate the dataset
        if validate:
            is_valid = self.validator.validate_dataset()
            if not is_valid:
                raise COCOValidationError()

        return helper

    def _remove_unlinked_anns(self):
        """Remove annotations that have non-existing image or categories ids."""
        linked_images = self.anns['image_id'].isin(self.imgs.index)
        linked_cats = self.anns['category_id'].isin(self.cats.index)

        # Update the annotations in self to remove unlinked anns
        self._anns = COCODataFrame(pd.DataFrame(self.anns[linked_images & linked_cats]), 'annotation')

    def to_coco(self) -> COCO:
        """Convert `COCOHelper` to `pycocotools.COCO`"""
        coco = COCO(None)
        coco.dataset = self.to_json_dataset()
        coco.createIndex()
        return coco

    #
    # # # # # # # # # # # #
    # PERSISTENCE METHODS #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def save(
            self,
            coco_dir: Union[str, Path],
            fix_img_path: bool = False,
            copy_images: bool = False
    ) -> None:
        """
        Save the current `COCOHelper` to a directory.

        Args:
            coco_dir: Output root directory.
            fix_img_path: NotImplemented.
            copy_images: NotImplemented.

        Returns:
            None.
        """
        annotation_file = Path(coco_dir) / Path(self.paths.ann_dir).name / self.paths.ann_fname
        self.write_annotations_file(annotation_file)

        if copy_images:
            self._copy_images(Path(coco_dir) / Path(self.paths.img_dir))

        elif fix_img_path:
            raise NotImplementedError()

            # TODO: fix the self.img_dir property so that, changing the self._coco_root_dir to coco_dir,
            #       the paths to the original images are still correct.

            # TODO: we could have a different approach: call a method to change the _coco_root_dir before saving,
            #       then always save on the current _coco_root_dir (we could return a copy of the cocoh object when
            #       we change the _coco_root_dir). EG:
            #           cocoh.set_coco_root_dir('./newpath').save()
            #       or:
            #           cocoh_newpath = cocoh.set_coco_root_dir('./newpath')
            #           cocoh_newpath.save()

    @classmethod
    def load(
            cls,
            coco_dir: str,
            ann_fname: str = COCOHelperPaths.ann_fname,
            ann_dir: str = COCOHelperPaths.ann_dir,
            img_dir: str = COCOHelperPaths.img_dir,
            validate: bool = False
    ) -> COCOHelper:
        """
        Create a COCOHelper from a COCO dataset stored in a directory.

        Args:
            coco_dir: path to the directory containing the dataset.
            ann_fname: name of the annotation file to be load.
            ann_dir: name/relative-path to the directory where annotations are
                stored.
            img_dir: name/relative-path to the directory where images are stored.
            validate: If True, validate the dataset.

        Returns:
            A COCOHelper object.
        """

        paths = COCOHelperPaths(ann_fname=ann_fname, ann_dir=ann_dir, img_dir=img_dir)
        annotation_file_path = os.path.join(coco_dir, paths.ann_dir, paths.ann_fname)
        return COCOHelper.load_json(annotation_file_path, img_dir=paths.img_dir, validate=validate)

    @classmethod
    def load_json(
            cls,
            json_annotations_file: str,
            img_dir: str = COCOHelperPaths.img_dir,
            validate: bool = False
    ) -> COCOHelper:
        """
        Create COCOHelper from json annotation file of the COCO dataset stored in a directory.

        Args:
            json_annotations_file: path to the json file containing the dataset
                annotations.
            img_dir: name/relative-path to the directory where images are
                stored, respect to the coco dataset root.
            validate: If True, validate the dataset.

        Returns:
            A COCOHelper object.
        """
        # TODO add coco_dir parameter and change the anns_dir accordingly if specified.
        try:
            annotations = cls._read_annotations_file(json_annotations_file)
            ann_dir = os.path.dirname(json_annotations_file)
            ann_fname = os.path.basename(json_annotations_file)
        except FileNotFoundError as e:
            logging.warning(f"Cannot read annotation file."
                            f"If you want to load a json string or a dict containing annotations use load.")
            raise e
        coco_dir = os.path.dirname(ann_dir)
        return cls.load_data(annotations, coco_dir, ann_fname, ann_dir, img_dir, validate)


    @classmethod
    def load_data(
            cls,
            annotations: Dict[str, pd.DataFrame],
            coco_dir: str,
            ann_fname: str = COCOHelperPaths.ann_fname,
            ann_dir: str = COCOHelperPaths.ann_dir,
            img_dir: str = COCOHelperPaths.img_dir,
            validate: bool = False
    ) -> COCOHelper:

        with Timer("Loading dataframes...", "Done: ", log_fn=logging.info):
            imgs_df = DataFrame.from_records(annotations['images'])
            anns_df = DataFrame.from_records(annotations['annotations'])
            cats_df = None
            lics_df = None
            info = None
            if 'categories' in annotations.keys():
                cats_df = DataFrame.from_records(annotations['categories'])
            if 'licenses' in annotations.keys():
                lics_df = DataFrame.from_records(annotations['licenses'])
            if 'info' in annotations.keys():
                info = annotations['info']

            paths = COCOHelperPaths(ann_fname=ann_fname, ann_dir=ann_dir, img_dir=img_dir)
            coco_helper = COCOHelper(imgs_df, anns_df, cats_df, lics_df, info,
                                     coco_dir=coco_dir, paths=paths, validate=validate)
        return coco_helper

    def write_annotations_file(self, annotation_file_path: Union[str, Path]):
        """Save the current COCOHelper as a COCO json file."""
        os.makedirs(os.path.dirname(annotation_file_path), exist_ok=True)
        with open(annotation_file_path, 'w') as ann_file:
            ann_file.write(json.dumps(self.to_json_dataset(), indent=4))

    @classmethod
    def _read_annotations_file(cls, annotation_file: str) -> dict:
        """Read a COCO json file as a dict."""
        with Timer("Loading annotations into memory...", "Done: ", log_fn=logging.info):
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            assert type(annotations) == dict, 'annotation file format {} not supported'.format(type(annotations))
        return annotations

    def to_json_dataset(self) -> dict:
        """Convert the current COCOHelper to a dict with the same structure of the COCO json file."""
        return {
            'categories': df_to_records(self.cats, self._colmaps.cat),
            'images': df_to_records(self.imgs, self._colmaps.img),
            'annotations': df_to_records(self.anns, self._colmaps.ann),
            'licenses': df_to_records(self.licenses, self._colmaps.lic),
            'info': self._info,
            # 'cocohelper_paths': self._paths,
        }

    def _copy_images(self, target_img_dir: Union[str, Path]):
        raise NotImplementedError()  # TODO: TBD
        # target_img_dir = Path(target_img_dir)
        # orig_img_dir = Path(self._coco_root_dir) / Path(self.img_dir)
        # os.makedirs(target_img_dir, exist_ok=True)
        # for img_id, img_fname in self.img_nms_by_id.items():
        #     shutil.copy(orig_img_dir / img_fname, target_img_dir / img_fname)

    #
    # # # # # # # # # #
    # CORE PROPERTIES #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @property
    def cats(self) -> COCODataFrame:
        """Dataframe containing the categories data of the COCO dataset."""
        return self._cats

    @property
    def imgs(self) -> COCODataFrame:
        """Dataframe containing the images metadata of the COCO dataset."""
        return self._imgs

    @property
    def anns(self) -> COCODataFrame:
        """Dataframe containing the annotations data of the COCO dataset."""
        return self._anns

    @property
    def licenses(self) -> Optional[COCODataFrame]:
        """Dataframe containing the licenses of the COCO dataset."""
        return self._lics

    @property
    def info(self):
        """Dataframe containing extra information of the COCO dataset."""
        return copy.copy(self._info)

    @property
    def paths(self) -> COCOHelperPaths:
        """Information about folder and file organization for a COCO dataset."""
        return copy.copy(self._paths)

    @property
    def root_path(self):
        """Path to the root directory containing the COCO dataset."""
        return Path(self._root_path)

    @root_path.setter
    def root_path(self, value: Union[str, Path]):
        self._root_path = Path(value)

    @property
    def joins(self):
        """Get a COCOJoins object, that enable easy access to different joins dataset tables."""
        return COCOJoins(self)

    @property
    def validator(self):
        """Get a COCOValidator object, that enable easy access to different validation methods."""
        return COCOValidator(json_data=self.to_json_dataset(), dataset_dir=self.root_path)

    #
    # # # # # # # # # # # # # # #
    # SPECIAL DATAFRAME GETTERS #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def unlabelled_imgs(self) -> pd.DataFrame:
        """
        Get only the unlabelled images as a DataFrame.

        Returns:
            A pandas.DataFrame containing the unlabelled images.
        """
        return self.imgs[~self.imgs.index.isin(self.anns.image_id)]

    @property
    def labelled_imgs(self) -> pd.DataFrame:
        """
        Get only the labelled images as a DataFrame.

        Returns:
            A pandas.DataFrame containing the labelled images.
        """
        return self.imgs[self.imgs.index.isin(self.anns.image_id)]

    #
    # # # # # # # # # # # # # # # # # #
    # SPECIAL IMAGES DROPPING METHODS #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def drop_unlabelled(self) -> COCOHelper:
        """
        Get a new `COCOHelper` dataset that does not contain unlabelled images.

        Returns:
            A new `COCOHelper` object containing only labelled images.
        """
        return self.copy(img_df=self.labelled_imgs)

    def drop_labelled(self) -> COCOHelper:
        """
        Get a new COCOHelper dataset that does only contain unlabelled images.

        Returns:
            A new `COCOHelper` object containing only unlabelled images.
        """
        return self.copy(img_df=self.unlabelled_imgs)

    #
    # # # # # # # # #
    # COCO FILTERS  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def filter_cats(
            self,
            cfilter: Optional[Filter] = None,
            *,
            cat_ids: Optional[IDXSelector[int]] = None,
            cat_nms: Optional[IDXSelector[str]] = None,
            supercat_nms: Optional[IDXSelector[str]] = None,
            composition: Type[ComposeFilter] = AndFilter,
            invert: bool = False
    ) -> COCOHelper:
        """
        Get a copy of the dataset with filtered categories.

        Args:
            cfilter: a custom Filter for the COCOHelper.
            cat_ids: a filter for the category ids.
            cat_nms: a filter for the category names.
            supercat_nms: a filter for the super-category names.
            composition: a composition type for the filters (defaults to "and"
                behavior between each filter).
            invert: if True, invert the way the filter works.

        Returns:
            A COCOHelper with data filtered according to the given filters.
        """
        cats = self.filtered_cats(
            cfilter,
            cat_ids=cat_ids,
            cat_nms=cat_nms,
            supercat_nms=supercat_nms,
            composition=composition,
            invert=invert)
        return self.copy(cat_df=cats)

    def filter_imgs(
            self,
            cfilter: Optional[Filter] = None,
            *,
            img_ids: Optional[IDXSelector[int]] = None,
            img_nms: Optional[IDXSelector[str]] = None,
            cat_ids: Optional[IDXSelector[int]] = None,
            cat_nms: Optional[IDXSelector[str]] = None,
            supercat_nms: Optional[IDXSelector[str]] = None,
            composition: Type[ComposeFilter] = AndFilter,
            invert: bool = False
    ) -> COCOHelper:
        """
        Get a copy of the dataset with filtered images.

        Args:
            cfilter: a custom Filter for the COCOHelper.
            img_ids: a filter for the image ids.
            img_nms: a filter for the image file names.
            cat_ids: a filter for the category ids.
            cat_nms: a filter for the category names.
            supercat_nms: a filter for the super-category names.
            composition: a composition type for the filters (defaults to "and"
                behavior between each filter).
            invert: if True, invert the way the filter works.

        Returns:
            A COCOHelper with data filtered according to the given filters.
        """
        imgs = self.filtered_imgs(cfilter,
                                  img_ids=img_ids, img_nms=img_nms,
                                  cat_ids=cat_ids, cat_nms=cat_nms, supercat_nms=supercat_nms,
                                  composition=composition, invert=invert)
        return self.copy(img_df=imgs)

    def filter_anns(
            self,
            cfilter: Optional[Filter] = None,
            *,
            ann_ids: Optional[IDXSelector[int]] = None,
            img_ids: Optional[IDXSelector[int]] = None,
            img_nms: Optional[IDXSelector[str]] = None,
            cat_ids: Optional[IDXSelector[int]] = None,
            cat_nms: Optional[IDXSelector[str]] = None,
            supercat_nms: Optional[IDXSelector[str]] = None,
            area_rng: Optional[Tuple[float, float]] = None,
            is_crowd: Optional[bool] = None,
            composition: Type[ComposeFilter] = AndFilter,
            invert: bool = False
    ) -> COCOHelper:
        """
        Get a copy of the dataset with filtered annotations.

        Args:
            cfilter: a custom Filter for the COCOHelper.
            ann_ids: a filter for the annotation ids.
            img_ids: a filter for the image ids.
            img_nms: a filter for the image file names.
            cat_ids: a filter for the category ids.
            cat_nms: a filter for the category names.
            supercat_nms: a filter for the super-category names.
            area_rng: a filter for the annotation area.
            is_crowd: a filter for the annotation being a crowd or not
                ("is_crowd" in the annotation of the COCO JSON file).
            composition: a composition type for the filters (defaults to "and"
                behavior between each filter).
            invert: if True, invert the way the filter works.

        Returns:
            A COCOHelper with data filtered according to the given filters.
        """
        anns = self.filtered_anns(cfilter,
                                  ann_ids=ann_ids, img_ids=img_ids, img_nms=img_nms,
                                  cat_ids=cat_ids, cat_nms=cat_nms, supercat_nms=supercat_nms,
                                  area_rng=area_rng, is_crowd=is_crowd,
                                  composition=composition, invert=invert)
        return self.copy(ann_df=anns)

    def filter(
            self,
            cfilter: Filter,
            *,
            ann_ids: Optional[IDXSelector[int]] = None,
            img_ids: Optional[IDXSelector[int]] = None,
            img_nms: Optional[IDXSelector[str]] = None,
            cat_ids: Optional[IDXSelector[int]] = None,
            cat_nms: Optional[IDXSelector[str]] = None,
            supercat_nms: Optional[IDXSelector[str]] = None,
            area_rng: Optional[Tuple[float, float]] = None,
            is_crowd: Optional[bool] = None,
            composition: Type[ComposeFilter] = AndFilter,
            invert: bool = False,
            drop_orphans: bool = True
    ) -> COCOHelper:
        """
        Get a copy of the dataset with the applied filters.

        Args:
            cfilter: a custom Filter for the COCOHelper.
            ann_ids: a filter for the annotation ids.
            img_ids: a filter for the image ids.
            img_nms: a filter for the image file names.
            cat_ids: a filter for the category ids.
            cat_nms: a filter for the category names.
            supercat_nms: a filter for the super-category names.
            area_rng: a filter for the annotation area.
            is_crowd: a filter for the annotation being a crowd or not
                ("is_crowd" in the annotation of the COCO JSON file).
            composition: a composition type for the filters (defaults to "and"
                behavior between each filter).
            invert: if True, invert the way the filter works.
            drop_orphans: if True, drop orphans when applying the filter.

        Returns:
            A COCOHelper with data filtered according to the given filters.
        """
        if cfilter is None:
            # TODO: check that at least one of the input is not None
            cfilter = composition(cfilters.imgs_filter(img_ids, img_nms, composition),
                                  cfilters.cats_filter(cat_ids, cat_nms, supercat_nms, composition),
                                  cfilters.anns_filter(ann_ids, area_rng, is_crowd, composition))
            if invert:
                cfilter = NotFilter(cfilter)

        if drop_orphans:
            joined_anns = self.joins.anns_cats_imgs
            joined_anns = cfilter.apply(joined_anns)
            cats = self.joins.extract_cats(joined_anns)
            imgs = self.joins.extract_imgs(joined_anns)
            anns = self.joins.extract_anns(joined_anns)
        else:
            cats = self.filtered_cats(cfilter)
            imgs = self.filtered_imgs(cfilter)
            anns = self.filtered_anns(cfilter)
        return self.copy(cat_df=cats, img_df=imgs, ann_df=anns)

    #
    # # # # # # # # # # # # # # # #
    # DATAFRAME FILTERING/GETTERS #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def filtered_cats(
            self,
            cfilter: Optional[Filter] = None,
            *,
            cat_ids: Optional[IDXSelector[int]] = None,
            cat_nms: Optional[IDXSelector[str]] = None,
            supercat_nms: Optional[IDXSelector[str]] = None,
            composition: Type[ComposeFilter] = AndFilter,
            invert: bool = False
    ) -> pd.DataFrame:
        """
        Get dataset's categories, potentially filtered by the provided filters.

        Args:
            cfilter: a custom Filter for the COCOHelper.
            cat_ids: a filter for the category ids.
            cat_nms: a filter for the category names.
            supercat_nms: a filter for the super-category names.
            composition: a composition type for the filters (defaults to "and"
                behavior between each filter).
            invert: if True, invert the way the filter works.

        Returns:
            A pandas.DataFrame containing the filtered categories.
        """
        cats = self.cats
        # cats = self.joins.imgs_anns_cats.set_index('category_id', drop=False)
        if cfilter is None:
            # TODO: check that at least one of the input is not None
            cfilter = cfilters.cats_filter(cat_ids, cat_nms, supercat_nms, composition)
        if invert:
            cfilter = NotFilter(cfilter)
        cats = cfilter.apply(cats)
        return cats.drop_duplicates()
        # return self.joins.extract_cats(cats).dropna()

    def filtered_imgs(
            self,
            cfilter: Optional[Filter] = None,
            *,
            img_ids: Optional[IDXSelector[int]] = None,
            img_nms: Optional[IDXSelector[str]] = None,
            cat_ids: Optional[IDXSelector[int]] = None,
            cat_nms: Optional[IDXSelector[str]] = None,
            supercat_nms: Optional[IDXSelector[str]] = None,
            composition: Type[ComposeFilter] = AndFilter,
            invert: bool = False
    ) -> pd.DataFrame:
        """
        Get dataset's images, after join with annotations and categories and potentially filtered by a filter.

        Args:
            cfilter: a custom Filter for the COCOHelper.
            img_ids: a filter for the image ids.
            img_nms: a filter for the image file names.
            cat_ids: a filter for the category ids.
            cat_nms: a filter for the category names.
            supercat_nms: a filter for the super-category names.
            composition: a composition type for the filters (defaults to "and"
                behavior between each filter).
            invert: if True, invert the way the filter works.

        Returns:
            A pandas.DataFrame containing the filtered images.
        """
        imgs = self.joins.imgs_anns_cats  # .set_index('image_id', drop=False)
        if cfilter is None:
            # TODO: check that at least one of the input is not None
            cfilter = composition(cfilters.imgs_filter(img_ids, img_nms, composition),
                                  cfilters.cats_filter(cat_ids, cat_nms, supercat_nms, composition))
        if invert:
            cfilter = NotFilter(cfilter)
        imgs = cfilter.apply(imgs)
        return self.joins.extract_imgs(imgs).dropna()

    def filtered_anns(
            self,
            cfilter: Optional[Filter] = None,
            *,
            ann_ids: Optional[IDXSelector[int]] = None,
            img_ids: Optional[IDXSelector[int]] = None,
            img_nms: Optional[IDXSelector[str]] = None,
            cat_ids: Optional[IDXSelector[int]] = None,
            cat_nms: Optional[IDXSelector[str]] = None,
            supercat_nms: Optional[IDXSelector[str]] = None,
            area_rng: Optional[Tuple[float, float]] = None,
            is_crowd: Optional[bool] = None,
            composition: Type[ComposeFilter] = AndFilter,
            invert: bool = False
    ) -> pd.DataFrame:
        """
        Get dataset's annotations after join with categories and images, and potentially after a filtering.

        Args:
            cfilter: a custom Filter for the COCOHelper.
            ann_ids: a filter for the annotation ids.
            img_ids: a filter for the image ids.
            img_nms: a filter for the image file names.
            cat_ids: a filter for the category ids.
            cat_nms: a filter for the category names.
            supercat_nms: a filter for the super-category names.
            area_rng: a filter for the annotation area.
            is_crowd: a filter for the annotation being a crowd or not
                ("is_crowd" in the annotation of the COCO JSON file).
            composition: a composition type for the filters (defaults to "and"
                behavior between each filter).
            invert: if True, invert the way the filter works.

        Returns:
            A pandas.DataFrame containing the filtered annotations.
        """
        anns = self.joins.anns_cats_imgs
        if cfilter is None:
            # TODO: check that at least one of the input is not None
            cfilter = composition(cfilters.imgs_filter(img_ids, img_nms, composition),
                                  cfilters.cats_filter(cat_ids, cat_nms, supercat_nms, composition),
                                  cfilters.anns_filter(ann_ids, area_rng, is_crowd, composition))
        if invert:
            cfilter = NotFilter(cfilter)
        anns = cfilter.apply(anns)
        return self.joins.extract_anns(anns).dropna()

    #
    # # # # # # # # # # # # # # # #
    # DUPLICATE DROPPING METHODS  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def drop_duplicate_cats(self):
        """Drop duplicate categories (same values with different index)."""
        cats, id_mapping = drop_duplicate_rows(self.cats)
        anns = fix_fk_after_drop_duplicate(self.anns.copy(), 'category_id', id_mapping)
        return self.copy(cat_df=cats, ann_df=anns)

    def drop_duplicate_imgs(self):
        """Drop duplicate images (same values with different index)."""
        imgs, id_mapping = drop_duplicate_rows(self.imgs)
        anns = fix_fk_after_drop_duplicate(self.anns.copy(), 'image_id', id_mapping)
        return self.copy(img_df=imgs, ann_df=anns)

    def drop_duplicate_anns(self):
        """Drop duplicate annotations (same values with different index)."""
        anns, id_mapping = drop_duplicate_rows(self.anns.copy(), ignore_columns=['bbox', 'segmentation'])
        return self.copy(ann_df=anns)

    def drop_duplicate_licenses(self):
        """Drop duplicate licenses (same values with different index)."""
        lic_df, id_mapping = drop_duplicate_rows(self.licenses)

        # license_id is an optional field, if the column is there it will be fixed.
        if 'license_id' in self.imgs.columns:
            img_df = fix_fk_after_drop_duplicate(self.imgs.copy(), 'license_id', id_mapping)
            return self.copy(lic_df=lic_df, img_df=img_df)

        return self.copy(lic_df=lic_df)

    def merge(
            self,
            *coco_helper: COCOHelper,
            drop_duplicates: bool = True
    ) -> COCOHelper:
        """
        Merge different COCO datasets with all categories, images, annotations and licenses merged.

        Args:
            *coco_helper: coco dataset(s) to merge with this coco dataset.
            drop_duplicates: if True, merge duplicate rows dropping redundant.

        Returns:
            A COCOHelper resulting from merging multiple datasets.
        """
        from cocohelper.merge import merge_coco
        all_coco = list(coco_helper) + [self]
        return merge_coco(*all_coco, drop_duplicates=drop_duplicates)

    #
    # # # # # # # # #
    # IMAGE LOADER  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def get_img(
            self,
            img_id: int
    ) -> np.ndarray:
        """
        Load the image with img_id as a numpy array.

        Args:
            img_id: The id of the image to load.

        Returns:
            A numpy array with shape (H, W, C).
        """
        try:
            image_file_name = self.filtered_imgs().loc[img_id, 'file_name']
        except KeyError:
            raise COCOImageNotFoundError(img_id)

        image_path = self.root_path / self.paths.img_dir / image_file_name
        with Image.open(image_path) as img:
            image_array = np.array(img)
        return image_array

    #
    # # # # # # # # # #
    # SAMPLES LOADERS #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO: transform this to an iterator
    def get_ann_sample(
            self,
            ann_id: Optional[int] = None,
            idx: Optional[int] = None,
            transform: Optional[Transform] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Load a single annotation with the corresponding image.

        Args:
            ann_id: The id of annotation to load, partially optional (if not
                provided, idx must be provided).
            idx: The index of annotation to load, partially optional (if not
                provided, ann_id must be provided).
            transform: An optional Transform to modify the image and annotation.

        Returns:
            The image as a numpy array and the annotation infos as a dict.
        """
        assert (ann_id is not None) ^ (idx is not None), \
            "One and only one of the two input parameters `ann_id`  and `idx`  must be provided (not None)"

        if ann_id is None:
            ann_id = self.anns.index[idx]
        try:
            ann_cat = self.joins.anns_cats.loc[ann_id]
        except KeyError:
            raise COCOAnnotationNotFoundError(ann_id)

        img_data = self.get_img(ann_cat.image_id)
        anns = ann_cat.to_dict()
        if transform is not None:
            img_data, anns = transform.apply(img_data, [anns])
            anns = anns[0]
        return img_data, anns

    def get_img_sample(
            self,
            img_id: Optional[int] = None,
            idx: Optional[int] = None,
            transform: Optional[Transform] = None
    ) -> Tuple[dict, list]:
        """
        Load an image with infos and annotations.

        Args:
            img_id: The id of the image to load, partially optional (if not
                provided, idx must be provided).
            idx: The index of the image to load, partially optional (if not
                provided, img_id must be provided).
            transform: An optional Transform to modify the image and annotations.

        Returns:
            A dictionary with image infos and data, and a list of annotations.
        """
        assert (img_id is not None) ^ (idx is not None), \
            "One and only one of the two input parameters `img_id`  and `idx`  must be provided (not None)"

        if img_id is None:
            img_id = self.imgs.index[idx]

        try:
            img = self.imgs.loc[img_id]
        except KeyError:
            raise COCOImageNotFoundError(img_id)

        img_data = self.get_img(img_id)
        anns = self.filtered_anns(cfilters.imgs_filter(ids=img_id))
        ann_cats = self.joins.anns_cats  # .set_index('annotation_id')
        ann_cats = ann_cats.loc[anns.index]
        ann_cats = ann_cats.to_dict(orient='records')
        if transform is not None:
            img_data, ann_cats = transform.apply(img_data, ann_cats)
        img = img.to_dict()
        img['image'] = img_data
        return img, ann_cats

    @staticmethod
    def new_info_dict() -> dict:
        """Get a generic info dict for COCO format."""
        return {
            "contributor": "COCOHelpers",
            "date_created": dt.datetime.now().astimezone(dt.timezone.utc).strftime("%Y-%b-%d, %I:%M:%S (%Z)"),
            "url": "",
            "version": "1.0",
            "year": int(dt.datetime.now().astimezone(dt.timezone.utc).strftime("%Y"))
        }
