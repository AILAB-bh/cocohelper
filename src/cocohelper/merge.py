"""
Merge multiple COCO datasets together.
"""
from cocohelper import COCOHelper
from typing import List, Tuple
import pandas as pd


def merge_coco(
        *coco_helpers: COCOHelper,
        drop_duplicates: bool = True
) -> COCOHelper:
    """
    Merge multiple `COCOHelper` merging all categories, images and annotations.

    Args:
        *coco_helpers: the list of COCOHelper datasets to merge.
        drop_duplicates: if True, duplicate rows with different ids will be
            merged together.

    Returns:
        Merged COCOHelper.
    """
    if len(coco_helpers) < 2:
        raise ValueError("Thi minimum number of coco helpers that can be merged is 2. Please feed two or more "
                         "COCOHelper objects.")

    # TODO: manage duplicates values accordingly to parameters.

    lic_df = _merge_licenses(*coco_helpers)
    cat_df, cat_id_mapping = _merge_categories(*coco_helpers)
    img_df, image_id_mapping = _merge_images(*coco_helpers)
    ann_df = _merge_annotations(*coco_helpers, cat_id_mapping=cat_id_mapping, image_id_mapping=image_id_mapping)
    info = _merge_info(*coco_helpers)
    merged = coco_helpers[0].copy(ann_df=ann_df, img_df=img_df, cat_df=cat_df, lic_df=lic_df, info=info)

    if drop_duplicates:
        return merged.drop_duplicate_cats().drop_duplicate_imgs().drop_duplicate_anns().drop_duplicate_licenses()
    return merged


def _merge_info(
        *coco_helpers: COCOHelper
) -> dict:
    """
    Merge multiple COCOHelpers info fields.

    Args:
        *coco_helpers: the list of COCOHelper datasets to merge.

    Returns:
        A merged dictionary having information about the merging of multiple
        COCO datasets.
    """
    info = COCOHelper.new_info_dict()
    info['contributor'] += ' -- Merger'
    info['merged_infos'] = [ds.info for ds in coco_helpers]
    return info


def _merge_licenses(
        *coco_helpers: COCOHelper
) -> pd.DataFrame:
    #
    # TODO [Technical Debt]
    #  Each image could have the "license" or "license_id" field, so we should always have a
    #  null-able column in the image data frame and we should map the old license_id to the new ones.
    #  We will have to implement this later and refactor this.
    #
    # TODO: if the license is the same we should merge them (we want the union of the two sets of licenses).
    licenses = pd.concat([ds.licenses for ds in coco_helpers], ignore_index=True)
    licenses['license_id'] = licenses.index
    return licenses.set_index('license_id')


def _merge_categories(
        *coco_helpers: COCOHelper
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Merge the categories of all datasets.

    Returns:
        A tuple with 2 items: the first is a dataframe with the merged categories,
        the second is a list that contains a mapping of old ids to new ids.
        The list has as many items as coco datasets merged. For example, if you
        access the item at index 0, you get the mapping for the first dataset.
    """
    dataframes = []
    for i, ds in enumerate(coco_helpers):
        category_df = ds.cats.copy().reset_index()
        category_df['_dataset'] = i
        dataframes.append(category_df)

    concat_df = pd.concat(dataframes, ignore_index=True)
    old_cats = concat_df.copy()
    old_cats['new_id'] = -1

    concat_df['category_id'] = concat_df.index

    for _, cat in concat_df.iterrows():
        mask = (old_cats['name'] == cat['name']) & (old_cats['supercategory'] == cat['supercategory'])
        old_cats.loc[mask, 'new_id'] = cat['category_id']

    cat_id_mapping = old_cats[['_dataset', 'category_id', 'new_id']] \
        .groupby('_dataset') \
        .apply(_create_cat_mapping).to_dict()

    concat_df = concat_df.drop(columns=['_dataset'])
    return concat_df.set_index('category_id'), cat_id_mapping


def _merge_images(
        *coco_helpers: COCOHelper
) -> Tuple[pd.DataFrame, List[dict]]:
    """Merge coco helpers images fields."""
    dataframes = []
    for i, ds in enumerate(coco_helpers):
        images_df = ds.imgs.copy().reset_index()
        images_df['_dataset'] = i
        dataframes.append(images_df)

    concat_df = pd.concat(dataframes, ignore_index=True)
    concat_df['_old_id'] = concat_df['image_id']
    concat_df['image_id'] = concat_df.index

    image_id_mapping = concat_df[['_dataset', '_old_id', 'image_id']].groupby('_dataset') \
        .apply(_create_image_mapping).to_dict()

    concat_df = concat_df.drop(columns=['_dataset', '_old_id'])

    return concat_df.set_index('image_id'), image_id_mapping


def _merge_annotations(
        *coco_helpers: COCOHelper,
        cat_id_mapping: List[dict],
        image_id_mapping: List[dict]
) -> pd.DataFrame:
    """Merge coco helpers annotations fields."""
    dataframes = []
    for i, ds in enumerate(coco_helpers):
        image_mapping = image_id_mapping[i]
        cat_mapping = cat_id_mapping[i]
        anns_df = ds.anns.copy().reset_index()
        anns_df['category_id'] = anns_df['category_id'].map(cat_mapping).fillna(anns_df['category_id'])
        anns_df['image_id'] = anns_df['image_id'].map(image_mapping).fillna(anns_df['image_id'])
        dataframes.append(anns_df)

    concat_df = pd.concat(dataframes, ignore_index=True)
    concat_df['annotation_id'] = concat_df.index
    return concat_df.set_index('annotation_id')


def _create_image_mapping(x):
    """
    TODO
    """
    return pd.Series(x['image_id'].values, index=x['_old_id']).to_dict()


def _create_cat_mapping(x):
    """
    TODO
    """
    return pd.Series(x['new_id'].values, index=x['category_id']).to_dict()
