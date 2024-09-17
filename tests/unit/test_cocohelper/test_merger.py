from typing import List
import pandas as pd
import pytest

from cocohelper import COCOHelper


# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.

@pytest.fixture()
def ch1():
    return COCOHelper.load_json('tests/data/coco_merge1/annotations/coco.json')

@pytest.fixture()
def ch2():
    return COCOHelper.load_json('tests/data/coco_merge2/annotations/coco.json')


# TEST MERGE WITHOUT DROPPING DUPLICATES
@pytest.fixture
def ch_merged_with_duplicates(ch1, ch2):
    return ch1.merge(ch2, drop_duplicates=False)

def test_merge_with_duplicates(ch_merged_with_duplicates):
    return
    # todo:
    #  validator = COCOValidator(ch_merged_with_duplicates)
    #  assert validator.validate_dataset({})

def test_merged_with_duplicates_nb_annotations(ch_merged_with_duplicates, ch1, ch2):
    assert len(ch_merged_with_duplicates.anns) == len(ch1.anns) + len(ch2.anns)

def test_merged_with_duplicates_nb_images(ch_merged_with_duplicates, ch1, ch2):
    assert len(ch_merged_with_duplicates.imgs) == len(ch1.imgs) + len(ch2.imgs)

def test_merged_with_duplicates_nb_categories(ch_merged_with_duplicates, ch1, ch2):
    assert len(ch_merged_with_duplicates.cats) == len(ch1.cats) + len(ch2.cats)

def test_merged_with_duplicates_nb_licenses(ch_merged_with_duplicates, ch1, ch2):
    assert len(ch_merged_with_duplicates.licenses) == len(ch1.licenses) + len(ch2.licenses)


# TEST MERGE DROPPING DUPLICATES

@pytest.fixture
def ch_merged(ch1, ch2):
    return ch1.merge(ch2, drop_duplicates=True)

def test_merge_no_dupl(ch_merged):
    return
    # todo:
    #  validator = COCOValidator(ch_merged)
    #  assert validator.validate_dataset({})


def test_merged_no_dupl_nb_annotations(ch_merged, ch1, ch2):
    anns1 = set(__records_footprint(__df2records(ch1.anns)))
    anns2 = set(__records_footprint(__df2records(ch2.anns)))
    assert len(ch_merged.anns) == len(set.union(anns1, anns2))

def test_merged_no_dupl_nb_images(ch_merged, ch1, ch2):
    imgs1 = set(__records_footprint(__df2records(ch1.imgs)))
    imgs2 = set(__records_footprint(__df2records(ch2.imgs)))
    assert len(ch_merged.imgs) == len(set.union(imgs1, imgs2))

def test_merged_no_dupl_nb_categories(ch_merged, ch1, ch2):
    cats1 = set(__records_footprint(__df2records(ch1.cats)))
    cats2 = set(__records_footprint(__df2records(ch2.cats)))
    assert len(ch_merged.cats) == len(set.union(cats1, cats2))

def test_merged_no_dupl_nb_licenses(ch_merged, ch1, ch2):
    licenses1 = set(__records_footprint(__df2records(ch1.licenses)))
    licenses2 = set(__records_footprint(__df2records(ch2.licenses)))
    assert len(ch_merged.licenses) == len(set.union(licenses1, licenses2))



def test_double_merge(ch_merged, ch1, ch2):
    ch_merged = ch_merged.merge(ch2)
    imgs1 = set(__records_footprint(__df2records(ch1.imgs)))
    imgs2 = set(__records_footprint(__df2records(ch2.imgs)))
    assert len(ch_merged.imgs) == len(set.union(imgs1, imgs2))

def test_double_merge_interlaced(ch_merged, ch1, ch2):
    ch_merged = ch_merged.merge(ch1)
    imgs1 = set(__records_footprint(__df2records(ch1.imgs)))
    imgs2 = set(__records_footprint(__df2records(ch2.imgs)))
    assert len(ch_merged.imgs) == len(set.union(imgs1, imgs2))


def test_merged_dataset_info_length(ch1, ch2):
    ch_merged = ch1.merge(ch2)
    assert len(ch_merged.info['merged_infos']) == 2


def test_self_merge(ch1):
    self_merged = ch1.merge(ch1)
    # todo:
    #  validator = COCOValidator(ch_merged)
    #  assert validator.validate_dataset({})

def test_self_merge_nb_categories(ch1):
    self_merged = ch1.merge(ch1)
    # If two category are identical, we should merge on a single category
    assert len(self_merged.cats) == len(ch1.cats)

def test_self_merge_nb_imgs(ch1):
    self_merged = ch1.merge(ch1)
    # If two the images are identical, we should merge on a single image
    assert len(self_merged.imgs) == len(ch1.imgs)

def test_self_merge_nb_anns(ch1):
    self_merged = ch1.merge(ch1)
    # If two annotations are identical, we should merge on a single ann.
    assert len(self_merged.anns) == len(ch1.anns)

def test_self_merge_nb_licenses(ch1):
    self_merged = ch1.merge(ch1)
    # If two licenses are identical, we should merge on a single one.
    assert len(self_merged.licenses) == len(ch1.licenses)

def test_self_merge_nb_infos(ch1):
    self_merged = ch1.merge(ch1)
    assert len(self_merged.info['merged_infos']) == 2

def __df2records(df: pd.DataFrame):
    return df.reset_index(drop=True).to_dict(orient='records')


def __records_footprint(records: List[dict]) -> List[str]:
    return [str(d.values()) for d in records]

