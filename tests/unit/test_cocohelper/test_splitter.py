from cocohelper import COCOHelper
from cocohelper.splitters.proportional import ProportionalDataSplitter
from cocohelper.splitters.kfold import KFoldSplitter
from cocohelper.splitters.stratified import StratifiedDataSplitter
import pytest


@pytest.fixture()
def ch():
    return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')


@pytest.fixture()
def ch_stratified():
    return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco_stratified.json')


def test_input_list_split(ch):
    """ Feeding a series of numbers or a list of numbers should be equivalent"""
    splitter = ProportionalDataSplitter(1, 2, 1)
    result1 = splitter.apply(ch)
    splitter = ProportionalDataSplitter(1, 2, 1)
    result2 = splitter.apply(ch)
    for sset in zip(result1, result2):
        assert len(sset[0].imgs) == len(sset[1].imgs)


def test_int_split(ch):
    splitter = ProportionalDataSplitter(70, 30)
    train, test = splitter.apply(ch)

    tot_len = len(ch.imgs)
    test_len = int(tot_len * 0.3)
    train_len = tot_len - test_len

    assert len(train.imgs) == train_len
    assert len(test.imgs) == test_len


def test_float_split(ch):
    splitter = ProportionalDataSplitter(0.7, 0.3)
    train, test = splitter.apply(ch)

    tot_len = len(ch.imgs)
    test_len = int(tot_len * 0.3)
    train_len = tot_len - test_len

    assert len(train.imgs) == train_len
    assert len(test.imgs) == test_len


def test_args_split(ch):
    splitter = ProportionalDataSplitter(0.7, 0.3)
    train, test = splitter.apply(ch)

    tot_len = len(ch.imgs)
    test_len = int(tot_len * 0.3)
    train_len = tot_len - test_len

    assert len(train.imgs) == train_len
    assert len(test.imgs) == test_len


def test_stratified_split(ch_stratified):
    splitter = StratifiedDataSplitter(0.75, 0.25)
    splits = splitter.apply(ch_stratified)

    s0_props = splits[0].joins.imgs_anns['category_id'].value_counts()
    s1_props = splits[1].joins.imgs_anns['category_id'].value_counts()

    assert s0_props.equals(s1_props)
    assert s0_props.equals(s1_props)


def test_kfold_split(ch):
    splitter = KFoldSplitter(n_fold=7)
    splits = splitter.apply(ch)
    assert len(splits) == 7
    for split in splits:
        assert len(split.imgs) == 2


def test_stratified_kfold_split(ch_stratified):
    splitter = KFoldSplitter(n_fold=3, stratified=True)
    splits = splitter.apply(ch)
    assert len(splits) == 3
    for split in splits:
        assert len(split.imgs) > 0


def test_kfold_iter(ch):
    splitter = KFoldSplitter(n_fold=7)
    i = 0
    for train, val in splitter.iter(ch):
        assert len(train.imgs) == 12
        assert len(val.imgs) == 2
        i += 1
    assert i == 7
