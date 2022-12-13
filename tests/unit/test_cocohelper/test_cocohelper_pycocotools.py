from cocohelper import COCOHelper, filters
from cocohelper.filters import AND
from cocohelper.filters.cocofilters import cocofilters
from cocohelper.filters.strategies import HAVING_VALUE, ANY_VALUE, ALL_VALUES

# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.

from cocohelper.utils.dataframe import df_to_records


ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')
coco = ch.to_coco()


def test_cocohelper_pycocotools():
    do_cats_test()
    do_cats_test(cat_ids=[0, 1])
    do_cats_test(cat_nms='balloon')
    do_cats_test(cat_nms=['balloon', 'super_balloon'])
    do_cats_test(supercat_nms='class')

    do_imgs_test()
    do_imgs_test(img_ids=[0, 1, 2, 5, 8])
    do_imgs_test(cat_ids=[0, 1])
    do_imgs_test(cat_ids=[0])
    do_imgs_test(cat_nms=['balloon'])

    # Supercategory is not well managed in pycocotools
    # test_imgs(supercat_nms='class')

    # This does not exist in pycocotools: can't find img ids from img file name.
    # test_imgs(img_nms=["24631331976_defa3bb61f_k.jpg", "3825919971_93fb1ec581_b.jpg"])

    # test_anns() # does not work as expected in pycocotools (retrieve an empty set, not all annotations like in other getters)

    do_anns_test(ann_ids=[0, 1])
    do_anns_test(ann_ids=[0])
    do_anns_test(ann_ids=[0, 10])
    do_anns_test(img_ids=[10, 13])
    do_anns_test(cat_ids=[1, 2])
    do_anns_test(cat_ids=[1, 2], area_rng=[0, 100000])
    do_anns_test(cat_ids=[1, 2], area_rng=[0, 50000])
    do_anns_test(cat_ids=[1, 2], area_rng=[0, 10000])
    do_anns_test(cat_ids=[1, 2], area_rng=[0, 1000])
    do_anns_test(cat_nms=['balloon'])

    # do_anns_test(area_rng=[0, 10000])


def test_cocohelper_filter_pycocotools():
    do_cats_filter_test()
    do_cats_filter_test(cat_ids=[0, 1])
    do_cats_filter_test(cat_nms='balloon')
    do_cats_filter_test(cat_nms=['balloon', 'super_balloon'])
    do_cats_filter_test(supercat_nms='class')

    do_imgs_filter_test()
    do_imgs_filter_test(img_ids=[0, 1, 2, 5, 8])
    do_imgs_filter_test(cat_ids=[0, 1])
    do_imgs_filter_test(cat_ids=[0])
    do_imgs_filter_test(cat_nms=['balloon'])

    # Supercategory is not well managed in pycocotools
    # do_imgs_filter_test(supercat_nms='class')

    # This does not exist in pycocotools: can't find img ids from img file name.
    # do_imgs_filter_test(img_nms=["24631331976_defa3bb61f_k.jpg", "3825919971_93fb1ec581_b.jpg"])

    # do_anns_filter_test() # does not work as expected in pycocotools (retrieve an empty set, not all annotations like in other getters)

    do_anns_filter_test(ann_ids=[0, 1])
    do_anns_filter_test(ann_ids=[0])
    do_anns_filter_test(ann_ids=[0, 10])
    do_anns_filter_test(img_ids=[10, 13])
    do_anns_filter_test(cat_ids=[1, 2])
    do_anns_filter_test(cat_ids=[1, 2], area_rng=[0, 10000])
    do_anns_filter_test(cat_ids=[1, 2], area_rng=[0, 1000])
    do_anns_filter_test(cat_nms=['balloon'])

    # do_anns_filter_test(area_rng=[0, 10000])


def do_cats_test(cat_ids=None, cat_nms=None, supercat_nms=None):
    helper_cats = ch.filtered_cats(cat_ids=cat_ids, cat_nms=cat_nms, supercat_nms=supercat_nms)

    catIds = cat_ids if cat_ids is not None else tuple()
    catNms = cat_nms if cat_nms is not None else tuple()
    supNms = supercat_nms if supercat_nms is not None else tuple()
    pycoco_cats = coco.loadCats(coco.getCatIds(catNms, supNms, catIds))

    assert df_to_records(helper_cats, ch._colmaps.cat) == pycoco_cats


def do_cats_filter_test(cat_ids=None, cat_nms=None, supercat_nms=None):
    helper = ch.filter_cats(cat_ids=cat_ids, cat_nms=cat_nms, supercat_nms=supercat_nms)

    catIds = cat_ids if cat_ids is not None else tuple()
    catNms = cat_nms if cat_nms is not None else tuple()
    supNms = supercat_nms if supercat_nms is not None else tuple()
    pycoco_cats = coco.loadCats(coco.getCatIds(catNms, supNms, catIds))

    assert df_to_records(helper.cats, ch._colmaps.cat) == pycoco_cats


def do_imgs_test(img_ids=None, cat_ids=None, cat_nms=None):
    imgs_filter = cocofilters.imgs_filter(ids=img_ids)
    cats_filter = cocofilters.cats_filter(ids=cat_ids, nms=cat_nms, strategy=ALL_VALUES)
    filtered_imgs = ch.filtered_imgs(cfilter=AND(imgs_filter, cats_filter))

    imgIds = img_ids if img_ids is not None else tuple()
    catIds = cat_ids if cat_ids is not None else tuple()
    catNms = cat_nms if cat_nms is not None else tuple()
    if cat_ids is not None or cat_nms is not None:
        catIds = coco.getCatIds(catNms, [], catIds)
    pycoco_imgs = coco.loadImgs(coco.getImgIds(imgIds, catIds))

    cocoh_pycoco = ch.copy(img_df=filtered_imgs).to_coco()
    cocoh_pycoco_imgs = cocoh_pycoco.loadImgs(cocoh_pycoco.getImgIds())
    assert cocoh_pycoco_imgs == pycoco_imgs


def do_imgs_filter_test(img_ids=None, cat_ids=None, cat_nms=None):
    #     helper = ch.filter_imgs(img_ids=img_ids, cat_ids=cat_ids, cat_nms=cat_nms)
    imgs_filter = cocofilters.imgs_filter(ids=img_ids)
    cats_filter = cocofilters.cats_filter(ids=cat_ids, nms=cat_nms, strategy=ALL_VALUES)
    helper = ch.filter_imgs(AND(imgs_filter, cats_filter))
    from cocohelper.filters.cocofilters import anns_filter


    imgIds = img_ids if img_ids is not None else tuple()
    catIds = cat_ids if cat_ids is not None else tuple()
    catNms = cat_nms if cat_nms is not None else tuple()
    if cat_ids is not None or cat_nms is not None:
        catIds = coco.getCatIds(catNms, [], catIds)
    pycoco_imgs = coco.loadImgs(coco.getImgIds(imgIds, catIds))

    assert df_to_records(helper.imgs, ch._colmaps.img) == pycoco_imgs


def do_anns_test(ann_ids=None, img_ids=None, cat_ids=None, cat_nms=None, supercat_nms=None, area_rng=None,
                 is_crowd=None):
    ann_flag = ann_ids is not None
    img_flag = img_ids is not None
    cat_flag = cat_ids is not None or cat_nms is not None or supercat_nms is not None

    # With pycocotools COCO interface we only have two options:
    # 1. not use any ann/img/cat retrieval option
    # 2. use only exclusively one retriavl option (ann, img or cat).
    assert (ann_flag == img_flag == cat_flag == False) or (ann_flag ^ img_flag ^ cat_flag), "Invalid test"

    if (not img_flag) and (not cat_flag) and ((area_rng is not None) or (is_crowd is not None)):
        # pyccocotools does not support to filters ann_ids given area_rng and is_crowd:
        # these flags are only used to fitler ann_ids when starting from imgs or categories.
        assert False, "Invalid test"

    helper_anns = ch.filtered_anns(ann_ids=ann_ids, img_ids=img_ids, cat_ids=cat_ids, cat_nms=cat_nms,
                                   supercat_nms=supercat_nms, area_rng=area_rng, is_crowd=is_crowd)

    annIds = ann_ids if ann_ids is not None else tuple()
    areaRng = area_rng if area_rng is not None else tuple()
    imgIds = img_ids if img_ids is not None else tuple()
    catIds = cat_ids if cat_ids is not None else tuple()
    catNms = cat_nms if cat_nms is not None else tuple()
    supNms = supercat_nms if supercat_nms is not None else tuple()

    if cat_flag:
        catIds = coco.getCatIds(catNms, supNms, catIds)
        annIds = coco.getAnnIds(catIds=catIds, areaRng=areaRng, iscrowd=is_crowd)

    elif img_flag:
        imgIds = coco.getImgIds(imgIds, catIds)
        annIds = coco.getAnnIds(imgIds=imgIds, areaRng=areaRng, iscrowd=is_crowd)

    pycoco_anns = coco.loadAnns(annIds)

    assert df_to_records(helper_anns, ch._colmaps.ann) == pycoco_anns


def do_anns_filter_test(ann_ids=None, img_ids=None, cat_ids=None, cat_nms=None, supercat_nms=None, area_rng=None,
                        is_crowd=None):
    ann_flag = ann_ids is not None
    img_flag = img_ids is not None
    cat_flag = cat_ids is not None or cat_nms is not None or supercat_nms is not None

    # With pycocotools COCO interface we only have two options:
    # 1. not use any ann/img/cat retrieval option
    # 2. use only exclusively one retriavl option (ann, img or cat).
    assert (ann_flag == img_flag == cat_flag == False) or (ann_flag ^ img_flag ^ cat_flag), "Invalid test"

    if (not img_flag) and (not cat_flag) and ((area_rng is not None) or (is_crowd is not None)):
        # pyccocotools does not support to filters ann_ids given area_rng and is_crowd:
        # these flags are only used to fitler ann_ids when starting from imgs or categories.
        assert False, "Invalid test"

    helper = ch.filter_anns(ann_ids=ann_ids, img_ids=img_ids, cat_ids=cat_ids, cat_nms=cat_nms,
                            supercat_nms=supercat_nms, area_rng=area_rng, is_crowd=is_crowd)

    annIds = ann_ids if ann_ids is not None else tuple()
    areaRng = area_rng if area_rng is not None else tuple()
    imgIds = img_ids if img_ids is not None else tuple()
    catIds = cat_ids if cat_ids is not None else tuple()
    catNms = cat_nms if cat_nms is not None else tuple()
    supNms = supercat_nms if supercat_nms is not None else tuple()

    if cat_flag:
        catIds = coco.getCatIds(catNms, supNms, catIds)
        annIds = coco.getAnnIds(catIds=catIds, areaRng=areaRng, iscrowd=is_crowd)

    elif img_flag:
        imgIds = coco.getImgIds(imgIds, catIds)
        annIds = coco.getAnnIds(imgIds=imgIds, areaRng=areaRng, iscrowd=is_crowd)

    pycoco_anns = coco.loadAnns(annIds)

    assert df_to_records(helper.anns, ch._colmaps.ann) == pycoco_anns
