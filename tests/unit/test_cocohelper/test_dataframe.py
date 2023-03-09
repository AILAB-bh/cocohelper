from cocohelper import COCOHelper


ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')


def test_to_dict_overriding():
    assert 'image_id' not in ch.imgs.to_dict().keys()
    assert 'image_id' in ch.imgs.reset_index().to_dict(include_index=True).keys()

    assert 'annotation_id' not in ch.anns.to_dict().keys()
    assert 'annotation_id' in ch.anns.to_dict(include_index=True).keys()
