# Usage

## Tutorials
Have a look at our tutorials in the Jupyter notebooks available [here](notebooks/). 
If you'd like to contribute with new tutorials, look at our [contributing guide](../doc/src/contributing.md).

## Examples

#### Load a coco dataset

```python
from cocohelper import COCOHelper


ch = COCOHelper.load_dict('path/to/coco.json')
```


#### Visualize images
```python
from cocohelper.visualizer import COCOVisualizer

COCOVisualizer(ch).visualize(img_id=1, show_bbox=True, show_segmentation=True)
```

#### Split train/val/test
```python
from cocohelper.splitters.proportional import ProportionalDataSplitter

splitter = ProportionalDataSplitter(70, 20, 10)
ch_train, ch_val, ch_test = splitter.apply(ch)
```
