from pathlib import Path
from cocohelper import utils


# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.

def test_subtract_path():
    assert utils.path.subtract('/hello/mad/world', './mad/world') == Path('/hello')
    assert utils.path.subtract('/hello/mad/world', './mad/world/') == Path('/hello')
    assert utils.path.subtract('/hello/mad/world/', './mad/world') == Path('/hello')
    assert utils.path.subtract('/hello/mad/world/', './mad/world/') == Path('/hello')
    assert utils.path.subtract('hello/mad/world/', './mad/world') == Path('hello')
    assert utils.path.subtract('hello/mad/world', './mad/world') == Path('hello')
    assert utils.path.subtract('hello/mad/world', './mad/world/') == Path('hello')
    assert utils.path.subtract('hello/mad/world', 'mad/world/') == Path('hello')
    assert utils.path.subtract('hello/mad/world', 'mad/world') == Path('hello')

    assert utils.path.subtract('hello/mad/world/', './mad/worldX') is None
    assert utils.path.subtract('hello/mad/world/', './mad/Xworld') is None
    assert utils.path.subtract('hello/mad/world/', './madX/world') is None
    assert utils.path.subtract('hello/mad/world/', 'X/mad/world') is None
