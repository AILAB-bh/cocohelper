# Installation

COCOHelper library is distributed as a wheel and can be installed with pip,
poetry or other dependency manager tools.

## Install from public PyPi server:
We still have to publish a release to the public PyPi server.

## Install from wheel:
You can simply install through pip one of the provided wheel files:

```shell
pip install cocohelper-<VERSION>-py3-none-any.whl
```
_(replace`<VERSION>` with the version you want to install)_
 
Or you can add the dependency to your poetry project if you use it:
```shell
poetry add cocohelper-<VERSION>-py3-none-any.whl
```


## Install through private AILAB PyPi server
If you are part of AILAB-BH you can install COCOHelper using our private
[AILAB-PyPi server](https://tps-innovation-dev.np-0000111.npaeuw1.bakerhughes.com/ailab-pypiserver/).

#### PIP
If you want to install COCOHelper with _pip_, simply execute:
```shell
$ pip install cocohelper --extra-index-url  https://tps-innovation-dev.np-0000111.npaeuw1.bakerhughes.com/ailab-pypiserver/simple/
```

#### Poetry
If your project use Poetry and you want to add the dependency to your project,
add the following lines to `pyproject.toml`:
```toml
[[tool.poetry.source]]
name = "ailab"
url = "https://tps-innovation-dev.np-0000111.npaeuw1.bakerhughes.com/ailab-pypiserver/simple/"
secondary = true
```

And then execute:
```shell
poetry add cocohelper
```
