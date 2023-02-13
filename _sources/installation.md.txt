# Installation

We didn't publish the wheel file to the public _pypiserver_ yet, so you can 
install COCOHelper package from the wheel file provided in one of the releases
provided in the 
[releases page](https://github.com/AILAB-bh/cocohelper/releases).

You can use pip to install the wheel, e.g.:
```shell
$ pip install cocohelper-0.3.3-py3-none-any.whl
```

### Installation inside AILAB-BH
If you are part of AILAB-BH you can install COCOHelper from the 
_ailab-pypi-server_ in the local network as every other package.

```shell
$ pip install cocohelper --extra-index-url  http://10.79.85.55:28080/simple --trusted-host 10.79.85.55
```

