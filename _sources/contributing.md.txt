# How to contribute  

Here is a list of items where we would appreciate your contribution: [link](https://github.com/[TODO]/cocohelper/issues?page=2&q=is%3Aissue+is%3Aopen+label%3Acontribution-welcome).  

For contributions, we follow a process similar to that of [keras-cv](https://github.com/keras-team/keras-cv/blob/master/.github/CONTRIBUTING.md).  
If you want to submit a contribution, please follow the steps below:

> #### Step 1. Open an issue:
> Before you make any changes, we recommend opening an issue if one doesn't already exist. In the issue, you can discuss your proposed changes, and we can give you feedback and validate them.
> If your code change involves fixing a bug, please include a notebook that shows how to reproduce the wrong behavior. You can use [Colab](https://colab.research.google.com/) for this purpose.
> If you only want to introduce minor changes, such as documentation fixes, then feel free to open a PR without discussion.  
> 
> #### Step 2. Make code changes:
> To make code changes, you need to fork the repository. You will need to set up a development environment and run the unit tests (see the "Setup environment" section for more details).  
> 
> #### Step 3. Create a pull request:
> Once the change is ready, open a pull request from your branch in your fork to the master branch in [https://TODO](https://TODO).  
> 
> #### Step 4. Sign the Contributor License Agreement:
> After creating the pull request, you will need to sign the Contributor License Agreement (CLA). 
> The agreement can be found at [https://TODO](https://TODO).  
> 
> #### Step 5. Code review:
> Continuous integration tests will automatically directly run on your pull request, sharing a final report via GitHub actions.  
> 
> #### Step 6. Merging:
> Once the pull request is approved, a team member will take care of merging.

## Setup environment:
Setting up your development environment requires you to fork the COCO Helper repository, 
clone the repository, install the dependencies, and execute the setup file.
You can do it as:  

```shell  
gh repo fork [TODO]/cocohelper --clone --remote
cd cocohelper
pip install ".[tests]"
python setup.py develop
```  

The first line needs an installation of [the GitHub CLI](https://github.com/cli/cli).
After running these commands, you should be able to run the unit tests using `pytest coco_helper`.  
Please report any issues running tests following these steps.

Note that this will _not_ install custom ops. If you'd like to install custom ops from source, you can compile the binaries and add them to your local environment manually (requires Bazel):  

```shell  
python build_deps/configure.py  
bazel build coco_helper/custom_ops:all
mv bazel-bin/coco_helper/custom_ops/*.so coco_helper/custom_ops
```

## Run tests
KerasCV is tested using [PyTest](https://docs.pytest.org/en/6.2.x/).  

#### Run a test file:
To run a test file, run `pytest path/to/file` from the root directory of keras\_cv.  

#### Run a single test case:
To run a single test, you can use `-k=<your_regex>`  
to use regular expression to match the test you want to run. For example, you  
can use the following command to run all the tests in `cut_mix_test.py`,  
whose names contain `label`,
```  
pytest coco_helper/layers/preprocessing/cut_mix_test.py -k="label"  
```  

#### Run all tests:
You can run the unit tests for KerasCV by running:  
```  
pytest coco_helper/  
```  

#### Tests that require custom ops:
For tests that require custom ops, you'll have to compile the custom ops and make them available to your local Python code:  
```shell  
python build_deps/configure.py
bazel build coco_helper/custom_ops:all
cp bazel-bin/coco_helper/custom_ops/*.so coco_helper/custom_ops/
```

Tests which use custom ops are disabled by default, but can be run by setting the environment variable `TEST_CUSTOM_OPS=true`.  

## Code Style
Your code must adhere to the Python [styleguide](https://google.github.io/styleguide/pyguide.html) provided by Google. 
Make sure you also read the docstring [guidelines](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
