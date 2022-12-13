# How to contribute  

If you want to submit a contribution, please follow the steps below:

> #### Step 1. Open an issue:
> Before you make any changes, we recommend opening an issue if one doesn't 
> already exist. In the issue, you can discuss your proposed changes, and we can
> give you feedback and validate them.
> If your code change involves fixing a bug, please include a script or a 
> notebook that shows how to reproduce the wrong behavior.
> If you only want to introduce minor changes, such as documentation fixes,
> then feel free to open a PR without discussion.  
>
> #### Step 2. Make code changes:
> To make code changes, you need to fork the repository.
> You will need to set up a development environment and run the unit tests
> (see the "Setup environment" section for more details).  
> 
> #### Step 3. Create a pull request:
> Once the change is ready, open a pull request from your branch in your fork 
> to the master branch in [https://github.com/AILAB-bh/cocohelper](https://github.com/AILAB-bh/cocohelper).  
> 
> #### Step 4. Sign the Contributor License Agreement:
> After creating the pull request, you will need to sign the Contributor License Agreement (CLA). 
> The agreement can be found at [TODO](https://TODO).  
> 
> #### Step 5. Code review:
> Continuous integration tests will automatically directly run on your pull 
> request, sharing a final report via GitHub actions.  
> 
> #### Step 6. Merging:
> Once the pull request is approved, a team member will take care of merging.

## Setup environment:
Setting up your development environment requires you to fork and clone the COCO
Helper repository, create a virtual environment having poetry package installed,
install the dependencies, and execute the setup file.

After cloning the repo you can install our default conda environment with 
_mamba_ (faster) or _conda_:

```shell  
conda env create -f environment.yml
```

An environment named `cocohelper` will be created; now you have to activate the
environment and install the project dependencies (both dev and package dependencies).

This project manage the dependencies with `poetry`, so you have to run:

```shell
conda activate cocohelper
poetry install
```

> NB: if you don't want to use _conda_, you can use another virtual environment,
> but you should manually install poetry with version 1.2 or higher in it
> to be able to run `poetry install` correctly.


## Code Style
We are trying to adhere as much as possible to the [Google styleguide for python](https://google.github.io/styleguide/pyguide.html). 

Considerations:
- **Style in our codebase is far from perfection at the moment** - 
  every improvement of existing code going in this direction is welcome.
- **New code should adhere as much as possible to Google styleguide** -
  future pull requests will also be evaluated for code style.


## Clean Code and Software Design
We are trying to create a clean-codebase, following SOLID and other software
engineering principles. We are far from perfection, but we are constantly trying
to improve.

##### Considerations:
- **Design in our codebase is not perfect** - 
  we are trying to improve, PRs going in this direction are welcome.
- **New code should adhere as much as possible to good software design principles** -
  future pull requests will also be evaluated for design.


##### Useful sources:
- [Robert Martin](https://en.wikipedia.org/wiki/Robert_C._Martin) Clean Code 
  book (here a good reference for the [main clean code principles](https://gist.github.com/wojteklu/73c6914cc446146b8b533c0988cf8d29)).
- [A software design principles wiki](http://principles-wiki.net/start).
- [SOLID principles](https://en.wikipedia.org/wiki/SOLID) (wikipedia)



## Tests
You can run the test suite stored in `./tests` using:
```shell
pytest tests/unit
```

> **A minimum requirement for a pull request to be acceptable is that all tests 
> in the test suite must pass.**

### Enlarge the test suite
If new source files is added (or in general new code), you should add new
test cases in the test suite following the **AAA** principle:
1. **Arrange** - create the system under test.
2. **Act** - minimum code represent the functionality of the system under test 
    that we want to test, hopefully, it will be a single instruction (a single function call). 
3. **Assert** - check that the functionality under test behave as expected.

We are using pytest, so for the _Arrange_ step we strongly encourage to 
rely on fixtures.


##  Type Safety
We currently use **MyPy** to have a type-safe library even if it's written
with a weakly typed language as python.

> **A minimum requirement for a pull request to be acceptable is that MyPy 
> code analysis must pass.**

To run a check with MyPy use:
```shell
mypy src
```


## Documentation
All the code should be documented using the google-style docstrings.
Make sure you also read the [google docstring guidelines](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

The documentation will be generated with sphinx, currently using a wrapper
script to manage an internal custom configuration in AILAB 
(**ailab-apigen** package).

> **A minimum requirement for a pull request to be acceptable is that new
> code should be documented and sphinx/ailab-apigen should be able to generate
> the documentation website.**
