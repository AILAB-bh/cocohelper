from setuptools import setup, find_packages

setup(
    name='cocohelper',
    version='1.0.0',
    packages=find_packages(),
    description='Object Oriented library to manage MS-COCO-like datasets efficiently.',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    author='Riccardo Del Chiaro, Gabriele Valvano, Elia Lotti, Andrea Panizza',
    author_email='gabriele.valvano@bakerhughes.com',
    url='https://github.com/AILAB-bh/coco-helper',
    license='Apache License, Version 2.0',
    keywords=["coco", "mscoco", "dataset"],
    install_requires=[
            'numpy>=1.24.1',
            'opencv-python>=4.6.0.66',
            'pandas>=1.5.3',
            'pillow>=9.3.0',
            'pycocotools>=2.0.0',
            'scikit-learn>=1.1.3',
            'scipy>=1.4.0',
            'shapely>=2.0.6',
            'tqdm>=4.66.4',
        ],
    python_requires='>=3.8',
)
