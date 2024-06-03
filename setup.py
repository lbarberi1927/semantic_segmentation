# from setuptools import setup, find_packages
# 
# setup(
#     name="san",
#     version="0.1",
#     packages=find_packages(),
#     include_package_data=True,
#     install_requires=[
#         "flask",
#         "werkzeug",
#         "gradio",
#     ],
#     entry_points={
#         "console_scripts": [
#             "san_server=scripts.san_server:main",
#         ],
#     },
# )
# 
from setuptools import setup, find_packages

setup(
    name="san",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "werkzeug",
        "gradio",
        "cython",
        "scipy",
        "shapely",
        "timm==0.9.10",
        "h5py",
        "submitit",
        "scikit-image",
        "wandb",
        "setuptools",
        "numpy==1.22.4",
        "Pillow==9.3.0",
        "pycocotools~=2.0.4",
        "fvcore",
        "tabulate",
        "tqdm",
        "ftfy",
        "regex",
        "opencv-python",
        "open_clip_torch==2.16.0",
        # Dependencies from install.sh (except those requiring special handling)
        # "torch==1.12.1+cu113",  # This should be handled separately due to the CUDA-specific version
        # "torchvision==0.13.1+cu113",  # Same as above
        # "torchaudio==0.12.1",  # This might also require handling depending on the system
        # 'git+https://github.com/facebookresearch/detectron2.git@v0.6', # This requires manual installation or a different approach
    ],
    entry_points={
        "console_scripts": [
            "san_server=scripts.san_server:main",
        ],
    },
    # Optionally, if you have any package data to include
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
        # And include any files found in the "data" subdirectory of the "mypkg" package, also:
        "mypkg": ["data/*"],
    },
)
