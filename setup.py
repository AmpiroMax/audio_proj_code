from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1.0",
    packages=find_packages(),
    # install_requires=[
    #     "torch",
    #     "torchaudio",
    #     "einops",
    #     "clip",
    #     "Pillow",
    #     "huggingface_hub",
    #     "pytorch_lightning",
    # ],
    # entry_points={
    #     'console_scripts': [
    #         'train_image_cond=src.train.train_image_cond:main',
    #     ],
    # },
)
