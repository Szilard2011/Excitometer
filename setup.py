from setuptools import setup, find_packages

setup(
    name='excitemeter',
    version='0.1.0',
    description='A model to predict excitement levels from audio input.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Szilárd',
    author_email='szilard.palnagy@gmail.com',
    url='https://github.com/Szilard2011/Excitometer',  
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.12.0',
        'datasets>=1.14.0',
        'scikit-learn>=0.24.2',
        'numpy>=1.19.2',
        'pandas>=1.1.3',
        'tqdm>=4.62.0',
        'soundfile>=0.10.3',
        'librosa>=0.8.1',
        'matplotlib>=3.3.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
