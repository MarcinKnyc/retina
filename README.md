# retina

## Python version

`Python 3.12` is required.

## Data download

DRIVE: Download <https://www.kaggle.com/datasets/zionfuo/drive2004>, and save it as `retina/data/01_raw/DRIVE.zip`.

STARE: Downloaded automagically.

## Usage

`kedro run --runner=SequentialRunner`

The results are in `02_intermediate/results.png`.

Change `debug: False` to `debug: True` in `./conf/base/parameters.yml` to test if the pipeline functions correctly. It changes the amount of images sampled to 5. This reduces the runtime significantly, as the most time-intensive operation is the processing of test images.

## Todo

1. Change the datasets to those from the paper.
2. Make sure the image reading works correctly with the images from new datasets.
3. Change the machine learning model to the 3 models in the paper.
4. Tune the pipeline to achieve accuracy on par with the paper.

## Notes

- There used to be another pipeline, it was removed. No need for `kedro run -p <pipeline_name>`.
- All 3 datasets from the paper are in <https://www.kaggle.com/datasets/ipythonx/retinal-vessel-segmentation/data>.
