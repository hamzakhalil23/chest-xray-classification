# Dataset setup

Dataset: [COVID19+PNEUMONIA+NORMAL Chest X-Ray Images](https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images).

Download from the publisher and review its terms and source attribution. Images are not included in this repository.

Arrange images in `data/COVID/`, `data/NORMAL/` and `data/PNEUMONIA/`, then set the notebook DATA_DIR to that location. In Colab, mount your own Google Drive and update the path.

Recorded counts are historical outputs: 5,228 images. Before rerunning, check readable files, duplicates, class mapping and patient/source overlap, and preserve the split manifest. Do not publish sensitive images or identifiers.
