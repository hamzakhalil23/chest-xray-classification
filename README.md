# Chest X-ray Classification with Transfer Learning and Grad-CAM

A Master's Deep Learning Applications coursework project comparing a custom CNN, ResNet50 and EfficientNet-B0 for classifying chest X-ray images into **COVID**, **NORMAL** and **PNEUMONIA**.

This is an educational research prototype. It has not been clinically validated and must not be used for medical decisions.

## Implementation

- PyTorch and torchvision for model development; timm for EfficientNet-B0.
- Training augmentation, ImageNet normalisation, AdamW, cosine learning-rate scheduling and early stopping.
- Evaluation using accuracy, weighted F1, macro one-vs-rest ROC-AUC and confusion matrices.
- Grad-CAM visualisations and a Gradio interface with OpenCV preprocessing and heuristic input checks.

## Dataset and experiment

The main notebook references the Kaggle `sachinkumar413/covid-pneumonia-normal-chest-xray-images` dataset. Its saved output records 5,228 images: 1,626 COVID, 1,802 normal and 1,800 pneumonia images. Data is not included. See `data/README.md`.

The historical experiment uses a random split with seed 42: 3,659 training, 784 validation and 785 test images. It does not implement stratification or patient grouping. Duplicate and patient overlap have not been audited from the supplied files.

## Historical results

| Model | Test accuracy | Weighted F1 | Macro ROC-AUC |
|---|---:|---:|---:|
| Custom CNN | 93.50% | 0.9348 | 0.9842 |
| ResNet50 | 97.83% | 0.9784 | 0.9984 |
| EfficientNet-B0 | 97.83% | 0.9784 | 0.9974 |

Source: saved outputs of the main Milestone3 notebook. These results have not yet been independently reproduced. The EfficientNet-B0 confusion matrix contains 768 correct predictions out of 785. Earlier report text gives different values and needs reconciliation. External-image improvement claims are omitted because reproducible evaluation artefacts were not supplied.

## Repository contents and status

`notebooks/chest_xray_classification.ipynb` contains the main experiment and historical textual outputs. `app.py` contains the Gradio demonstration. Supporting documentation is in `docs/`.

Trained weights and configuration were not present in the supplied folders. Training and inference have not been rerun during repository preparation. This repository publishes the implementation and historical evidence, not a currently verified live deployment.

## Environment

Create a Python environment, then install the dependency draft with `python -m pip install -r requirements-notebook.txt`. Package versions still need to be resolved and recorded from a tested run. GPU training is recommended; the historical notebook output identifies a Tesla T4.

The notebook currently uses Google Colab Drive mounting. For local execution, replace the Colab mount/download cells with local dataset and output paths. The expected class folders are `COVID/`, `NORMAL/` and `PNEUMONIA/`.

For the app, export the trusted `efficientnet_b0.pt` and matching `config.json` from the notebook into `models/`, then run:

```sh
python app.py
```

The app resolves its default checkpoint relative to this repository. An explicit `MODEL_PATH` environment variable can override the weight path. The class configuration must match the checkpoint. Model binaries are excluded from Git.

## Limitations

- Historical results cover one random image-level split of one curated dataset.
- The app adds preprocessing and three-pass test-time augmentation not evaluated by the notebook's reported test metrics.
- Input rejection uses unvalidated heuristics. Model scores are not established calibrated diagnostic probabilities.
- Grad-CAM illustrates model attribution; it does not establish clinical validity.
- No reproducible external test, demographic fairness assessment or prospective clinical evaluation is included.

## Attribution and licence

Document the dataset's original sources and redistribution terms before including example images. A code licence has not yet been selected. Add verified project/demo links after checking the existing accounts and deployment.
