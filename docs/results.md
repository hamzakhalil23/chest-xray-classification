# Historical results

Source: local Milestone3 notebook, saved outputs. No fresh evaluation was performed during packaging.

| Model | Test accuracy | Weighted F1 | Macro AUC |
|---|---:|---:|---:|
| Custom CNN | 93.50% | 0.9348 | 0.9842 |
| ResNet50 | 97.83% | 0.9784 | 0.9984 |
| EfficientNet-B0 | 97.83% | 0.9784 | 0.9974 |

EfficientNet confusion matrix, rows actual and columns predicted, class order COVID, NORMAL, PNEUMONIA:

```text
227   1   1
  0 267   3
  0  12 274
```

768/785 predictions are correct. Pneumonia recall is 274/286, approximately 95.80%. Earlier report prose quoted 96.18% overall accuracy, inconsistent with this saved matrix. The repository reports the saved output with this qualification. External-test improvement claims were excluded because source images, labels and prediction records were unavailable.

The notebook’s historical output predates packaging edits and should be regenerated for reproducibility. The app applies additional preprocessing and TTA, so these figures do not measure the complete app pipeline.
