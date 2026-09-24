# Methodology

Three-class image-folder classification with 224×224 input and ImageNet normalisation. Training adds horizontal flips, rotation, brightness/contrast jitter and affine translation. The dataset is split randomly with seed 42 into 3,659 training, 784 validation and 785 test images; this is not a stratified or patient-grouped split.

Models: custom four-block CNN, torchvision ResNet50 with a replacement classification head, and timm EfficientNet-B0. Training uses cross-entropy, AdamW, cosine learning-rate scheduling, gradient clipping and early stopping. Frozen-backbone and no-augmentation ResNet50 experiments are included. These experiments also vary training settings and should not be treated as perfectly isolated causal comparisons.

Before reporting new results, verify dataset provenance, duplicate/patient overlap and class mapping, preserve a split manifest, and record tested dependency versions.
