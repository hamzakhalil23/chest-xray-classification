# Model card

Intended use: educational comparison of CNN architectures on a three-class chest X-ray dataset. Classes: COVID, NORMAL, PNEUMONIA. No clinical use is supported.

No trained checkpoint is distributed. Saved notebook results cover one random image-level split; patient independence and external generalisation are not established. No demographic audit, calibration assessment or clinical validation was supplied.

The app adds grayscale conversion, border cropping, contrast enhancement and three inference passes (original, horizontal flip, fixed five-degree rotation). Input filtering is heuristic. Scores are uncalibrated model outputs. Grad-CAM shows attribution for a single processed-image pass, while the prediction averages all three passes. It is not a bounding-box detector or a clinical explanation.
