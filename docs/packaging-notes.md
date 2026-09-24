# Packaging notes

Prepared from the local Milestone3 notebook and app. Original source folders were not changed.

Removed student identifiers, obsolete repository/demo links, unsupported clinical claims and conflicting narrative. Retained model/training code and selected historical textual outputs. Embedded image outputs were excluded pending provenance/redistribution review.

App changes: aligned model paths with notebook exports, aligned heatmap overlay with the cropped processed image, replaced random TTA rotation with fixed rotation, corrected the heuristic score denominator and displayed no image on Grad-CAM failure. These changes have syntax validation only; end-to-end behaviour awaits the trained checkpoint and a tested ML environment.

The dataset and package versions remain external dependencies. No licence has been inferred from old badges.
