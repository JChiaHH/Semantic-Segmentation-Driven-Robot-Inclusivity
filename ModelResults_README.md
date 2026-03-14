# Model Results README

This document complements [`ModelTraining_README.md`](ModelTraining_README.md).
Use it to review the dataset scale, quantitative metrics, and qualitative inference examples for the three semantic segmentation models used in this project.

## Dataset Size Summary

This workflow uses a combined dataset made up of real-world and synthetic point clouds.
The dataset summary is:

| Dataset Type | No. of Point Clouds | No. of Points | Overall Percentage |
| --- | ---: | ---: | ---: |
| Real World | 14 | 37,352,003 | 82.34% |
| Synthetic | 10 | 8,005,974 | 17.66% |
| Total | 24 | 45,357,977 | 100% |

After oversampling, the training dataset contains `68,535,242` points.

## Quantitative Model Performance

The quantitative performance summary for the three models in this workflow is:

| Model | Best Epoch | Training IoU | Training Accuracy | Validation IoU | Validation Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandLA-Net | 250 | 0.6220 | 67.65% | 0.2657 | 36.35% |
| KPConv | 399 | 0.7331 | 77.4% | 0.2509 | 37.89% |
| PointTransformer | 185 | 0.5448 | 79.83% | 0.2081 | 38.78% |

## Qualitative Model Performance

| Model | Training Set Inference | Generalization Inference |
| --- | --- | --- |
| RandLA-Net | Performs well on trained-set inference. | Difficult to generalize and prone to misclassification. |
| KPConv | Segments points more precisely but is prone to under-classification. | Captures classes with good precision but remains prone to under-classification. |
| PointTransformer | Correctly segments a larger number of points but still exhibits some misclassifications. | Demonstrates the strongest overall inference performance, detecting multiple classes accurately with minor misclassifications. |

These results should be interpreted with the practical constraints of this project in mind.
Because access to construction sites is difficult, logistically constrained, safety-limited, and time-intensive, the current model performance may not yet be ideal for a task of this complexity.
That said, this workflow is scalable and can be improved further with an expanded dataset, whether through more synthetic data, more real-world data, or both.

## Example Inference Outputs

### RandLA-Net

Training-set inference example:

![RandLA-Net training example](images/Training_RandLANet.png)

Generalization example:

![RandLA-Net generalization example](images/Generalization_RandLAnet.png)

### KPConv

Training-set inference example:

![KPConv training example](images/Training_kpConv.png)

Generalization example:

![KPConv generalization example](images/Generlization_KPConv.png)

### PointTransformer

Training-set inference example:

![PointTransformer training-set inference example](images/TrainingSetInference_PointTransformers.png)

Generalization example:

![PointTransformer generalization example](images/Generalization_PointTransformers.png)

## Related files

- Device / environment setup: [`README.md`](README.md)
- Dataset preparation: [`dataset_prep_readme.md`](dataset_prep_readme.md)
- Training and inference commands: [`ModelTraining_README.md`](ModelTraining_README.md)

## References

- RII Pipeline: [`rii_pipeline/README.md`](rii_pipeline/README.md)
- Handheld LiDAR LIOSAM: [https://github.com/JChiaHH/Handheld_Lidar_LIOSAM](https://github.com/JChiaHH/Handheld_Lidar_LIOSAM)
