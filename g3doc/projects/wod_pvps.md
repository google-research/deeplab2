# Panoramic Video Panoptic Segmentation

Waymo Open Dataset: Panoramic Video Panoptic Segmentation (WOD-PVPS) [1], is a
large-scale dataset that offers high-quality multi-camera video panoptic
segmentation labels for autonomous driving. The labels are consistent over time
for video processing and consistent across multiple cameras mounted on the
vehicles for full panoramic scene understanding.

The new task of Panoramic Video Panoptic Segmentation requires generating dense
panoptic segmentation predictions consistent in both time and multi-cameras. To
build a baseline for such a challenging task, we extend the ViP-DeepLab [2] to
the multi-camera setting.

## Prerequisite

1.  Make sure the software is properly [installed](../setup/installation.md).

2.  Make sure the
    [target dataset](https://waymo.com/open/data/perception/#2d-video-panoptic-segmentation)
    is correctly prepared.

## Model Zoo

## Citing WOD-PVPS

If you find this code helpful in your research or wish to refer to the baseline
results, please use the following BibTeX entry.

*   Waymo Open Dataset: Panoramic Video Panoptic Segmentation:

```
@article{mei2022waymo,
  title={Waymo Open Dataset: Panoramic Video Panoptic Segmentation},
  author={Mei, Jieru and Zhu, Alex Zihao and Yan, Xinchen and Yan, Hang and Qiao, Siyuan and Zhu, Yukun and Chen, Liang-Chieh and Kretzschmar, Henrik and Anguelov, Dragomir},
  journal={arXiv preprint arXiv:2206.07704},
  year={2022}
}

```

*   ViP-DeepLab:

```
@inproceedings{vip_deeplab_2021,
  author={Siyuan Qiao and Yukun Zhu and Hartwig Adam and Alan Yuille and Liang-Chieh Chen},
  title={{ViP-DeepLab}: Learning Visual Perception with Depth-aware Video Panoptic Segmentation},
  booktitle={CVPR},
  year={2021}
}

```

*   Panoptic-DeepLab:

```
@inproceedings{panoptic_deeplab_2020,
  author={Bowen Cheng and Maxwell D Collins and Yukun Zhu and Ting Liu and Thomas S Huang and Hartwig Adam and Liang-Chieh Chen},
  title={{Panoptic-DeepLab}: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation},
  booktitle={CVPR},
  year={2020}
}

```

### References

1.  Jieru Mei, Alex Zihao Zhu, Xinchen Yan, Hang Yan, Siyuan Qiao, Yukun Zhu,
    Liang-Chieh Chen, Henrik Kretzschmar, Dragomir Anguelov. "Waymo Open
    Dataset: Panoramic Video Panoptic Segmentation." In arXiv: 2206.07704, 2022.

2.  Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
    "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
    Segmentation." In CVPR, 2021.
