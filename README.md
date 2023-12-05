# cft-chexpert
This is the best single CNN model on the CheXpert chest X-ray multi-label classification competition leaderboard.
The model is developed with **Category-wise Fine-Tuning (CFT)** approach proposed in the paper below.
It achieves mAUC 91.8%, approaching the best ensemble on the leaderboard (93.0%). This model has also surpassed a series of ensembles on the leaderboard, ranking at #53.
By ensembling multiple single CNN models developed with CFT, the ensemble can achieve mAUC 93.3%, outperforming the best ensemble on the leaderboard (93.0%). The leaderboard can be found [here](https://stanfordmlgroup.github.io/competitions/chexpert/).

![image](https://github.com/maxium0526/cft-chexpert/assets/38188772/f33a9ce4-89a2-44d2-8116-c189fe17e8a3)

## The Paper
### Category-Wise Fine-Tuning for Image Multi-label Classification with Partial Labels
- [Research Gate](https://www.researchgate.net/publication/375956350_Category-Wise_Fine-Tuning_for_Image_Multi-label_Classification_with_Partial_Labels) (Feel free to ask for the full-text paper through RG)
- [Springer](https://link.springer.com/chapter/10.1007/978-981-99-8145-8_26)

#### Abstract
Image multi-label classification datasets are often partially labeled (for each sample, only the labels on some categories are known).
One popular solution for training convolutional neural networks is treating all unknown labels as negative labels, named Negative mode.
But it produces wrong labels unevenly over categories, decreasing the binary classification performance on different categories to varying degrees.
On the other hand, although Ignore mode that ignores the contributions of unknown labels may be less effective than Negative mode, it ensures the data have no additional wrong labels, which is what Negative mode lacks.
In this paper, we propose Category-wise Fine-Tuning (CFT), a new post-training method that can be applied to a model trained with Negative mode to improve its performance on each category independently.
Specifically, CFT uses Ignore mode to one-by-one fine-tune the logistic regressions (LRs) in the classification layer.
The use of Ignore mode reduces the performance decreases caused by the wrong labels of Negative mode during training.
Particularly, Genetic Algorithm (GA) and binary crossentropy are used in CFT for fine-tuning the LRs.
The effectiveness of our methods was evaluated on the CheXpert competition dataset and achieves state-of-the-art results, to our knowledge.
A single model submitted to the competition server for the official evaluation achieves mAUC 91.82% on the test set, which is the highest single model score in the leaderboard and literature.
Moreover, our ensemble achieves mAUC 93.33% (The competition was recently closed.
We evaluate the ensemble on a local machine after the test set is released and can be downloaded.) on the test set, superior to the best in the leaderboard and literature (93.05%).
Besides, the effectiveness of our methods is also evaluated on the partially labeled versions of the MS-COCO dataset.

#### Citing This Paper

    @inproceedings{chong2023category,
      title={Category-Wise Fine-Tuning for Image Multi-label Classification with Partial Labels},
      author={Chong, Chak Fong and Yang, Xu and Wang, Tenglong and Ke, Wei and Wang, Yapeng},
      booktitle={International Conference on Neural Information Processing},
      pages={332--345},
      year={2023},
      organization={Springer}
    }

## Running the Code
This repository provides a demo code (`predict.py`) that inferences the model on an input chest X-ray and produces the predicted likelihood of 14 pathologies. To run this code, simply:

    python predict.py <path_to_the_image>

For example,

    python predict.py demo_img.jpg

## Training
As we are still working on the paper, we will publish training code in the future.

## Disclaimer

This model is developed using the CheXpert dataset. Please refer to the [Licenses](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2) of the dataset before using the model for any purposes.
