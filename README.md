# Environment-Aware Embeddings for Keyword Spotting 

This repository contains the implementation of "Boosting keyword spotting through on-device learnable user speech characteristics". It implements an on-device learning architecture, composed of a (pretrained) backbone and an environment-aware embeding. The embeddings can learn the speech caracteristics of a target user or the characteristics of the background noise present in the environment. The embeddings are fused with the backbone and can be trained separately or together with the classifier and/or the backbone. Please cite the following publication if you use our implementation:

```
@misc{cioflan2024boostingkeywordspottingondevice,
      title={Boosting keyword spotting through on-device learnable user speech characteristics}, 
      author={Cristian Cioflan and Lukas Cavigelli and Luca Benini},
      year={2024},
      eprint={2403.07802},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2403.07802}, 
}
```

The repository contains the resources to train and fine-tune neural networks. The resources containing deployment and on-device training will be added soon. 

## Requirements

To install the packages required to pretraind and fine-tune the models, a conda environment can be created from environment.yml by running:
```
conda env create -f environment.yml
```


## Example

To run the main script, use the command:

```
python main.py --config_file example.json
```

The preprocessing, environment, architecture, training, and experimental parameters can be configured individually in the `.json` configuration file. For instance, `example.json` sets up the pretraining of a DSCNNS network, followed by the learning of the user embedings, fused with the backbone through element-wise multiplication. The number of training utterances per user per class during the fine-tuning stage is fixed to four. The pretrained model is robust to noises (i.e, "noise aware", as described by Cioflan et al. in [On-Device Domain Learning for Keyword Spotting on Low-Power Extreme Edge Embedded Systems](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10595987)) and the testing environment is a *meeting* environment with speech background noise.


## Contributors

* Cristian Cioflan, ETH Zurich [cioflanc@iis.ee.ethz.ch](cioflanc@iis.ee.ethz.ch)
* Jacky Choi, ETH Zurich
* Maximilian Grözinger, ETH Zurich


## License
Unless explicitly stated otherwise, the code is released under Apache 2.0, see the LICENSE file in the root of this repository for details. 

As an exception, the proposed partitions of the [Google Speech Commands v2](https://arxiv.org/abs/1804.03209) dataset available in `dataset/` are released under Creative Commons Attribution-NoDerivatives 4.0 International. Please see the LICENSE file in their respective directory. 
