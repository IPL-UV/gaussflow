<div align="center">    
 
# Gaussianization Flows

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->


<!--  
Conference   
-->   
</div>
 
## Description

This project features some exploration to get a fully parameterized Gaussianization scheme. There is a big normalizing flows community with many different algorithms for density estimation and sampling. There is also a relatively small community using Gaussianization and density destructors for other applications including information theory measures. This is an attempt to bridge the two communities together.

## References

This project was inspired by:

* [RBIG](https://github.com/IPL-UV/rbig) - original algorithm
* [Gaussianization Flows](https://github.com/chenlin9/Gaussianization_Flows) - the fully parameterized method.
* [nflows](https://github.com/bayesiains/nflows) - the research normalizing flows library.





<!-- ## How to run

First, install dependencies   
```bash
# clone project   
git clone https://github.com/IPL-UV/gaussflow

# install project   
cd gaussflow 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
``` -->

<!-- ## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
``` -->

<!-- ### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```    -->


