# InterpNet

Code to reproduce results from [InterpNET: Neural Introspection for Interpretable Deep Learning](https://arxiv.org/abs/1710.09511).

> Abstract: Humans are able to explain their reasoning. On the contrary, deep neural networks are not. This paper attempts to bridge this gap by introducing a new way to design interpretable neural networks for classification, inspired by physiological evidence of the human visual system's inner-workings. This paper proposes a neural network design paradigm, termed InterpNET, which can be combined with any existing classification architecture to generate natural language explanations of the classifications. The success of the module relies on the assumption that the network's computation and reasoning is represented in its internal layer activations. While in principle InterpNET could be applied to any existing classification architecture, it is evaluated via an image classification and explanation task. Experiments on a CUB bird classification and explanation dataset show qualitatively and quantitatively that the model is able to generate high-quality explanations. While the current state-of-the-art METEOR score on this dataset is 29.2, InterpNET achieves a much higher METEOR score of 37.9.

## Reproducing Results

Clone the repository and install requirements. You will also need [TensorFlow](https://www.tensorflow.org/):
```
$ git clone [repo_url]
$ cd interpnet
$ pip install -r requirements.txt
$ python
>>> import nltk
>>> nltk.download('punkt')
```

### Get the Data
Download [the data](https://drive.google.com/file/d/1i4Fyn9fFXCGDVcqY8hCbdHM70HTeDrhE/view?usp=sharing) and move to `interpnet/` folder.

Unzip the data:
```
$ unzip data.zip
```

This data contains 8,192 vectors preprocessed using Bilinear Compact Pooling. If you want the real images data, you can get it from [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

### Install METEOR (you will need java, should come installed by default)
```
$ mkdir ~/src
$ cd ~/src
$ wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
$ tar -xzf meteor-1.5.tar.gz
```

### Train the Models and evaluate them (this takes at least 4 hours on a Nvidia 1080 GPU)

To check progress at any time, run `python print_runs.py`.

Captioning:
```
$ python main.py --dataroot data/ --experiment run-captioning/ --captioning
$ python prep_evaluate.py --dataroot data/ --experiment run-captioning/
$ java -Xmx2G -jar ~/src/meteor-1.5/meteor-1.5.jar run-captioning/explanations.txt run-captioning/reference.txt -l en -norm -r 10 > run-captioning/meteor_results.txt
$ python evaluate.py --dataroot data/ --experiment run-captioning/
```

InterpNet(0):
```
$ python main.py --dataroot data/ --experiment run0/ --hiddenlayers 1 --output_only
$ python prep_evaluate.py --dataroot data/ --experiment run0/
$ java -Xmx2G -jar ~/src/meteor-1.5/meteor-1.5.jar run0/explanations.txt run0/reference.txt -l en -norm -r 10 > run0/meteor_results.txt
$ python evaluate.py --dataroot data/ --experiment run0/
```

InterpNet(1):
```
$ python main.py --dataroot data/ --experiment run1/ --hiddenlayers 1
$ python prep_evaluate.py --dataroot data/ --experiment run1/
$ java -Xmx2G -jar ~/src/meteor-1.5/meteor-1.5.jar run1/explanations.txt run1/reference.txt -l en -norm -r 10 > run1/meteor_results.txt
$ python evaluate.py --dataroot data/ --experiment run1/
```

InterpNet(2):
```
$ python main.py --dataroot data/ --experiment run2/ --hiddenlayers 2
$ python prep_evaluate.py --dataroot data/ --experiment run2/
$ java -Xmx2G -jar ~/src/meteor-1.5/meteor-1.5.jar run2/explanations.txt run2/reference.txt -l en -norm -r 10 > run2/meteor_results.txt
$ python evaluate.py --dataroot data/ --experiment run2/
```