# TensorFlow implementation of "A Hybrid Convolutional Variational Autoencoder for Text Generation"

paper:[https://arxiv.org/abs/1702.02390](https://arxiv.org/abs/1702.02390)
original implementation (Theano):[https://github.com/stas-semeniuta/textvae](https://github.com/stas-semeniuta/textvae)

This is NOT an original implementation. There may be some minor differences from the original structure.

## Prerequisites

 * Python 3.5
 * tensorflow-gpu==1.3.0
 * matplotlib==2.0.2
 * numpy==1.13.1
 * scikit-learn==0.19.0


## Preparation

Dataset is not contained. Please prepare your own dataset.

 * Sentence

Pickle file of Numpy array of word ids (shape=[batch_size, sentence_length]).

 * Dictionary

Pickle file of Python dictionary. It should contain "<EOS>", "<PAD>", "<GO>" as meta words.

```python
  dictionary = {word1: id1,
                word2: id2,
                ...}
```

## Usage
### Train

1. modify config.py
2. run

```bash
  python train.py
```

### Get sample sentences

1. modify sampling.py
2. run

```bash
  python sampling.py
```

## License

MIT

## Author

Ryo Kamoi
