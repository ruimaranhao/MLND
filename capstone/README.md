# MLND Deep Learning Capstone

This project is done in the context of Udacity's Machine Learning Nano Degree
capstone project. The idea is to use deep learning, via tensorflow, to detect
Street View House Number.

## Data

This project uses the [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/).

As taken from the official website, SVHN is a real-world image dataset for
developing machine learning and object recognition algorithms with minimal
requirement on data preprocessing and formatting. It can be seen as similar
in flavor to MNIST (e.g., the images are of small cropped digits), but
incorporates an order of magnitude more labeled data (over 600,000 digit images)
and comes from a significantly harder, unsolved, real world problem
(recognizing digits and numbers in natural scene images). SVHN is obtained from
house numbers in Google Street View images.

The dataset consists of

- 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
- 73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data
Comes in two formats:
  1. Original images with character level bounding boxes.
  2. MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).

Visit the website to obtain further information and links to download the data.

## Source Code

The code is split into a couple of notebooks, where the data was fetched,
explored and the tensorflow model was trained. It is also provided a web
service that could be used to predict the house numbers.

## Requirements

This project requires **Python 3.0** and the following Python libraries installed:

- [TensorFlow](http://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [SciPy library](http://www.scipy.org/scipylib/index.html)
- [Six](http://pypi.python.org/pypi/six/)
- [h5py](http://pypi.python.org/pypi/h5py/)
- [Pillow](http://pypi.python.org/pypi/Pillow/)
- [Flask](http://flask.pocoo.org/)

To install them, go to the `code` folder and type `python install -r requirements.txt`. 

Also, [iPython Notebook](http://ipython.org/notebook.html) is required to run the
jupyter notebooks.

## Run

The exploratory notebooks can be run using `Ipython Notebook`. The webservice
is run using python. 

## Reference

Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. (PDF)

URL: http://ufldl.stanford.edu/housenumbers

For questions regarding the dataset, please contact streetviewhousenumbers@gmail.com
