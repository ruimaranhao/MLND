#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import h5py
import numpy as np
import string

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent

def maybe_download(url, filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print('Failed to verify ' + filename + '. Please delete and try again')
  return filename

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  if not os.path.exists(root+'/digitStruct.mat'):
    print("digitStruct.mat is missing")
  return root+'/digitStruct.mat'

def get_attr(c, i, attr):
    d = c[c['digitStruct']['bbox'][i][0]][attr].value.squeeze()
    if d.dtype == 'float64':
        return d.reshape(-1)
    return np.array([c[x].value for x in d]).squeeze()

def get_label(c, i):
    d = c[c['digitStruct']['name'][i][0]].value.tostring()
    return d.replace(b'\x00', b'')

def load_data(path):
    c = h5py.File(path)
    images = a = np.ndarray(shape=(c['digitStruct']['name'].shape[0], ), dtype='|S15')
    labels = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    labels.fill(10)
    tops = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    heights = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    widths = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    lefts = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    for i in range(c['digitStruct']['name'].shape[0]):
        images[i] = get_label(c, i)
        l = get_attr(c, i, 'label')
        t = get_attr(c, i, 'top')
        h = get_attr(c, i, 'height')
        w = get_attr(c, i, 'width')
        le = get_attr(c, i, 'left')

        labels[i, :l.shape[0]] = l
        tops[i, :t.shape[0]] = t
        heights[i, :h.shape[0]] = h
        widths[i, :w.shape[0]] = w
        lefts[i, :le.shape[0]] = le

        if (i % 5000 == 0):
            print(i, "elapsed")

    return labels, images, tops, heights, widths, lefts

def maybe_pickle(struct, trainTuple, testTuple, extraTuple, force=False):
    if os.path.exists(struct + '.pickle') and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % struct)
    else:
      print('Pickling %s.' % struct + '.pickle')
      permutation = np.random.permutation(extraTuple[1].shape[0])[:2000]
      dataset = {
            'train': {
                'labels': trainTuple[0],
                'images': trainTuple[1],
                'tops': trainTuple[2],
                'heights': trainTuple[3],
                'widths': trainTuple[4],
                'lefts': trainTuple[5],
            },
            'test': {
                'labels': testTuple[0],
                'images': testTuple[1],
                'tops': testTuple[2],
                'heights': testTuple[3],
                'widths': testTuple[4],
                'lefts': testTuple[5],
            },
            'extra': {
                'labels': extraTuple[0],
                'images': extraTuple[1],
                'tops': extraTuple[2],
                'heights': extraTuple[3],
                'widths': extraTuple[4],
                'lefts': extraTuple[5],
            },
            'valid': {
                'labels': extraTuple[0][permutation],
                'images': extraTuple[1][permutation],
                'tops': extraTuple[2][permutation],
                'heights': extraTuple[3][permutation],
                'widths': extraTuple[4][permutation],
                'lefts': extraTuple[5][permutation],
            }
      }
      try:
        with open( struct + '.pickle', 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to',  struct + '.pickle', ':', e)

    return  struct + '.pickle'
