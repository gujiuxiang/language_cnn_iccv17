"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""
import skimage
import skimage.io
import os
import json
import cPickle
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize
#from misc._init_paths
#import caffe, test_model, cap_eval_utils,\
#import misc.sg_utils as utils
import cv2, numpy as np
import matplotlib
import matplotlib.pyplot as plt

def det_label_init():
    # Load the vocabulary
    '''
    vocab_file = 'misc/visual-concepts/vocabs/vocab_train.pkl'
    vocab = utils.load_variables(vocab_file)

    # Set up Caffe
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # Load the model
    mean = np.array([[[103.939, 116.779, 123.68]]]);
    base_image_size = 565;
    prototxt_deploy = 'misc/visual_concepts/code/output/vgg/mil_finetune.prototxt.deploy'
    model_file = 'misc/visual_concepts/code/output/vgg/snapshot_iter_240000.caffemodel'
    model = test_model.load_model(prototxt_deploy, model_file, base_image_size, mean, vocab)
    # define functional words
    functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are']
    is_functional = np.array([x not in functional_words for x in vocab['words']])

    # load the score precision mapping file
    eval_file = 'misc/visual_concepts/code/code/output/vgg/snapshot_iter_240000.caffemodel_output/coco_valid1_eval.pkl'
    pt = utils.load_variables(eval_file);

    # Set threshold_metric_name and output_metric_name
    threshold_metric_name = 'prec';
    output_metric_name = 'prec';
    return model,functional_words,threshold_metric_name,output_metric_name,vocab,is_functional,pt
    '''
def det_label_init_gt():
    # Load the vocabulary
    '''
    vocab_file = 'misc/visual-concepts/vocabs/vocab_train.pkl'
    vocab = utils.load_variables(vocab_file)

    # define functional words
    functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are']
    is_functional = np.array([x not in functional_words for x in vocab['words']])

    # load the score precision mapping file
    eval_file = 'misc/visual_concepts/code/code/output/vgg/snapshot_iter_240000.caffemodel_output/coco_valid1_eval.pkl'
    pt = utils.load_variables(eval_file);

    return functional_words,vocab,is_functional,pt
    '''
def det_label_fun(im, model,functional_words,threshold_metric_name,output_metric_name,vocab,is_functional,pt,wtoi,):
    # Load the image
    '''
    # Run the model
    dt = {}
    dt['sc'], dt['mil_prob'] = test_model.test_img(im, model['net'], model['base_image_size'], model['means'])

    # Compute precision mapping - slow in per image mode, much faster in batch mode
    det_atts = []
    det_atts_w = []
    prec = np.zeros(dt['mil_prob'].shape)
    for ii in xrange(prec.shape[0]):
        for jj in xrange(prec.shape[1]):
            prec[ii, jj] = cap_eval_utils.compute_precision_score_mapping( \
                pt['details']['score'][:, jj] * 1, \
                pt['details']['precision'][:, jj] * 1, \
                dt['mil_prob'][ii, jj] * 1);
        dt['prec'] = prec[ii,:]
        #cv2.imshow('image', im)
        # Output words
        out = test_model.output_words_image(prec[ii,:], prec[ii,:], \
                                            min_words=3, threshold=0.5, vocab=vocab, is_functional=is_functional)
        #plt.rcParams['figure.figsize'] = (10, 10)
        #plt.imshow(im[:, :, [2, 1, 0]])
        #plt.gca().set_axis_off()
        index = 0
        for (a, b, c) in out:
            if a not in functional_words:
                if index < 16:
                    det_atts.append(wtoi[a])
                    det_atts_w.append(np.round(b, 2))
                    index = index + 1
                    # print '{:s} [{:.2f}, {:.2f}]   '.format(a, np.round(b,2), np.round(c,2))
    return det_atts, det_atts_w
    '''
def prepro_captions(imgs):
    # preprocess all the captions
    print 'example processed tokens:'
    for i, img in enumerate(imgs):
        img['processed_tokens'] = []
        for j, s in enumerate(img['captions']):
            txt = str(s).lower().translate(None, string.punctuation).strip().split()
            img['processed_tokens'].append(txt)
            if i < 10 and j == 0: print txt


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str, cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    vocab = [w for w, n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    print 'number of words in vocab would be %d' % (len(vocab),)
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words)

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print 'max length sentence in raw data: ', max_len
    print 'sentence length distribution (count, number of words):'
    sum_len = sum(sent_lengths.values())
    for i in xrange(max_len + 1):
        print '%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len)

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print 'inserting the special UNK token'
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for txt in img['processed_tokens']:
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def assign_splits(imgs, params):
    num_val = params['num_val']
    num_test = params['num_test']

    for i, img in enumerate(imgs):
        if i < num_val:
            img['split'] = 'val'
        elif i < num_val + num_test:
            img['split'] = 'test'
        else:
            img['split'] = 'train'

    print 'assigned %d to val, %d to test.' % (num_val, num_test)

def upsample_image(im, sz):
      h = im.shape[0]
      w = im.shape[1]
      s = np.float(max(h, w))
      I_out = np.zeros((sz, sz, 3), dtype = np.float);
      I = cv2.resize(im, None, None, fx = np.float(sz)/s, fy = np.float(sz)/s, interpolation=cv2.INTER_LINEAR);
      SZ = I.shape;
      I_out[0:I.shape[0], 0:I.shape[1],:] = I;
      return I_out, I, SZ

def load_variables(pickle_file_name):
  """
  d = load_variables(pickle_file_name)
  Output:
    d     is a dictionary of variables stored in the pickle file.
  """
  if os.path.exists(pickle_file_name):
    with open(pickle_file_name, 'rb') as f:
      d = cPickle.load(f)
    return d
  else:
    raise Exception('{:s} does not exists.'.format(pickle_file_name))

def encode_captions(imgs, params, wtoi,model,functional_words,threshold_metric_name,output_metric_name,vocab,is_functional,pt):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """
    #from misc.sg_utils import as utils
    vocab = load_variables('misc/visual_concepts/code/vocabs/vocab_train.pkl')
    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions
    counts = np.zeros((len(imgs), len(vocab['words'])), dtype=np.float)
    label_attributes = []
    label_attributes_prob = []
    label_arrays = []
    label_semantic = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    image_files = []

    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        semant_label = np.zeros(len(vocab['words']), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]
            pos = [vocab['words'].index(tmp_j_k) for tmp_j_k in s if tmp_j_k in vocab['words']]
            pos = list(set(pos))
            counts[i, pos] = counts[i, pos] + 1

        sort_counts=np.argsort(counts[i], axis=0)[::-1]
        import numpy
        sort_key = sorted(counts[i], reverse=True)
        for m in range(len(sort_key)):
            if sort_key[m] >0 :
                semant_label[m] = wtoi[vocab['words'][sort_counts[m]]]

        image_files.append(img['file_path'])

        label_arrays.append(Li)
        label_semantic.append(semant_label)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n


    #L_semantic = np.concatenate(label_semantic, axis=0)  # put all the labels together
    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print 'encoded captions to array of size ', `L.shape`
    return L, label_start_ix, label_end_ix, label_length, label_semantic, label_attributes, label_attributes_prob

def encode_captions_gt(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """
    #import misc.sg_utils as utils
    vocab = load_variables('/home/jxgu/github/cvpr2017_im2text_jxgu.pytorch/misc/visual-concepts/vocabs/vocab_train.pkl')
    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions
    counts = np.zeros((len(imgs), len(vocab['words'])), dtype=np.float)
    label_attributes = []
    label_attributes_prob = []
    label_arrays = []
    label_semantic = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    image_files = []

    functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are']
    is_functional = np.array([x not in functional_words for x in vocab['words']])

    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        semant_label = np.zeros(len(vocab['words']), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]
            pos = [vocab['words'].index(tmp_j_k) for tmp_j_k in s if tmp_j_k in vocab['words']]
            pos = list(set(pos))
            counts[i, pos] = counts[i, pos] + 1

        sort_counts=np.argsort(counts[i], axis=0)[::-1]
        import numpy
        sort_key = sorted(counts[i], reverse=True)
        index = 0
        for m in range(len(sort_key)):
            if sort_key[m] >0 :
                tw = vocab['words'][sort_counts[m]]
                if tw not in functional_words:
                    semant_label[index] = wtoi[tw]
                    index = index + 1

        image_files.append(img['file_path'])

        label_arrays.append(Li)
        label_semantic.append(semant_label)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n


    #L_semantic = np.concatenate(label_semantic, axis=0)  # put all the labels together
    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print 'encoded captions to array of size ', `L.shape`
    return L, label_start_ix, label_end_ix, label_length, label_semantic

def main(params):
    #model, functional_words, threshold_metric_name, output_metric_name, vocab, is_functional, pt = det_label_init()

    imgs = json.load(open(params['input_json'], 'r'))
    seed(123)  # make reproducible
    shuffle(imgs)  # shuffle the order

    # tokenization and preprocessing
    prepro_captions(imgs)

    # create the vocab
    vocab = build_vocab(imgs, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    # assign the splits
    assign_splits(imgs, params)

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length, L_semantic = encode_captions_gt(imgs, params, wtoi)

    # create output h5 file
    N = len(imgs)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("semantic_words", dtype='uint32', data=L_semantic)
    f.create_dataset("labels", dtype='uint32', data=L)
    f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f.create_dataset("label_length", dtype='uint32', data=label_length)
    dset = f.create_dataset("images", (N, 3, 256, 256), dtype='uint8')  # space for resized images
    for i, img in enumerate(imgs):
        # load the image
        I = imread(os.path.join(params['images_root'], img['file_path']))
        try:
            Ir = imresize(I, (256, 256))
        except:
            print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir, Ir, Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2, 0, 1)
        # write to h5
        dset[i] = Ir
        if i % 1000 == 0:
            print 'processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N)
    f.close()
    print 'wrote ', params['output_h5']

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):

        jimg = {}
        jimg['split'] = img['split']
        if 'file_path' in img: jimg['file_path'] = img['file_path']  # copy it over, might need
        if 'id' in img: jimg['id'] = img['id']  # copy over & mantain an id, if present (e.g. coco ids, useful)

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='/home/jxgu/github/cvpr2017_im2text_jxgu.pytorch/data/mscoco/coco_raw.json',  help='input json file to process into hdf5')
    parser.add_argument('--num_val',  type=int,default=5000,
                        help='number of images to assign to validation data (for CV etc)')
    parser.add_argument('--output_json', default='data/cocotalk_0406.json', help='output json file')
    parser.add_argument('--output_h5', default='data/cocotalk_0406.h5', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--images_root', default='/home/jxgu/github/cvpr2017_im2text_jxgu.pytorch/data/mscoco/images',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--num_test', default=5000, type=int,
                        help='number of test images (to withold until very very end)')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)
