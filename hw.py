# -*- coding: utf-8 -*-
""" CIS6261TML -- Homework 1 -- hw.py

# This file is the main homework file
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist

import scipy.stats as stats

# our neural network architectures
import nets
import attacks

## os / paths
def ensure_exists(dir_fp):
    if not os.path.exists(dir_fp):
        os.makedirs(dir_fp)

## parsing / string conversion to int / float
def is_int(s):
    try:
        z = int(s)
        return z
    except ValueError:
        return None


def is_number(s):
    try:
        z = int(s)
        return z
    except ValueError:
        try:
            z = float(s)
            return z
        except ValueError:
            return None


"""
## Save model to file
"""
def save_model(model, base_fp):
    # save the model: first the weights then the arch
    model.save_weights('{}-weights.h5'.format(base_fp))
    with open('{}-architecture.json'.format(base_fp), 'w') as f:
        f.write(model.to_json())


import hashlib

def memv_filehash(fp):
    hv = hashlib.sha256()
    buf = bytearray(512 * 1024)
    memv = memoryview(buf)
    with open(fp, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(memv), 0):
            hv.update(memv[:n])
    return hv.hexdigest()


"""
## Load model from file
"""        
def load_model(base_fp):
    # Model reconstruction from JSON file
    arch_json_fp = '{}-architecture.json'.format(base_fp)
    if not os.path.isfile(arch_json_fp):
        return None
        
    with open(arch_json_fp, 'r') as f:
        model = keras.models.model_from_json(f.read())

    wfp = '{}-weights.h5'.format(base_fp)

    # Load weights into the new model
    model.load_weights(wfp)
    
    hv = memv_filehash(wfp)
    
    print('Loaded model from file ({}) -- [{}].'.format(base_fp, hv[-17:-1].upper()))
    return model, hv
    
    
    

"""
## Load and preprocess the dataset
"""
def load_preprocess_mnist_data(train_size=50000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # MNIST has overall shape (60000, 28, 28) -- 60k images, each is 28x28 pixels
    print('Loaded mnist data; shape: {} [y: {}], test shape: {} [y: {}]'.format(x_train.shape, y_train.shape,
                                                                                      x_test.shape, y_test.shape))
    # Let's flatten the images for easier processing (labels don't change)
    flat_vector_size = 28 * 28
    x_train = x_train.reshape(x_train.shape[0], flat_vector_size)
    x_test = x_test.reshape(x_test.shape[0], flat_vector_size)
    
    assert x_train.shape[0] > train_size

    # Put the labels in "one-hot" encoding using keras' to_categorical()
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # let's split the training set further
    aux_idx = train_size

    x_aux = x_train[aux_idx:,:]
    y_aux = y_train[aux_idx:,:]

    x_temp = x_train[:aux_idx,:]
    y_temp = y_train[:aux_idx,:]

    x_train = x_temp
    y_train = y_temp

    return (x_train, y_train), (x_test, y_test), (x_aux, y_aux)


"""
## Plots a set of images (all m x m)
## input is  a square number of images, i.e., np.array with shape (z*z, dim_x, dim_y) for some integer z > 1
"""
def plot_images(im, dim_x=28, dim_y=28, one_row=False, out_fp='out.png', save=False, show=True, cmap='gray', fig_size=(14,14), titles=None, titles_fontsize=12):
    fig = plt.figure(figsize=fig_size)
    im = im.reshape((-1, dim_x, dim_y))

    num = im.shape[0]
    assert num <= 3 or np.sqrt(num)**2 == num or one_row, 'Number of images is too large or not a perfect square!'
    
    if titles is not None:
        assert num == len(titles)
    
    if num <= 3:
        for i in range(0, num):
            plt.subplot(1, num, 1 + i)
            plt.axis('off')
            if type(cmap) == list:
                assert len(cmap) == num
                plt.imshow(im[i], cmap=cmap[i]) # plot raw pixel data
            else:
                plt.imshow(im[i], cmap=cmap) # plot raw pixel data
            if titles is not None:
                plt.title(titles[i], fontsize=titles_fontsize)
    else:
        sq = int(np.sqrt(num))
        for i in range(0, num):
            if one_row:
                plt.subplot(1, num, 1 + i)
            else:
                plt.subplot(sq, sq, 1 + i)
            plt.axis('off')
            if type(cmap) == list:
                assert len(cmap) == num
                plt.imshow(im[i], cmap=cmap[i]) # plot raw pixel data
            else:
                plt.imshow(im[i], cmap=cmap) # plot raw pixel data
            if titles is not None:
                plt.title(titles[i], fontsize=titles_fontsize)

    if save:
        plt.savefig(out_fp)

    if show:
        plt.show()
    else:
        plt.close()



"""
## Plots an adversarial perturbation, i.e., original input x, adversarial example x_adv, and the difference (perturbation)
"""
def plot_adversarial_example(model, orig_x, adv_x, fname='adv_exp.png', show=True, save=True):
    perturb = adv_x - orig_x
    
    # compute confidence
    in_label, in_conf = nets.pred_label_and_conf(model, orig_x)
    
    # compute confidence
    adv_label, adv_conf = nets.pred_label_and_conf(model, adv_x)
    
    titles = ['Label: {} (conf: {:.3f})'.format(in_label, in_conf), 'Perturbation',
              'Label: {} (conf: {:.3f})'.format(adv_label, adv_conf)]
    
    images = np.r_[orig_x, perturb, adv_x]
    
    # plot images
    plot_images(images, dim_x=28, dim_y=28, fig_size=(8,3), titles=titles, titles_fontsize=12,  out_fp=fname, save=save, show=show)
    

"""
## Runs FGSMk
"""
def run_iterative_fgsm(model, x_in, y, eps, max_iter=100, conf=0.8):
    def done_fn_untargeted(model, x_in, x_adv, t, i, max_iter, conf):
        if i >= max_iter:
            return True
        y_v = model.predict(x_adv, verbose=0).reshape(-1)
        return np.argmax(y_v) != t and np.amax(y_v) >= conf
    
    terminate_fn = lambda m, x, xa, t, i: done_fn_untargeted(m, x, xa, t, i, max_iter, conf)
    x_adv, iters = attacks.iterative_fgsm(model, x_in, y, eps, terminate_fn)

    return x_adv
    


def done_fn_gn(model, x_in, x_adv, target, i, conf=0.8, max_iter=100):
    if i >= max_iter:
        return True
    
    y_pred_v = model.predict(x_adv, verbose=0)[0]
    y_pred = np.argmax(y_pred_v, axis=-1)
        
    return y_pred == target and y_pred_v[y_pred] >= conf


### Distortion function --- Fill in for problem 1 ###
"""
## Computes the distortion between x_in and x_adv
"""
def distortion(x_in, x_adv):
    ## TODO ##
    ## Insert your code here to calculate the distortion
    diff = x_adv - x_in
    mse = np.mean(np.square(diff))
    distortion = np.sqrt(mse)
    return distortion
    raise NotImplementedError()


### Random images --- Fill in for problem 2
def random_images(size=(1,28*28)):
    ## TODO ##
    ## Insert your code here
    return np.random.rand(*size)
    raise NotImplementedError()


### Craft 'num_adv_samples' adversarial examples
def craft_adversarial_fgsmk(model, x_aux, y_aux, num_adv_samples, eps):

    x_adv_samples = np.zeros((num_adv_samples, x_aux[0].shape[0]))
    correct_labels = np.zeros((num_adv_samples,))   
    
    avg_dist = 0.0
    sys.stdout.write('Crafting {} adversarial examples (untargeted FGSMk -- eps: {})'.format(num_adv_samples, eps))
    x_benign = None
    for i in range(0, num_adv_samples):
        ridx = np.random.randint(low=0, high=x_aux.shape[0])
    
        x_input = x_aux[ridx, :].reshape((1, -1))
        y_input = y_aux[ridx, :].reshape((1, -1))
        
        # keep track of the benign examples
        x_benign = x_input if x_benign is None else np.r_[x_benign, x_input]
        
        correct_labels[i] = np.argmax(y_input[0], axis=-1)
        
        x_adv = run_iterative_fgsm(model, x_input, correct_labels[i], eps)
        x_adv_samples[i,:] = x_adv
        
        avg_dist += distortion(x_input, x_adv)
        
        sys.stdout.write('.')
        sys.stdout.flush()
    print('Done.')
    
    avg_dist /= num_adv_samples

    return x_benign, correct_labels, x_adv_samples, avg_dist
  
  
def randomized_smoothing_predict_fn(model, x, sigma=1.0, num_samples=100, noise_type='gaussian'):

    y_pred_avg = np.zeros((x.shape[0], 10))    
    for i in range(0, num_samples):
        
        ## TODO --- implement randomized smoothing
        ## 1. Generate Gaussian noise with mean 0 and standard deviation sigma
        ## 2. Add this noise to 'x' and store the result in 'x_noisy'
        ## Note: you need to ensure that the shapes match because 'x' could either be a single image or several of images.
        # Generate noise
        if noise_type == 'gaussian':
            noise = np.random.normal(0, sigma, size=x.shape)
        elif noise_type == 'Laplace':
            noise = np.random.laplace(-sigma, sigma, size=x.shape)
        else:
            raise ValueError("Invalid noise type: %s" % noise_type)
        #raise NotImplementedError()
        x_noisy = x + noise 
        x_noisy_clipped = tf.clip_by_value(x_noisy, 0, 255.0) # clip

        y_pred = model.predict(x_noisy_clipped, verbose=0)
        assert y_pred.shape == y_pred_avg.shape
        
        y_pred_avg += y_pred
    
    y_pred_avg /= num_samples
    return y_pred_avg
        

## this is the main
def main():

    ######### Fill in your UFID here! ##############
    ufid = 70611190
    #for example: ufid = 12345678 # if your UFID is 1234-5678
    
    if ufid == 0 or ufid == 12345678:
        print('You must fill in your UFID first!')
        sys.exit(0)
       
        
    # set the seed for numpy and tensorflow based on the UFID
    np.random.seed(ufid)
    tf.random.set_seed(ufid)
    
    r = np.random.uniform() + tf.random.uniform((1,))[0]
    print('----- UFID: {} ; r: {:.6f}'.format(ufid, r))


    num_classes = 10 # mnist number of classes
    
    # figure out the problem number
    assert len(sys.argv) >= 3, 'Incorrect number of arguments!'
    p_split = sys.argv[1].split('problem')
    assert len(p_split) == 2 and p_split[0] == '', 'Invalid argument {}.'.format(sys.argv[1])
    problem_str = p_split[1]

    assert is_number(problem_str) is not None, 'Invalid argument {}.'.format(sys.argv[1])
    problem = float(problem_str)
    probno = int(problem)

    if probno < 0 or probno > 4:
        assert False, 'Problem {} is not a valid problem # for this assignment/homework!'.format(problem)

    ## change this line to override the verbosity behavior
    verb = True if probno == 0 else False

    # get arguments
    model_str = sys.argv[2]
    if model_str.startswith('simple'):
        simple_args = model_str.split(',')
        assert simple_args[0] == 'simple' and len(simple_args) == 3, '{} is not a valid network description string!'.format(model_str)
        hidden = is_int(simple_args[1])
        reg_const = is_number(simple_args[2])
        assert hidden is not None and hidden > 0 and reg_const is not None and reg_const >= 0.0, '{} is not a valid network description string!'.format(model_str)
        target_model_train_fn = lambda: nets.get_simple_classifier(num_hidden=hidden, l2_regularization_constant=reg_const,
                                                                   verbose=verb)
    elif model_str == 'deep':
        target_model_train_fn = lambda: nets.get_deeper_classifier(verbose=verb)
    else:
        assert False, '{} is not a valid network description string!'.format(model_str)

    # load the dataset
    train, test, aux = load_preprocess_mnist_data()
    x_train, y_train = train
    target_train_size = x_train.shape[0]
    x_test, y_test = test
    x_aux, y_aux = aux
    
    x_aux = x_aux.astype(float)
    y_aux = y_aux.astype(float)

    dirpath = os.path.join(os.getcwd(), 'models')
    ensure_exists(dirpath)
    base_fp = '{}/{}'.format(dirpath, model_str)

    ########################################################
    ########### Model Training (Problem 0) #################
    ########################################################
    train_model = probno == 0 # train
    if train_model:
        assert len(sys.argv) == 4, 'Incorrect number of arguments!'
        model = target_model_train_fn()  # compile the target model
        
        num_epochs = is_int(sys.argv[3])

        assert num_epochs is not None and 0 < num_epochs <= 10000, '{} is not a valid size for the number of epochs to train the target model!'.format(sys.argv[3])

        # train the target model
        train_loss, train_accuracy, test_loss, test_accuracy = nets.train_model(model, x_train, y_train, x_test, y_test, num_epochs, verbose=verb)

        print('Trained target model on {} records. Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size,
                                                                                    100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))

        save_model(model, base_fp)
    else:
        model, _ = load_model(base_fp)
        if model is None:
            print('Model files do not exist. Train the model first!'.format(base_fp))
            sys.exit(1)


    ## make sure the target model is not trainable (so we don't accidentally change the weights)
    model.trainable = False

    target_conf = 0.8    

    ########################################################
    ##################### Problem 1 ########################
    ########################################################
    if probno == 1: 
        assert len(sys.argv) == 5, 'Incorrect number of arguments!'

        input_idx = is_int(sys.argv[3])
        assert 0 <= input_idx <= x_aux.shape[0], 'Invalid input index (must be between 0 and {})!'.format(x_aux.shape[0])
        alpha = is_int(sys.argv[4])
        assert 1 <= alpha <= 50, 'Invalid alpha!'

        max_iter = 100      # maximum number of iterations for the attack

        x_input = x_aux[input_idx, :].reshape((1, -1))

        y_input_lab = y_aux[input_idx, :]
        y_true_lab = np.argmax(y_input_lab, axis=-1)

        y_pred = model.predict(x_input, verbose=0).reshape(-1)
        y_pred_lab = np.argmax(y_pred, axis=-1)

        print('\nSelected input {} -- true label: {}, predicted label: {} (confidence: {:.2f}%)'.format(
                    input_idx, y_true_lab, y_pred_lab, 100.0*float(y_pred[y_pred_lab])))

        terminate_fn = lambda m, x, xa, t, i: done_fn_gn(m, x, xa, t, i, conf=target_conf, max_iter=max_iter)

        t = time.process_time()
        print('\nRunning the gradient noise attack (<={} iterations).'.format(max_iter))
        x_adv, iters = attacks.gradient_noise_attack(model, x_input, y_input_lab, max_iter, terminate_fn=terminate_fn, alpha=alpha)
        elapsed_time = time.process_time() - t

        y_adv = model.predict(x_adv, verbose=0).reshape(-1)
        y_label = np.argmax(y_adv, axis=-1)

        status = '\tAttack failed'
        if iters < max_iter: 
            status = '\tAttack successful ({} iterations - {:.1f} seconds)'.format(iters, elapsed_time)

        print('{}, the adversarial example is classified as \'{}\' with {:.2f}% confidence by the model!'.format(status,  y_label, 100.0 * float(y_adv[y_label])))

        print('\tDistortion: {:.2f}'.format(distortion(x_input, x_adv)))
        plot_adversarial_example(model, x_input, x_adv, show=True, fname='gradient_noise')


    ########################################################
    ##################### Problem 2 ########################
    ########################################################
    elif probno == 2:

        assert len(sys.argv) == 3, 'Incorrect number of arguments!'

        # distribution of predictions for random images of the model
        num_samples = 1000
        predictions = np.zeros((10,))
        
        x_rand = random_images(size=(num_samples,28*28))
        y_pred = model.predict(x_rand, verbose=0)
        y_pred_lab = np.argmax(y_pred, axis=-1)
        
        for i in range(0, num_classes):
            predictions[i] = np.sum(y_pred_lab == i)
            
        fig = plt.figure()
        x = np.arange(0,10)+0.5
        plt.bar(x, (predictions / np.sum(predictions))*100.0)
        plt.ylabel('Percentage of random images')
        plt.xlabel('Class label')
        plt.xticks(ticks=x, labels=range(10))
        plt.show()
        
        # save it to file in case show() does not show the figure
        plt.savefig('distribution.png') 


    ########################################################
    ##################### Problem 3 ########################
    ########################################################
    elif probno == 3: 
        
        assert len(sys.argv) == 5, 'Incorrect number of arguments!'
        
        num_adv_samples = is_int(sys.argv[3])
        assert 1 < num_adv_samples <= x_aux.shape[0], 'Invalid number of samples (must be between 2 and {})!'.format(x_aux.shape[0])
        
        eps = is_number(sys.argv[4])
        eps = int(eps) # make integer
        assert 1 <= eps < 128.0, 'Invalid eps value.'
        
        samples_fp = 'fgsmk_samples_eps{}.npz'.format(eps)

        x_benign, correct_labels, x_adv_samples, avg_dist = craft_adversarial_fgsmk(model, x_aux, y_aux, num_adv_samples, eps)
        
        np.savez_compressed(samples_fp, x_benign=x_benign, correct_labels=correct_labels, x_adv_samples=x_adv_samples)
        
        

        # prediction function for evaluate_attack()
        model_predict_fn = lambda x: model.predict(x, verbose=0)
        
        benign_acc, adv_acc = attacks.evaluate_attack(model_predict_fn, x_benign, correct_labels, x_adv_samples)
        print('Untargeted FGSM attack eval --- benign acc: {:.1f}%, adv acc: {:.1f}% [eps={:.1f}, mean distortion: {:.3f}]\n'.format(benign_acc*100.0, adv_acc*100.0, eps, avg_dist))
        
        
    ########################################################
    ##################### Problem 4 ########################
    ########################################################
    elif probno == 4: 

        assert len(sys.argv) == 5 or len(sys.argv) == 6, 'Incorrect number of arguments!'
        
        eps = is_number(sys.argv[3])
        assert 1 <= eps < 128.0, 'Invalid eps value.'      

        
        samples_fp = 'fgsmk_samples_eps{}.npz'.format(eps)
        
        # load the data
        data = np.load(samples_fp)
        x_benign = data['x_benign']
        correct_labels = data['correct_labels']
        x_adv_samples = data['x_adv_samples']

        sigma_str = sys.argv[4]
        sigmas = [is_number(s) for s in sigma_str.split(',')]
        for sigma in sigmas:
            assert 0.0 <= sigma <= 1000.0, 'Invalid sigma value {}.'.format(sigma)
        
        noise_type = 'gaussian'
        if len(sys.argv) == 6:
            noise_type = sys.argv[5]
        
        benign_accs = []
        adv_accs = []
        for sigma in sigmas:
            if sigma == 0:
                # prediction function for evaluate_attack()
                model_predict_fn = lambda x: model.predict(x, verbose=0)
                
                benign_acc, adv_acc = attacks.evaluate_attack(model_predict_fn, x_benign, correct_labels, x_adv_samples)
                print('Untargeted FGSM attack eval --- benign acc: {:.1f}%, adv acc: {:.1f}% [eps={:.1f}]\n'.format(benign_acc*100.0, adv_acc*100.0, eps))
            
            else:
                # prediction function for evaluate_attack()
                model_predict_fn = lambda x: randomized_smoothing_predict_fn(model, x, sigma=sigma, num_samples=100, noise_type=noise_type)
                
                benign_acc, adv_acc = attacks.evaluate_attack(model_predict_fn, x_benign, correct_labels, x_adv_samples)
                print('[RS] Untargeted FGSM attack eval --- benign acc: {:.1f}%, adv acc: {:.1f}% [eps={:.1f}, sigma: {}, noise: {}]'.format(benign_acc*100.0, adv_acc*100.0, eps, sigma, noise_type))
                
            # keep track of this for potential later use
            benign_accs.append(benign_acc)
            adv_accs.append(adv_acc)
        

        print(benign_accs, adv_accs)

        ## TODO ##
        ## Insert your code here (for plotting a figure or outputting a table for problem 4.2)
        igmas = [0] + [is_number(s) for s in sigma_str.split(',')]
        plt.plot(sigmas, benign_accs, label='benign accuracy')
        plt.plot(sigmas, adv_accs, label='adversarial accuracy')
        plt.title('Adversarial and Benign Accuracy vs Sigma')
        plt.xlabel('sigma')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('graph1.png')
       # raise NotImplementedError()
        
        
    elif probno == 5:  ## problem 5 (bonus)

        ## TODO ##
        ## Insert your code here
        raise NotImplementedError()


if __name__ == '__main__':
    main()
