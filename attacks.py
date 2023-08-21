# -*- coding: utf-8 -*-
""" CIS6261TML -- Homework 1 -- attacks.py

# This file contains the attacks
"""

import os
import sys

import numpy as np

import tensorflow as tf

import scipy.stats as stats

# our neural network architectures
import nets
import attacks

"""
## Gradient Noise Attack. Returns the adversarial perturbation.
## How does this attack work????
## The attack runs until the termination condition or the maximum number of iterations is reached (whichever occurs first)
"""
def gradient_noise_attack(model, x_input, y_input, max_iter, terminate_fn, alpha=5, sf=1e-12):
    x_in = tf.convert_to_tensor(x_input, dtype=tf.float32)
    x_adv = x_in                        # initial adversarial example
    y_flat = np.argmax(y_input)

    for i in range(0, max_iter):

        # grab the gradient of the loss (given y_input) with respect to the input!
        grad_vec = nets.gradient_of_loss_wrt_input(model, x_adv, y_input)
        
        ### why might the following two lines be a good idea?
        if np.sum(np.abs(grad_vec)) < sf:
            grad_vec = 0.0001 * tf.random.normal(grad_vec.shape) 
        
        # create perturbation
        r = tf.random.uniform(grad_vec.shape)
        perturb = alpha * r * tf.sign(grad_vec)
        
        # add perturbation
        x_adv = x_adv + perturb

        iters = i+1 # save the number of iterations so we can return it
        
        x_adv = tf.clip_by_value(x_adv, 0, 255.0)
        
        # set the most likely incorrect label as target
        y_pred = model(x_adv)[0].numpy()
        y_pred[y_flat] = 0
        target_class_number = np.argmax(y_pred, axis=-1)

        # check if we should stop the attack early
        if terminate_fn(model, x_in, x_adv, target_class_number, iters):
            break

    return x_adv.numpy().astype(int), iters
    

"""
## Fast Gradient Sign Method (FGSM) for untargetted perturbations
##
"""
def do_untargeted_fgsm(model, in_x, y, alpha):

    grad_vec = nets.gradient_of_loss_wrt_input(model, in_x, y)
    
    ## TODO
    ## Insert your code here (~3 lines)
    ## 1. Calculate a perturbation (using the sign of the gradient) to increase the loss on 'in_x' for the *true* label
    ## 2. Scale the perturbation by alpha
    ## 3. Add the perturbation to the input image
    ## 4. Finally clip the adversarial example so it results in a valid image ('adv_x') [hint: you can use tf.clip_by_value()]
     # compute the perturbation as the sign of the gradient
    perturbation = tf.sign(grad_vec)

    # scale the perturbation by alpha
    scaled_perturbation = alpha * perturbation

    # add the perturbation to the input image
    adv_x = in_x + scaled_perturbation

    # clip the adversarial example to ensure it is a valid image
    adv_x = tf.clip_by_value(adv_x, 0.0, 255.0)
    
    #raise NotImplementedError()
    
    return adv_x ## the adversarial example
    
    
"""
## Iterative Fast Gradient Sign Method (FGSM) for untargetted perturbations
## 'stop_fn' is a caller-defined function that returns True when the attack should terminate
## The output is the adversarial example and the number of iterations performed
"""
def iterative_fgsm(model, in_x, y, eps, terminate_fn, num_classes=10):
    adv_x =  tf.convert_to_tensor(in_x, dtype=tf.float32)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes)
    
    minv = np.maximum(in_x.astype(float) - eps, 0.0)
    maxv = np.minimum(in_x.astype(float) + eps, 255.0)
    
    alpha = 8 # step magnitude
    
    i = 0
    while True:
        # do one step of FGSM
        adv_x = do_untargeted_fgsm(model, adv_x, y_onehot, alpha)
        
        # clip to ensure we stay within an epsilon radius of the input
        adv_x = tf.clip_by_value(adv_x, clip_value_min=minv, clip_value_max=maxv)

        # check if predicted label is the target
        adv_label, adv_conf = nets.pred_label_and_conf(model, adv_x)
        
        i += 1

        # call the stop function and exit if needed
        if terminate_fn(model, in_x, adv_x, y, i):
            break
        
    return adv_x.numpy().astype(int), i
    
   
   
"""
## Evaluates an adversarial example attack 
## 'model_predict_fn': a prediction function for the target model
## 'x_benign': benign samples with associate labels 'y_true_labels'
## 'y_true_labels': the labels of benign samples
## 'x_adv_samples': the adversarial examples produced for each of the benign samples
## 
## The output is a tuple of (benign accuracy, adversarial accuracy)
"""
def evaluate_attack(model_predict_fn, x_benign, y_true_labels, x_adv_samples):

    assert x_benign.shape[0] == y_true_labels.shape[0]
    assert x_benign.shape == x_adv_samples.shape
    
    y_true_labels = y_true_labels.astype(int)
        
    benign_preds = model_predict_fn(x_benign)
    y_preds = np.argmax(benign_preds, axis=-1)
    
    benign_accuracy = np.mean((y_true_labels == y_preds).astype(int))
    
    adv_preds = model_predict_fn(x_adv_samples)
    y_preds = np.argmax(adv_preds, axis=-1)
    adv_accuracy = np.mean((y_true_labels == y_preds).astype(int))
    
    return benign_accuracy, adv_accuracy
