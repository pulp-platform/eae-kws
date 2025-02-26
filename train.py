# Copyright (C) 2021-2024 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Author: Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
# Author: Maximilian Grözinger, ETH Zurich


import csv 
import math
import os
import random
import time
import torch

from copy import deepcopy
from collections import Counter
from datetime import datetime
from sklearn import manifold
from scipy.stats import entropy
from torch.utils.data import DataLoader

from dataset import DatasetProcessor
from utils import conf_matrix, npy_to_txt, per_noise_accuracy, save_histogram

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class Train():

    def __init__(self, audio_processor, training_parameters, architecture_parameters, experimental_parameters, model, device, log_path):
        self.audio_processor = audio_processor
        self.training_parameters = training_parameters
        self.architecture_parameters = architecture_parameters
        self.experimental_parameters = experimental_parameters
        self.model = model
        self.device = device

        # Training hyperparameters
        if (training_parameters['loss'] == "crossentropy"):
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise AttributeError("Loss " + training_parameters['loss'] + "is not supported")
        initial_lr = training_parameters['initial_lr']  # 0.0001 - ODDA; 0.001 - NA/NL-KWS
        self.initial_lr = initial_lr
        if (training_parameters['optimizer'] == 'adam'):
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = initial_lr)
        elif(training_parameters['optimizer'] == 'sgd'):
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr, momentum=training_parameters['momentum'])

        lambda_lr = lambda epoch: 1 if epoch<15 else 1/5 if epoch < 25 else 1/10 if epoch<35 else 1/20  # 100%
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda_lr)

        self.log_path = log_path


    def validate(self, model = None, mode='validation', batch_size = 0, statistics = False, integer = False, save = False, task = None):
        # Validate model

        training_parameters = self.training_parameters
        if (batch_size != 0):
            training_parameters['batch_size'] = batch_size   

        mdataset = DatasetProcessor(mode, self.audio_processor, training_parameters, task = task, device = self.device)
        mdataloader = DataLoader(mdataset, batch_size=training_parameters['batch_size'],
                        shuffle=False, num_workers=0)

        start=time.clock_gettime(0)        
        model.eval()  

        correct = 0
        total = 0

        with torch.no_grad():

            for minibatch, sample_batched in enumerate(mdataloader):
                if (self.experimental_parameters["learn"] == "users_noises"):
                    inputs, labels, ids, second_ids = sample_batched
                    second_ids = second_ids.to(self.device)
                else:
                    inputs, labels, ids = sample_batched
            
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ids = ids.to(self.device)

                # Train, compute loss, update optimizer
                if self.architecture_parameters['embeddings']:
                    if (self.experimental_parameters["learn"] == "users_noises"):
                        outputs = model(inputs, ids, second_ids)
                    else:
                        outputs = model(inputs, ids)
                else:
                    outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if minibatch % 20 == 0: 
                    print('[%3d / %3d] accuracy: %.3f' % (minibatch + 1, len(mdataloader),  100 * correct / total))
                    running_loss = 0.0
                

                if statistics == True:
                    conf_matrix(labels, predicted, self.training_parameters)
                    per_noise_accuracy(labels, predicted, noises)

                if (batch_size == 64): # patch to allow subset validation
                    break

        print('Accuracy of the network on the %s set: %.2f %%' % (mode, 100 * correct / total))
        return(100 * correct / total)


    def train(self, model, mode ='training'):
        # Train model
        mdataset = DatasetProcessor(mode, self.audio_processor, self.training_parameters, task = None, device = self.device)
        mdataloader = DataLoader(mdataset, batch_size=self.training_parameters['batch_size'],
                        shuffle=False, num_workers=0)

        best_acc = 0
        best_ep = 0
        best_model = None

        FILEPATH = self.log_path
        os.makedirs(FILEPATH, exist_ok=True)
        os.path.isfile(FILEPATH)

        header = ['epoch','train_acc','train_loss','val_acc']
        with open(FILEPATH+'/log.csv','w') as csvfile:    
            writer = csv.writer(csvfile, delimiter=',')
            # Gives the header name row into csv
            writer.writerow([h for h in header])   

        for epoch in range(0, self.training_parameters['epochs']):
            print("Epoch: " + str(epoch+1) +"/" + str(self.training_parameters['epochs']))
            start=time.clock_gettime(0)

            if self.training_parameters['batchnorm']:
                model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            for minibatch, sample_batched in enumerate(mdataloader):
                if (self.experimental_parameters["learn"] == "users_noises"):
                    inputs, labels, ids, second_ids = sample_batched
                    second_ids = second_ids.to(self.device)
                else:
                    inputs, labels, ids = sample_batched

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ids = ids.to(self.device)

                # Zero out the parameter gradients after each mini-batch
                self.optimizer.zero_grad()

                # Train, compute loss, update optimizer
                if self.architecture_parameters['embeddings']:
                    if (self.experimental_parameters["learn"] == "users_noises"):
                        outputs = model(inputs, ids, second_ids)
                    else:
                        outputs = model(inputs, ids)
                else:
                    outputs = model(inputs)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Compute training statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print information every 20 minibatches
                if minibatch % 10 == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, len(mdataloader), running_loss / 10, 100 * correct / total))
                    running_loss = 0.0


            print('Finished Training 1 epoch on GPU in {:.2f} seconds'.format(time.clock_gettime(0)-start))

            if (mode == 'training'):
                tmp_acc = self.validate(model, 'validation')
            else:  
                tmp_acc = self.validate(model, 'preval')

            statistics=[str(epoch), str(100 * correct / total), str(running_loss / 10), str(tmp_acc)]
            with open(FILEPATH+'/log.csv', 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(statistics)

            # Saving each model for MSWC analysis
            PATH = FILEPATH + '/acc_' + str(tmp_acc) + "_ep_" + str(epoch) + '.pth'
            torch.save(model.state_dict(), PATH)

            # Save best performing network
            if (tmp_acc > best_acc):
                best_acc = tmp_acc
                best_ep = epoch
                best_model = model
                PATH = FILEPATH + '/acc_' + str(best_acc) + "_ep_" + str(best_ep) + '.pth'
                torch.save(model.state_dict(), PATH)

            # Save with Early Exit
            if (epoch >= best_ep + 10):
                break
        PATH = FILEPATH +'/model.pth'
        torch.save(best_model.state_dict(), PATH)
        model = best_model
