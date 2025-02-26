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
# Author: Jacky Choi, ETH Zurich
# Author: Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
# Author: Maximilian Grözinger, ETH Zurich


import os
import time
import json
import math
import datetime
import csv
import sys
import argparse
import sys
import torch

from dataset import DatasetProcessor
from datagenerator import DatasetCreator
from train import Train
from architectures.dscnn import DSCNNS, DSCNNM, DSCNNL
from utils import parameter_generation

from torch.utils.data import DataLoader
from torchsummary import summary
from copy import deepcopy
from sklearn.metrics import confusion_matrix

import torch.nn.functional as F
import soundfile as sf
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def main():

    date = datetime.datetime.now()
    date_str = date.strftime("%Y_%m_%d-%p%I_%M_%S")

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--feature_bin_count', type=int, default=None, help="Selected mels") 
    parser.add_argument('--time_shift_ms', type=int, default=None, help="Shifting for augmentation") 
    parser.add_argument('--sample_rate', type=int, default=None, help="Sample rate") 
    parser.add_argument('--clip_duration_ms', type=int, default=None, help="Input length") 
    parser.add_argument('--window_size_ms', type=int, default=None, help="Window len") 
    parser.add_argument('--window_stride_ms', type=int, default=None, help="Window hop")
    parser.add_argument('--n_mels', type=int, default=None, help="Number of mels")
    parser.add_argument('--seed', type=int, default=None, help="Seed during training time")
         
    parser.add_argument('--device_id', type=str, default=None, help = 'Select GPU for use')
    parser.add_argument('--noise_dataset', type=str, default=None, help = "Noise dataset")
    parser.add_argument('--noise_dir_demand', type=str, default=None, help = "DEMAND noise path")
    parser.add_argument('--noise_dir_gscv2', type=str, default=None, help = "GSC noise path")
    parser.add_argument('--data_url', type=str, default=None, help = "Data URL")
    parser.add_argument('--keywords_dataset', type=str, default=None, help = "Keywords dataset")
    parser.add_argument('--data_dir_gscv2', type=str, default=None, help = "GSC noise path")
    parser.add_argument('--data_dir_mswc', type=str, default=None, help = "MSWC noise path")
    parser.add_argument('--data_dir_kinem', type=str, default=None, help = "KINEM noise path")
    parser.add_argument('--denoisify', type=int, default=None, help = 'Apply denoiser or not, and select the denoiser.')
    parser.add_argument('--reverb', type=str, default=None, help = 'Apply reverb')

    parser.add_argument('--noise_mode', type=str, default=None, help = 'noiseless, noiseaware, odda')
    parser.add_argument('--snr_range', nargs="*", type=int, default=None, help = 'SNR range, passed as list')
    parser.add_argument('--model', type=str, default=None, help = 'DSCNN')
    parser.add_argument('--channels', type=int, default=None, help = '64/172/276')
    parser.add_argument('--blocks', type=int, default=None, help = '4/4/5')
    parser.add_argument('--use_pretrained', type=int, default=None, help = 'Use pretrained model')
    parser.add_argument('--pretrained_directory', type=str, default=None, help = 'Location of pretrained model')
    parser.add_argument('--epochs', type=int, default=None, help = 'Number of epochs')
    parser.add_argument('--batch size', type=int, default=None, help = 'Batch size')
    parser.add_argument('--loss', type=str, default=None, help = 'Loss type')
    parser.add_argument('--initial_lr', type=float, default=None, help = 'Initial learning rate')
    parser.add_argument('--optimizer', type=str, default=None, help = 'Optimizer')
    parser.add_argument('--momentum', type=int, default=None, help = 'Momentum')
    parser.add_argument('--silence_percentage', type=int, default=None, help = 'Percentage of silence in the dataset')
    parser.add_argument('--unknown_percentage', type=int, default=None, help = 'Percentage of unknown in the dataset')
    parser.add_argument('--validation_percentage', type=int, default=None, help = 'Percentage of validation data in the dataset')
    parser.add_argument('--testing_percentage', type=int, default=None, help = 'Percentage of test data in the dataset')
    parser.add_argument('--background_frequency', type=int, default=None, help = 'Frequency of adding background noise. Default is 1 in ODDA.')
    parser.add_argument('--background_volume', type=int, default=None, help = 'Background noise volume. Argument is ignored if snr_range is set')
    parser.add_argument('--task', type=str, default=None, help='Selected task')
    parser.add_argument('--wanted_words_gscv2_12w', nargs="*", type=str, default=None, help='List of GSC12 words')
    parser.add_argument('--wanted_words_gscv2_35w', nargs="*", type=str, default=None, help='List of GSC35 words')
    parser.add_argument('--wanted_frequency_mswc', type=str, default=None, help='Threshold in samples/class to select MSWC words')
    parser.add_argument('--trainable', default=None, help='finetuning layers of network, other layers are frozen (full, classifier, embeddings)')
    parser.add_argument('--learn', type=str, default=None, help = 'Users, Noises, or both(Users_Noises)')
    parser.add_argument('--batch_size', type=int, default=None, help = 'Batch Size for Pretraining/Finetuning')
    parser.add_argument('--online_noise_train', nargs="*", default=None, help = 'Target noise to finetune on')
    parser.add_argument('--online_noise_test', nargs="*", default=None, help = 'Target noise to test finetune on')

    parser.add_argument('--pretrain', type=int, default=None, help='Perform (pre)training model')
    parser.add_argument('--fold', type=int, default = None, help='Data fold for test shifting')
    parser.add_argument('--metatrain', type=int, default=None, help='Perform metatraining for the model')
    parser.add_argument('--select', type=int, default=None, help='Perform data selection for ODDA')
    parser.add_argument('--finetune', type=int, default=None, help='Perform ODDA')
    parser.add_argument('--quantize', type=int, default=None, help='Quantize model. Not implemented.')
    parser.add_argument('--evaluate', type=int, default=None, help='Evaluate pretrained model.')

    parser.add_argument('--selection_method', type=str, default = None, help = 'Data selection method')
    parser.add_argument('--selection_interval_upper', type=int, default = None, help= 'Data selection interval upper bound')
    parser.add_argument('--selection_interval_lower', type=int, default = None, help= 'Data selection interval lower bound')

    parser.add_argument('--noise_train', nargs="*", type=str, default=None, help = 'List of noises on which we train the net: "DKITCHEN", "DLIVING"...')
    parser.add_argument('--noise_test', nargs="*", type=str, default=None, help = 'List of noises on which we test the net: "DKITCHEN", "DLIVING"....')
    parser.add_argument('--target_noise', type=str, default = None, help = 'ODDA on-site noise, select from the list above')
    parser.add_argument('--distance', type=str, default = None, help = 'Select microphone-to-subject distance. N(ear)/F(ar).')
    parser.add_argument('--microphone', type=str, default = None, help = 'Select microphone quality. L(ow-power)/P(hone)/R(hode).')

    parser.add_argument('--base_path', type=str, default = None, help='Path to current directory')
    parser.add_argument('--model_path', type=str, default = None, help='Path where the model will be saved')

    parser.add_argument('--embeddings', type=int, default = None, help='Use environment embeddings')
    parser.add_argument('--emb_format', type=str, default = None, help='Choose embedding format')

    parser.add_argument('--config_file', type=str, default=None, help = 'Configuration file')
    
    parser.add_argument('--evaluation_log_path', type=str, default = None, help='File to log finetuning results')

    args = vars(parser.parse_args())

    # Parameter generation
    environment_parameters, preprocessing_parameters, training_parameters, experimental_parameters, architecture_parameters = parameter_generation(args) 


    torch.manual_seed(environment_parameters['seed'])

    # Device setup
    if environment_parameters["device"] == "gpu" and torch.cuda.is_available():
        device = torch.device(f'cuda:{environment_parameters["device_id"]}')
    else:
        device = torch.device('cpu')
    print (torch.version.__version__)
    print(device)

            
    audio_processor = DatasetCreator(environment_parameters, training_parameters, preprocessing_parameters, experimental_parameters)

    train_size = audio_processor.get_size('training')
    val_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    pretrain_size = audio_processor.get_size('pretrain')
    preval_size = audio_processor.get_size('preval')
    pretest_size = audio_processor.get_size('pretest')

    n_classes = len(audio_processor.words_list)
    print("Pretrain dataset split (Train/Val/Test): " + str(pretrain_size) + "/" + str(preval_size) + "/" + str(pretest_size))
    print("Finetune Dataset split (Train/Val/Test): " + str(train_size) + "/" + str(val_size) + "/" + str(test_size))


    if (experimental_parameters['learn'] =='users'):
        n_user_pretrain_envs = audio_processor.get_user_number_pretrain()
        n_user_finetune_envs = audio_processor.get_user_number_finetune()
        n_noise_pretrain_envs = 0
        n_noise_finetune_envs = 0
    elif (experimental_parameters['learn'] =='noises'):
        n_user_pretrain_envs = audio_processor.get_noise_number_pretrain()
        n_user_finetune_envs = audio_processor.get_noise_number_finetune()
        n_noise_pretrain_envs = 0
        n_noise_finetune_envs = 0
    elif (experimental_parameters["learn"] == "users_noises"):
        n_user_pretrain_envs = audio_processor.get_user_number_pretrain()
        n_user_finetune_envs = audio_processor.get_user_number_finetune()
        n_noise_pretrain_envs = audio_processor.get_noise_number_pretrain()
        n_noise_finetune_envs = audio_processor.get_noise_number_finetune()      

    # Model generation
    if (architecture_parameters['model'].startswith("DSCNN")):
        if architecture_parameters['embeddings']:
            net = getattr(sys.modules["architectures.dscnn"], architecture_parameters['model'])(n_classes=n_classes, use_bias=False, emb_format=architecture_parameters['emb_format'],\
                                        n_user_envs = n_user_pretrain_envs + n_user_finetune_envs, n_noise_envs = n_noise_pretrain_envs + n_noise_finetune_envs)
        else:
            net = getattr(sys.modules["architectures.dscnn"], architecture_parameters['model'])(n_classes=n_classes, use_bias=True, emb_format=architecture_parameters['emb_format'],\
                                        n_user_envs = 0)
    elif (architecture_parameters['model'].startswith("GCN")):
        if architecture_parameters['embeddings']:
            net = getattr(sys.modules["architectures.gcn"], architecture_parameters['model'])(n_classes=n_classes, emb_format=architecture_parameters['emb_format'],\
                                        n_user_envs = n_user_pretrain_envs + n_user_finetune_envs)
        else:
            net = getattr(sys.modules["architectures.gcn"], architecture_parameters['model'])(n_classes=n_classes, emb_format=architecture_parameters['emb_format'], n_user_envs = 0)
    
    if architecture_parameters["use_pretrained"]:
        print("Loading model state dict")
        net.load_state_dict(torch.load(architecture_parameters['pretrained_directory']+'model.pth'))
    
    net.to(device)
    if (experimental_parameters['pretrain']):
        logmode = 'pretrain'
    elif (experimental_parameters['finetune']):
        logmode = 'finetune'
    else: 
        logmode = "undefined_test"
    train_log_path = experimental_parameters['model_path'] + '/'+logmode+'_emb'+str(architecture_parameters['embeddings'])+\
    '_'+architecture_parameters['emb_format']+'_model_'+architecture_parameters['model'] + '_' + date_str

    config_log = os.path.join(train_log_path, "log_config.txt")
    os.makedirs(train_log_path, exist_ok=True)
    with open(config_log, "w") as f:
        f.write(str(environment_parameters))
        f.write("\n")
        f.write(str(preprocessing_parameters))
        f.write("\n")
        f.write(str(training_parameters))
        f.write("\n")
        f.write(str(experimental_parameters))
        f.write("\n")
        f.write(str(architecture_parameters))
                        
    training_environment = Train(audio_processor, training_parameters, architecture_parameters, experimental_parameters, net, device, train_log_path)

    if experimental_parameters['pretrain']:
        # Accuracy on the testing set. 
        print ("Test accuracy before training")
        acc = training_environment.validate(net, mode='pretest', statistics=False)
        print ("Train")
        training_environment.train(net, mode='pretrain') 
        # Accuracy on the testing set. 
        print ("Test accuracy after training")
        acc = training_environment.validate(net, mode='pretest', statistics=False)

    trainable = str(architecture_parameters['trainable'])
    
    if (experimental_parameters['finetune']):
        
        print(f"Finetuning {architecture_parameters['trainable']} parts of the model")

        if not "full" in architecture_parameters["trainable"]:
            for param in net.parameters():
                param.requires_grad = False
            
            if "classifier" in architecture_parameters["trainable"]:
                print("Enable classifier finetuning")
                for param in net.fc1.parameters():
                    param.requires_grad = True
                net.fc1.requires_grad = True
            if "embeddings" in architecture_parameters["trainable"]:
                print("Enable embeddings finetuning")
                for param in net.emb.parameters():
                    param.requires_grad = True
                if experimental_parameters["learn"] == "users_noises":
                    for param in net.emb2.parameters():
                        param.requires_grad = True                    
                net.emb.requires_grad = True

        print ("PreTest accuracy")
        pretest_acc = training_environment.validate(net, mode='pretest', statistics=False)
        print ("Finetune Test accuracy before adaption")
        before_training_acc = training_environment.validate(net, mode='testing', statistics=False)
        print ("Adaption")
        training_environment.train(net, mode='training') 
        # Accuracy on the testing set. 
        print ("Finetune Test accuracy after adaption")
        after_training_acc = training_environment.validate(net, mode='testing', statistics=False)

    if (experimental_parameters['evaluate']):
        print ("PreTest accuracy")
        pretest_acc = training_environment.validate(net, mode='pretest', statistics=False)
        print ("Finetune Test accuracy before adaption")
        before_training_acc = training_environment.validate(net, mode='testing', statistics=False)
        # Accuracy on the testing set. 
        print ("Finetune Test accuracy after adaption")
        after_training_acc = training_environment.validate(net, mode='testing', statistics=False)

    log_file_path = experimental_parameters["evaluation_log_path"]
    if log_file_path != "":
        append_accuracies(log_file_path, trainable, pretest_acc, before_training_acc, after_training_acc, training_parameters, experimental_parameters)


# Function to append new accuracy values
def append_accuracies(file_path, trainable, pretest_acc, before_training, after_training, training_parameters, experimental_parameters):
    # Read the existing CSV file
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame(columns=["Trainable", "PreTestAccuracy", "LearningRate", "Batch_size", "Fold", "Accuracy Before Training", "Accuracy After Training"])

    # Append the new accuracies to the DataFrame
    new_data = {"Trainable": trainable, "PreTestAccuracy": pretest_acc, "LearningRate": training_parameters["initial_lr"], "Batch_size": training_parameters["batch_size"], "Fold": experimental_parameters["fold"], "Accuracy Before Training": before_training, "Accuracy After Training": after_training}
    df = df.append(new_data, ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)
    print(f"New accuracies appended successfully to {file_path}")


if __name__ == "__main__":
    main()



