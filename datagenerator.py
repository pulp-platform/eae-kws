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


import glob
import hashlib
import math
import os
import random
import re
import torch

import numpy as np
import soundfile as sf
import pandas as pd
from collections import defaultdict

from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path

class DatasetCreator(object):
    
    # Prepare data
    def __init__(self, environment_parameters, training_parameters, preprocessing_parameters, experimental_parameters):
            self.random_seed = 59185 

            # Set environment variables
            self.max_num_wavs_per_class = 2**27 - 1    # ~134M
            self.background_noise_label = '_background_noise_'
            self.silence_label = '_silence_'
            self.silence_index = 0
            self.unknown_word_label = '_unknown_'
            self.unknown_word_index = 1
            

            self.environment_parameters = environment_parameters
            self.training_parameters = training_parameters
            self.preprocessing_parameters = preprocessing_parameters
            self.experimental_parameters = experimental_parameters

            # Augmentation
            if (self.preprocessing_parameters['noise_mode'] == 'noiseaware'):
                self.generate_background_noise()
            if (self.preprocessing_parameters['reverb']):
                self.generate_reverberant_rooms()

            # Class management
            self.prepare_words_list()
            self.generate_data_dictionary()
            self.curate_words_list()


    def sample_selection_naive(self):

        if (self.experimental_parameters['fixnr']):
            self.experimental_parameters['samples/test/user/word']=1
            self.experimental_parameters['samples/val/user/word']=1
            self.experimental_parameters['samples/train/user/word']=self.experimental_parameters['utterances']
        else:
            self.experimental_parameters['samples/test/user/word']=-1
            self.experimental_parameters['samples/val/user/word']=-1
            self.experimental_parameters['samples/train/user/word']=-1


    def user_selection(self):
    
        if (self.experimental_parameters['utterances'] == 0):
            self.training_parameters['sessions'] = 0
            csv_filenumber = 0
        if (self.experimental_parameters['utterances'] == 1):
            self.training_parameters['sessions'] = 361
            csv_filenumber = 3
        elif (self.experimental_parameters['utterances'] == 2):
            self.training_parameters['sessions'] = 174
            csv_filenumber = 4
        elif (self.experimental_parameters['utterances'] == 3):
            self.training_parameters['sessions'] = 36
            csv_filenumber = 5
        elif (self.experimental_parameters['utterances'] == 4):
            self.training_parameters['sessions'] = 6
            csv_filenumber = 6
        elif (self.experimental_parameters['utterances'] == 5):
            self.training_parameters['sessions'] = 2
            csv_filenumber = 7
        elif (self.experimental_parameters['utterances'] == 7):
            self.training_parameters['sessions'] = 1
            csv_filenumber = 9
        else:
            self.training_parameters['sessions'] = 1


        if (self.experimental_parameters['fixnr']):
            self.experimental_parameters['samples/test/user/word']=1
            self.experimental_parameters['samples/val/user/word']=1
            self.experimental_parameters['samples/train/user/word']=self.experimental_parameters['utterances']
            self.experimental_parameters['samples/pretrain/train/user/word']=-1
            self.experimental_parameters['samples/pretest/train/user/word']=1
            file_finetune = 'dataset/dataset_simplified_individual_'+str(csv_filenumber)+'.csv'
            file_pretrain = 'dataset/dataset_simplified_individual_'+str(csv_filenumber)+'_pretraining.csv'
        else:
            self.experimental_parameters['samples/test/user/word']=1
            self.experimental_parameters['samples/val/user/word']=1
            self.experimental_parameters['samples/train/user/word']=-1
            self.experimental_parameters['samples/pretrain/train/user/word']=-1
            self.experimental_parameters['samples/pretest/train/user/word']=1
            file_finetune = 'dataset/dataset_simplified_individual_'+str(csv_filenumber)+'.csv'
            file_pretrain = 'dataset/dataset_simplified_individual_'+str(csv_filenumber)+'_pretraining.csv'

        return file_pretrain, file_finetune


    def curate_words_list(self):

        self.word_to_index = {}
        for word in self.all_words:
            if word in self.wanted_words_index:
                self.word_to_index[word] = self.wanted_words_index[word]
            else:
                self.word_to_index[word] = self.unknown_word_index
        if (len(self.words_list) == 12 and self.environment_parameters['keywords_dataset'] == "gscv2"):
            self.word_to_index[self.silence_label] = self.silence_index 
        elif (len(self.words_list) == 8 and self.environment_parameters['keywords_dataset'] == "gscv2"):
            self.word_to_index[self.silence_label] = self.silence_index 
        # Patch
        if (self.environment_parameters['keywords_dataset'] == "kinem"):
            self.word_to_index[self.silence_label] = self.silence_index 


    def prepare_words_list(self):

        # self.words_list and self.wanted_words_index can be combined

        if (self.environment_parameters['keywords_dataset'] == "gscv2"):
            if (len(self.experimental_parameters['wanted_words']) == 10):
                self.words_list = [self.silence_label, self.unknown_word_label] + self.experimental_parameters['wanted_words']    # 12 words 
            else:
                self.words_list = self.experimental_parameters['wanted_words']    # 6 / 35 words
        elif (self.environment_parameters['keywords_dataset'] == "mswc"):
            self.words_list = self.experimental_parameters['wanted_words']
        elif (self.environment_parameters['keywords_dataset'] == "kinem"):
            if (len(self.experimental_parameters['wanted_words']) == 10):
                self.words_list = [self.silence_label, self.unknown_word_label] + self.experimental_parameters['wanted_words']    # 12 words 
            else:
                self.words_list = self.experimental_parameters['wanted_words']

        self.wanted_words_index = {}
        for index, wanted_word in enumerate(self.experimental_parameters['wanted_words']):
            if (self.experimental_parameters['task'] =="gscv2_12w" or self.experimental_parameters['task'] =="gscv2_8w"):
                self.wanted_words_index[wanted_word] = index + 2    # 12 words
            else:
                self.wanted_words_index[wanted_word] = index    # 6 / 35 words / MSWC


    # Split dataset in training, validation, and testing set
    def which_set(self, filename, validation_percentage, testing_percentage, dataset_path):

        if (self.environment_parameters['keywords_dataset'] == 'gscv2'):

            base_name = os.path.basename(filename)
            hash_name = re.sub(r'_nohash_.*$', '', base_name)
            hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                                    (self.max_num_wavs_per_class + 1)) *
                                                 (100.0 / self.max_num_wavs_per_class))
            if percentage_hash < validation_percentage:
                result = 'preval'
            elif percentage_hash < (testing_percentage + validation_percentage):
                result = 'pretest'
            else:
                result = 'pretrain'
            return result

        elif (self.environment_parameters['keywords_dataset'] == 'mswc'):
            if filename in open(dataset_path[:-7]+"_align/en_dev.csv").read():
                result = 'preval'
            elif filename in open(dataset_path[:-7]+"_align/en_test.csv").read():
                result = 'pretest'
            else:
                result = 'pretrain'
            return result

        elif (self.environment_parameters['keywords_dataset'] == "kinem"):
            raise Exception ("Dataset split already defined in dataset directory.")
        else:
            raise ValueError ("Dataset not defined.")

    def get_user_number_pretrain(self):
        return len(self.user_list_pretrain)

    def get_user_number_finetune(self):
        return len(self.user_list_finetune)

    def get_noise_number_pretrain(self):
        return len(set(self.environment_parameters['offline_noise_train']+self.environment_parameters['offline_noise_test']))

    def get_noise_number_finetune(self):
        return len(set(self.environment_parameters['online_noise_train']+self.environment_parameters['online_noise_test']))


    # For each data set, generate a dictionary containing the path to each file, its label, and its speaker.
    # Ensure deterministic data shuffling
    def generate_data_dictionary(self):

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Creating occurences dictionary for frequency-based selection (e.g., MSWC datset)
        occurences_dict = {'testing': {}, 'validation': {}, 'training': {}}
        for word in self.experimental_parameters['wanted_words']:
            occurences_dict['testing'][word] = 0
            occurences_dict['validation'][word] = 0
            occurences_dict['training'][word] = 0

        # Prepare data sets
        self.all_words = {}
        self.data_set = {'validation': [], 'testing': [], 'training': [], 'pretrain': [], 'preval':[], 'pretest': []}
        unknown_set = {'validation': [], 'testing': [], 'training': [], 'pretrain': [], 'preval':[], 'pretest': []}
        
        # Find all audio samples
        if (self.environment_parameters['keywords_dataset'] == "gscv2"):
            search_path = os.path.join(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']], '*', '*.wav')
        elif(self.environment_parameters['keywords_dataset'] == "mswc"):
            search_path = os.path.join(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']], '*', '*.wav')
        elif(self.environment_parameters['keywords_dataset'] == "kinem"):
            search_path = os.path.join(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']], '*', '*', '*.wav.wav')

        if (self.environment_parameters['keywords_dataset'] == "gscv2"):  # Parsing the files individually

            # User management
            if (self.experimental_parameters['learn'] == "users") or (self.experimental_parameters['learn']=="users_noises"):

                pretrain_set, finetune_set = self.user_selection()

                self.df_csv_pretrain = pd.read_csv(pretrain_set)
                self.user_list_pretrain = self.df_csv_pretrain.iloc[:, 0].tolist()
                self.user_list_pretrain = sorted(self.user_list_pretrain)
                self.ids_dict_pretrain = {user_id: i for i, user_id in enumerate(self.user_list_pretrain)}

                self.df_csv_finetune = pd.read_csv(finetune_set)
                self.user_list_finetune = self.df_csv_finetune.iloc[:, 0].tolist()
                self.user_list_finetune = sorted(self.user_list_finetune)
                self.ids_dict_finetune = {user_id: i for i, user_id in enumerate(self.user_list_finetune)}

                if (self.experimental_parameters['pretrain']):
                    self.target_list = self.user_list_pretrain
                elif (self.experimental_parameters['finetune']):
                    self.target_list = self.user_list_finetune

                samples = []
                for word in self.experimental_parameters['wanted_words']:
                    search_path = os.path.join(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']], str(word), '*.wav')
                    for wav_path in glob.glob(search_path):
                        user_id = os.path.split(wav_path)[1].split('_')[0]
                        samples.append({'label': word, 'speaker': user_id, 'file': wav_path})
                samples = sorted(samples, key = lambda d: d['file'])

                # Prepare finetune list of users
                for user in self.user_list_finetune: 
                    # TODO: Selective finetuning (i.e., wanted_words != online_wanted_words)
                    for word in self.experimental_parameters['wanted_words']:
                        recordings = []
                        for processed_sample in samples:
                            if processed_sample['label'] == word and processed_sample['speaker'] == user:
                                # The finetune speakers are shifted considering the pretraining speakers
                                processed_sample['speaker'] = len(self.user_list_pretrain) + self.ids_dict_finetune[user] # + 1 # used to acknowledge non-speaker samples (i.e., silence)
                                recordings.append(processed_sample)
                        # Use more 'utterances' samples
                        if (self.experimental_parameters['samples/train/user/word'] == -1):
                            train_samples_cnt = len(recordings)-self.experimental_parameters['samples/test/user/word']-self.experimental_parameters['samples/val/user/word']
                        else:
                            train_samples_cnt = self.experimental_parameters['samples/train/user/word']

                        if (len(recordings) >= self.experimental_parameters['samples/test/user/word']): # assume at least the test_samples elements to be permuted
                            # Circular shift to generate folds
                            for i in range (self.experimental_parameters['fold']):
                                recordings.append(recordings.pop(0))

                        # The testset is the priority
                        if (len(recordings) >= self.experimental_parameters['samples/test/user/word']):
                            for i in range(0, self.experimental_parameters['samples/test/user/word']):
                                self.data_set['testing'].append(recordings[i])
                        # The valset is the second priority
                        if (len(recordings) >= self.experimental_parameters['samples/val/user/word'] + self.experimental_parameters['samples/test/user/word']):
                            for i in range(0, self.experimental_parameters['samples/val/user/word']):
                                self.data_set['validation'].append(recordings[i+self.experimental_parameters['samples/test/user/word']])
                        if (len(recordings) >= train_samples_cnt + self.experimental_parameters['samples/val/user/word'] + self.experimental_parameters['samples/test/user/word']):
                            for i in range(0, train_samples_cnt):
                                self.data_set['training'].append(recordings[i+self.experimental_parameters['samples/test/user/word']+self.experimental_parameters['samples/val/user/word']])

                        self.all_words[word] = True

                # Prepare pretrain list of users
                for word in self.experimental_parameters['wanted_words']:
                    curated_samples = []
                    for user in self.user_list_pretrain:
                        for processed_sample in samples:
                            if processed_sample['label'] == word and processed_sample['speaker'] == user:
                                processed_sample['speaker'] = self.ids_dict_pretrain[user] # + 1 # used to acknowledge non-speaker samples (i.e., silence)
                                curated_samples.append(processed_sample)
                    rand_shuffle = random.Random(self.random_seed)
                    rand_shuffle.shuffle(curated_samples)

                    self.data_set['pretrain'] = self.data_set['pretrain'] + curated_samples[0:int(0.8*len(curated_samples))]
                    self.data_set['preval'] = self.data_set['preval'] + curated_samples[int(0.8*len(curated_samples)):int(0.9*len(curated_samples))]
                    self.data_set['pretest'] = self.data_set['pretest'] + curated_samples[int(0.9*len(curated_samples)):]

                    self.all_words[word] = True
            else:

                self.sample_selection_naive()

                for wav_path in glob.glob(search_path):

                    _ , word = os.path.split(os.path.dirname(wav_path))
                    word = word.lower()
                    speaker_id = wav_path.split('/')[-1].split('_')[0]  # Hardcoded, should use regex.

                    # Ignore background noise, as it has been handled by generate_background_noise()
                    if word == self.background_noise_label:
                        continue

                    self.all_words[word] = True
                    # Determine the set to which the word should belong
                    set_index = self.which_set(wav_path, self.training_parameters['validation_percentage'], \
                        self.training_parameters['testing_percentage'], self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']])

                    # If it's a known class, store its detail, otherwise add it to 'unknown'
                    # For 35 target classes, there are no 'unknown samples'
                    if word in self.wanted_words_index:
                        self.data_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
                    else:
                        unknown_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})

        elif (self.environment_parameters['keywords_dataset'] == "mswc"): # Parsing the lists organizing the files
            with open(self.environment_parameters['data_dir'][:-7]+"_align/en_dev.csv", newline='') as csvfile:
                lines = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in lines:
                    word = row[1]
                    if (word in self.wanted_words_index):

                        self.all_words[word] = True
                        wav_path = str(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+row[0])+".wav"
                        speaker_id = row[0].split('_')[3].split('.')[0]
                        self.data_set['validation'].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
                        # Add word in occurences dict
                        occurences_dict['validation'][word] += 1

            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']][:-7]+"_align/en_test.csv", newline='') as csvfile:
                lines = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in lines:
                    word = row[1]
                    if (word in self.wanted_words_index):

                        self.all_words[word] = True
                        wav_path = str(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+row[0])+".wav"
                        speaker_id = row[0].split('_')[3].split('.')[0]
                        self.data_set['testing'].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
                        # Add word in occurences dict
                        occurences_dict['testing'][word] += 1

            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']][:-7]+"_align/en_train.csv", newline='') as csvfile:
                lines = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in lines:
                    word = row[1]
                    if (word in self.wanted_words_index):

                        self.all_words[word] = True
                        wav_path = str(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+row[0])+".wav"
                        speaker_id = row[0].split('_')[3].split('.')[0]
                        self.data_set['training'].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
                        # Add word in occurences dict
                        occurences_dict['training'][word] += 1

        elif (self.environment_parameters['keywords_dataset'] == "kinem"):
            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+'/noise'+self.environment_parameters['online_noise_test']+'_distance'+ \
                self.environment_parameters['distance'] + "_microphone" + self.environment_parameters['microphone'] + '_test.txt') as txtfile:

                lines = [line for line in txtfile]
                for line in lines:
                    class_index = line.split('/')[-1]
                    match = re.search(r'([a-zA-Z]+)(\d+)\.wav\.wav', class_index)
                    if match:
                        class_c, index_i = match.groups()
                    else:
                        # Handle the case where the regex doesn't match
                        class_c, index_i = None, None
                    speaker_id = line.split('/')[-2].split('_')[0].split('S')[-1]
                    if word in self.wanted_words_index:
                        self.data_set['testing'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    else:
                        unknown_set['testing'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    self.all_words[class_c.lower()] = True

            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+'/noise'+self.environment_parameters['online_noise_test']+'_distance'+ \
                self.environment_parameters['distance'] + "_microphone" + self.environment_parameters['microphone'] + '_train.txt') as txtfile:

                lines = [line for line in txtfile]
                for line in lines:
                    class_index = line.split('/')[-1]
                    match = re.search(r'([a-zA-Z]+)(\d+)\.wav\.wav', class_index)
                    if match:
                        class_c, index_i = match.groups()
                    else:
                        # Handle the case where the regex doesn't match
                        class_c, index_i = None, None
                    speaker_id = line.split('/')[-2].split('_')[0].split('S')[-1]
                    if word in self.wanted_words_index:
                        self.data_set['training'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    else:
                        unknown_set['training'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    self.all_words[class_c.lower()] = True

            # KINEM has no validation dataset, so we use the training data naively for validation
            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+'/noise'+self.environment_parameters['online_noise_test']+'_distance'+ \
                self.environment_parameters['distance'] + "_microphone" + self.environment_parameters['microphone'] + '_train.txt') as txtfile:

                lines = [line for line in txtfile]
                for line in lines:
                    class_index = line.split('/')[-1]
                    match = re.search(r'([a-zA-Z]+)(\d+)\.wav\.wav', class_index)
                    if match:
                        class_c, index_i = match.groups()
                    else:
                        # Handle the case where the regex doesn't match
                        class_c, index_i = None, None
                    speaker_id = line.split('/')[-2].split('_')[0].split('S')[-1]
                    if word in self.wanted_words_index:
                        self.data_set['validation'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    else:
                        unknown_set['validation'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    self.all_words[class_c.lower()] = True

        if not self.all_words:
            raise Exception('No .wavs found at ' + search_path)

        for index, wanted_word in enumerate(self.experimental_parameters['wanted_words']):
            if wanted_word not in self.all_words:
                if (self.environment_parameters['keywords_dataset'] == 'kinem'):
                    continue    # data could miss in evaluation
                raise Exception('Expected to find ' + wanted_word +
                                                ' in labels but only found ' +
                                                ', '.join(self.all_words.keys()))

        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_set['pretrain'][0]['file']

        # Silence and Unknown are added only when the user ID does NOT matter
        if (self.experimental_parameters["learn"] == "noises"):
            # Add silence and unknown words to each set
            for set_index in ['preval', 'pretest', 'pretrain']:

                set_size = len(self.data_set[set_index])
                if (len(self.experimental_parameters['wanted_words']) == 10):
                    silence_size = int(math.ceil(set_size * self.training_parameters['silence_percentage'] / 100))
                    for _ in range(silence_size):
                        self.data_set[set_index].append({
                                'label': self.silence_label,
                                'file': silence_wav_path,
                                # 'speaker': "None"
                                'speaker': 0 # Handle speaker ID for  
                        })

                # Pick some unknowns to add to each partition of the data set.
                rand_unknown = random.Random(self.random_seed)
                rand_unknown.shuffle(unknown_set[set_index])
                unknown_size = int(math.ceil(set_size * self.training_parameters['unknown_percentage'] / 100))
                self.data_set[set_index].extend(unknown_set[set_index][:unknown_size])

            for set_index in ['preval', 'pretest', 'pretrain']:
                rand_data_order = random.Random(self.random_seed)
                rand_data_order.shuffle(self.data_set[set_index])

            if self.experimental_parameters["fixnr"]:
                # instead of shifting samples just shift random seed to subsample
                random.seed(self.random_seed + self.experimental_parameters["fold"])
                grouped_data = defaultdict(list)
                for entry in self.data_set['pretrain']:
                    grouped_data[entry["label"]].append(entry)

                random_samples = []
                for word in self.experimental_parameters['wanted_words']:
                    if len(grouped_data[word]) >= self.experimental_parameters["utterances"]:
                        random_samples.extend(random.sample(grouped_data[word], self.experimental_parameters["utterances"]))
                    elif len(grouped_data[word]) > 0:
                        random_samples.extend(random.sample(grouped_data[word], len(grouped_data[word])))
                self.data_set['training'] = random_samples

                print ("self.data_set['training']: ", len(self.data_set['training']))

                grouped_data = defaultdict(list)
                for entry in self.data_set['preval']:
                    grouped_data[entry["label"]].append(entry)

                random_samples = []
                for word in self.experimental_parameters['wanted_words']:

                    if self.experimental_parameters['samples/val/user/word'] == -1:
                        random_samples.extend(grouped_data[word])
                    elif len(grouped_data[word]) >= self.experimental_parameters['samples/val/user/word']:
                        # Sample 1 random entries for the label
                        random_samples.extend(random.sample(grouped_data[word], self.experimental_parameters['samples/val/user/word']))
                    elif len(grouped_data[word]) > 0:
                        random_samples.extend(random.sample(grouped_data[word], len(grouped_data[word])))

                self.data_set['validation'] = random_samples

                grouped_data = defaultdict(list)
                for entry in self.data_set['pretest']:
                    grouped_data[entry["label"]].append(entry)

                random_samples = []
                for word in self.experimental_parameters['wanted_words']:
                    if self.experimental_parameters['samples/test/user/word'] == -1:
                        random_samples.extend(grouped_data[word])
                    elif len(grouped_data[word]) >= self.experimental_parameters['samples/test/user/word']:
                        # Sample 1 random entries for the label
                        random_samples.extend(random.sample(grouped_data[word], self.experimental_parameters['samples/test/user/word']))
                    elif len(grouped_data[word]) > 0:
                        random_samples.extend(random.sample(grouped_data[word], len(grouped_data[word])))
                self.data_set['testing'] = random_samples

                print ("self.data_set['testing']: ", len(self.data_set['testing']))

                random.seed(self.random_seed) 
                for set_index in ['validation', 'testing', 'training']:
                    rand_data_order = random.Random(self.random_seed)
                    rand_data_order.shuffle(self.data_set[set_index])
            else:
                self.data_set['training'] = deepcopy(self.data_set['pretrain'])
                self.data_set['validation'] = deepcopy(self.data_set['preval'])
                self.data_set['testing'] = deepcopy(self.data_set['pretest'])    

        else:
            # Make sure the ordering is random.
            for set_index in ['validation', 'testing', 'training', 'preval', 'pretest', 'pretrain']:
                rand_data_order = random.Random(self.random_seed)
                rand_data_order.shuffle(self.data_set[set_index])


    # Load noise dataset
    def generate_noise_set(self, dataset_directory, dataset_name, noise_list, noise_key):

        recordings = []
        names = []

        if (dataset_name == "kinem"):
            background_dir = os.path.join(self.environment_parameters['noise_dir_'+self.environment_parameters[noise_key]])
            background_dir = glob.glob(background_dir+'/' + noise_list[0] +'/S*_' + self.environment_parameters['microphone'] + self.environment_parameters['distance'])[0]
        else:
            background_dir = os.path.join(self.environment_parameters['noise_dir_'+self.environment_parameters[noise_key]])


        if not os.path.exists(background_dir):
            raise OSError("Background noise directory not found.")

        # Iterate through existing .wavs
        for wav_path in sorted(Path(background_dir).rglob('*.wav')):

            if (dataset_name == "gscv2"):
                noise_type = str(wav_path).split('/')[-1].split('.wav')[0]
            elif (dataset_name == "demand"):
                noise_type = str(wav_path).split('/')[-2]
            elif (dataset_name == "kinem"):
                noise_type = str(wav_path).split('/')[-3]
            else:
                raise ValueError("Dataset management not defined.")
            if noise_type in noise_list:
                if (dataset_name == "demand"):
                    if (noise_key=="offline_noise_val_dataset") or (noise_key=="online_noise_val_dataset"):
                        if ("ch15" in str(wav_path)):

                            noise_path = str(wav_path)
                            sf_loader_noise, _ = sf.read(noise_path)
                            wav_background_samples = torch.from_numpy(sf_loader_noise).float()

                            recordings.append(wav_background_samples)
                            names.append(noise_type)
                    elif (noise_key=="offline_noise_test_dataset") or (noise_key=="online_noise_test_dataset"):
                        if ("ch16" in str(wav_path)):

                            noise_path = str(wav_path)
                            sf_loader_noise, _ = sf.read(noise_path)
                            wav_background_samples = torch.from_numpy(sf_loader_noise).float()

                            recordings.append(wav_background_samples)
                            names.append(noise_type)
                    elif (noise_key=="offline_noise_train_dataset") or (noise_key=="online_noise_train_dataset"):
                        if ("ch15" not in str(wav_path) and "ch16" not in str(wav_path)):
                            noise_path = str(wav_path)
                            sf_loader_noise, _ = sf.read(noise_path)
                            wav_background_samples = torch.from_numpy(sf_loader_noise).float()
                            
                            recordings.append(wav_background_samples)
                            names.append(noise_type)

                elif (dataset_name == "gscv2"):
                    noise_path = str(wav_path)
                    sf_loader_noise, _ = sf.read(noise_path)
                    wav_background_samples = torch.from_numpy(sf_loader_noise).float()
                    
                    recordings.append(wav_background_samples)
                    names.append(noise_type)

                elif (dataset_name == "kinem"):
                    if ("background_noise.wav.wav" in str(wav_path)):
                        noise_path = str(wav_path)

                        sf_loader_noise, _ = sf.read(noise_path)
                        wav_background_samples = torch.from_numpy(sf_loader_noise).float()
                        
                        recordings.append(wav_background_samples[:, 0]) # add all samples of the first channel
                        names.append(noise_type)


        if ("SILENCE" in noise_list):
            # Add as many silence samples as samples for other noises
            n_samples = int(len(recordings)/(len(noise_list)-1))

            for i in range (0, n_samples):
                recordings.append(torch.Tensor(np.zeros(20000)))
                names.append("SILENCE")

        if not recordings:
            raise Exception('No background wav files were found in ' + dataset_directory)

        return recordings, names


    # Load complete set of background noises
    def generate_background_noise(self):
        
        self.offline_background_noise_train, self.offline_background_noise_train_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["offline_noise_train_dataset"]],
            self.environment_parameters["offline_noise_train_dataset"],
            self.environment_parameters["offline_noise_train"],
            "offline_noise_train_dataset"
            )
        self.offline_background_noise_val, self.offline_background_noise_val_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["offline_noise_test_dataset"]],
            self.environment_parameters["offline_noise_test_dataset"],
            self.environment_parameters["offline_noise_test"],
            "offline_noise_val_dataset"
            )
        self.offline_background_noise_test, self.offline_background_noise_test_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["offline_noise_test_dataset"]],
            self.environment_parameters["offline_noise_test_dataset"],
            self.environment_parameters["offline_noise_test"],
            "offline_noise_test_dataset"
            )
        self.online_background_noise_train, self.online_background_noise_train_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["online_noise_train_dataset"]],
            self.environment_parameters["online_noise_train_dataset"],
            self.environment_parameters["online_noise_train"],
            "online_noise_train_dataset"
            )
        self.online_background_noise_val, self.online_background_noise_val_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["online_noise_test_dataset"]],
            self.environment_parameters["online_noise_test_dataset"],
            self.environment_parameters["online_noise_test"],
            "online_noise_val_dataset"
            )
        self.online_background_noise_test, self.online_background_noise_test_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["online_noise_test_dataset"]],
            self.environment_parameters["online_noise_test_dataset"],
            self.environment_parameters["online_noise_test"],
            "online_noise_test_dataset"
            )


    # Generate reverb model
    def generate_reverberant_rooms(self):

        # For room parameters, see wham_room.py
        n_room_train = self.environment_parameters['reverb_train_n']
        n_room_val = self.environment_parameters['reverb_val_n']
        n_room_test = self.environment_parameters['reverb_test_n']

        modes = ["training", "validation", "testing"] 

        n_room = {
        "training": self.environment_parameters['reverb_train_n'],
        "validation": self.environment_parameters['reverb_val_n'],
        "testing": self.environment_parameters['reverb_test_n']
        }

        self.reverb_rooms = {
        "training": [],
        "validation": [],
        "testing": []
        }

        self.anechoic_rooms = {
        "training": [],
        "validation": [],
        "testing": []
        }

        for mode in modes:
            for idx_room in range(n_room[mode]):
                room_param_dict = reverb.gen_room_params()
                self.anechoic_rooms[mode].append(reverb.gen_anechoic_room(room_param_dict))
                self.reverb_rooms[mode].append(reverb.gen_reverb_room(room_param_dict))


    # Compute data set size
    def get_size(self, mode):
        return len(self.data_set[mode])
