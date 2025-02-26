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


import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCNNS(torch.nn.Module):
    def __init__(self, n_classes=12, use_bias=True, emb_format='add', n_user_envs = None, n_noise_envs = None):
        super(DSCNNS, self).__init__()
        self.max = 0
        self.emb_format = emb_format
        self.n_user_envs = n_user_envs
        self.n_noise_envs = n_noise_envs

        if (self.n_user_envs):
            if (self.emb_format == 'concatx2'):
                self.emb   = torch.nn.Embedding(n_user_envs, 64)
            elif (self.emb_format == 'concatx1'):
                self.emb   = torch.nn.Embedding(n_user_envs, 32)
            else:
                self.emb   = torch.nn.Embedding(n_user_envs, 64)
            if (self.emb_format == 'add' or self.emb_format == 'concatx1' or self.emb_format == 'concatx2'):
                nn.init.zeros_(self.emb.weight)
            elif (self.emb_format == 'mult'):
                nn.init.ones_(self.emb.weight)
            else:
                raise ValueError("Integration mode not supported.")

        if (self.n_noise_envs):
            if (self.emb_format == 'mult'):
                self.emb2 = torch.nn.Embedding(n_noise_envs, 64)
                nn.init.ones_(self.emb2.weight)
            else:
                raise ValueError("Integration mode not supported.")

        self.pad1  = nn.ConstantPad2d((1, 1, 5, 5), value=0.0)
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (10, 4), stride = (2, 2), bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()

        self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn2   = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn3   = torch.nn.BatchNorm2d(64)
        self.relu3 = torch.nn.ReLU()

        self.pad4  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv4 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn4   = torch.nn.BatchNorm2d(64)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn5   = torch.nn.BatchNorm2d(64)
        self.relu5 = torch.nn.ReLU()

        self.pad6  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv6 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn6   = torch.nn.BatchNorm2d(64)
        self.relu6 = torch.nn.ReLU()
        self.conv7 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn7   = torch.nn.BatchNorm2d(64)
        self.relu7 = torch.nn.ReLU()

        if (self.emb_format == 'concatx1'):
            self.pad8  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
            self.conv8 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
            self.bn8   = torch.nn.BatchNorm2d(64)
            self.relu8 = torch.nn.ReLU()
            self.conv9 = torch.nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
            self.bn9   = torch.nn.BatchNorm2d(32)
            self.relu9 = torch.nn.ReLU()
        else:
            self.pad8  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
            self.conv8 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
            self.bn8   = torch.nn.BatchNorm2d(64)
            self.relu8 = torch.nn.ReLU()
            self.conv9 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
            self.bn9   = torch.nn.BatchNorm2d(64)
            self.relu9 = torch.nn.ReLU()


        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)

        if (self.emb_format == 'concatx2' and self.n_user_envs):
            self.fc1   = torch.nn.Linear(128, n_classes, bias=use_bias)
        elif (self.emb_format == 'concatx1' and self.n_user_envs):
            self.fc1   = torch.nn.Linear(64, n_classes, bias=use_bias)
        else:  
            self.fc1   = torch.nn.Linear(64, n_classes, bias=use_bias)
        
    def forward(self, x, y = None, z=None):

        if (self.n_user_envs):
            y = self.emb(y.long())
        if (self.n_noise_envs):
            z = self.emb2(z.long())

        x = self.pad1 (x)
        x = self.conv1(x)       
        x = self.bn1  (x)         
        x = self.relu1(x)
        
        x = self.pad2 (x)
        x = self.conv2(x)           
        x = self.bn2  (x)            
        x = self.relu2(x)            
        x = self.conv3(x)            
        x = self.bn3  (x)            
        x = self.relu3(x)
        
        x = self.pad4 (x)
        x = self.conv4(x)            
        x = self.bn4  (x)            
        x = self.relu4(x)            
        x = self.conv5(x)            
        x = self.bn5  (x)            
        x = self.relu5(x)            

        x = self.pad6 (x)
        x = self.conv6(x)          
        x = self.bn6  (x)            
        x = self.relu6(x)          
        x = self.conv7(x)            
        x = self.bn7  (x)            
        x = self.relu7(x)
        
        x = self.pad8 (x)            
        x = self.conv8(x)            
        x = self.bn8  (x)            
        x = self.relu8(x)            
        x = self.conv9(x)            
        x = self.bn9  (x)           

        x = self.relu9(x)        

        x = self.avg(x)            
        x = torch.flatten(x, 1) 

        if (self.n_user_envs):
            if (self.emb_format == 'add'):
                x = torch.add(x, y)
            elif (self.emb_format == 'mult'):
                x = torch.mul(x, y)
            elif (self.emb_format == 'concatx1' or self.emb_format == 'concatx2'):
                x = torch.cat((x, y), dim=1)
            else:
                raise ValueError("Integration mode not supported.")

        if (self.n_noise_envs):
            if (self.emb_format == 'mult'):
                x = torch.mul(x, z)
            else:
                raise ValueError("Integration mode not supported.")

        x = self.fc1(x)
        return F.softmax(x, dim=1)

class DSCNNM(torch.nn.Module):
    def __init__(self, n_classes=12, use_bias=True, emb_format='add', n_user_envs = None, n_noise_envs = None):
        super(DSCNNM, self).__init__()

        self.emb_format = emb_format
        self.n_user_envs = n_user_envs
        self.n_noise_envs = n_noise_envs

        if (self.n_user_envs):
            if (self.emb_format == 'concatx2'):
                self.emb   = torch.nn.Embedding(n_user_envs, 172)
            elif (self.emb_format == 'concatx1'):
                self.emb   = torch.nn.Embedding(n_user_envs, 86)
            else:
                self.emb   = torch.nn.Embedding(n_user_envs, 172)
            if (self.emb_format == 'add' or self.emb_format == 'concatx1' or self.emb_format == 'concatx2'):
                nn.init.zeros_(self.emb.weight)
            elif (self.emb_format == 'mult'):
                nn.init.ones_(self.emb.weight)
            else:
                raise ValueError("Integration mode not supported.")

        if (self.n_noise_envs):
            raise ValueError("Noise embeddings not integrated.")

        self.pad1  = nn.ConstantPad2d((1, 1, 5, 5), value=0.0)
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 172, kernel_size = (10, 4), stride = (2, 2), bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(172)
        self.relu1 = torch.nn.ReLU()

        self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv2 = torch.nn.Conv2d(in_channels = 172, out_channels = 172, kernel_size = (3, 3), stride = (1, 1), groups = 172, bias = use_bias)
        self.bn2   = torch.nn.BatchNorm2d(172)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels = 172, out_channels = 172, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn3   = torch.nn.BatchNorm2d(172)
        self.relu3 = torch.nn.ReLU()

        self.pad4  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv4 = torch.nn.Conv2d(in_channels = 172, out_channels = 172, kernel_size = (3, 3), stride = (1, 1), groups = 172, bias = use_bias)
        self.bn4   = torch.nn.BatchNorm2d(172)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(in_channels = 172, out_channels = 172, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn5   = torch.nn.BatchNorm2d(172)
        self.relu5 = torch.nn.ReLU()

        self.pad6  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv6 = torch.nn.Conv2d(in_channels = 172, out_channels = 172, kernel_size = (3, 3), stride = (1, 1), groups = 172, bias = use_bias)
        self.bn6   = torch.nn.BatchNorm2d(172)
        self.relu6 = torch.nn.ReLU()
        self.conv7 = torch.nn.Conv2d(in_channels = 172, out_channels = 172, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn7   = torch.nn.BatchNorm2d(172)
        self.relu7 = torch.nn.ReLU()

        if (self.emb_format == 'concatx1'):
            self.pad8  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
            self.conv8 = torch.nn.Conv2d(in_channels = 172, out_channels = 172, kernel_size = (3, 3), stride = (1, 1), groups = 172, bias = use_bias)
            self.bn8   = torch.nn.BatchNorm2d(172)
            self.relu8 = torch.nn.ReLU()
            self.conv9 = torch.nn.Conv2d(in_channels = 172, out_channels = 86, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
            self.bn9   = torch.nn.BatchNorm2d(86)
            self.relu9 = torch.nn.ReLU()
        else:
            self.pad8  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
            self.conv8 = torch.nn.Conv2d(in_channels = 172, out_channels = 172, kernel_size = (3, 3), stride = (1, 1), groups = 172, bias = use_bias)
            self.bn8   = torch.nn.BatchNorm2d(172)
            self.relu8 = torch.nn.ReLU()
            self.conv9 = torch.nn.Conv2d(in_channels = 172, out_channels = 172, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
            self.bn9   = torch.nn.BatchNorm2d(172)
            self.relu9 = torch.nn.ReLU()

        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)

        if (self.emb_format == 'concatx2' and self.n_user_envs):
            self.fc1   = torch.nn.Linear(344, n_classes, bias=use_bias)
        elif (self.emb_format == 'concatx1' and self.n_user_envs):
            self.fc1   = torch.nn.Linear(172, n_classes, bias=use_bias)
        else:  
            self.fc1   = torch.nn.Linear(172, n_classes, bias=use_bias)

        
    def forward(self, x, y = None):

        if (self.n_user_envs):
            y = self.emb(y)
            y = y.unsqueeze(2)
            y = torch.repeat_interleave(y, 25, dim=2)
            y = y.unsqueeze(3)
            y = torch.repeat_interleave(y, 5, dim=3)

    
        x = self.pad1 (x)
        x = self.conv1(x)       
        x = self.bn1  (x)         
        x = self.relu1(x)
        
        x = self.pad2 (x)
        x = self.conv2(x)           
        x = self.bn2  (x)            
        x = self.relu2(x)            
        x = self.conv3(x)            
        x = self.bn3  (x)            
        x = self.relu3(x)
        
        x = self.pad4 (x)
        x = self.conv4(x)            
        x = self.bn4  (x)            
        x = self.relu4(x)            
        x = self.conv5(x)            
        x = self.bn5  (x)            
        x = self.relu5(x)            

        x = self.pad6 (x)
        x = self.conv6(x)          
        x = self.bn6  (x)            
        x = self.relu6(x)          
        x = self.conv7(x)            
        x = self.bn7  (x)            
        x = self.relu7(x)
        
        x = self.pad8 (x)            
        x = self.conv8(x)            
        x = self.bn8  (x)            
        x = self.relu8(x)            
        x = self.conv9(x)            
        x = self.bn9  (x)        

        if (self.n_user_envs):
            if (self.emb_format == 'add'):
                x = torch.add(x, y)
            elif (self.emb_format == 'mult'):
                x = torch.mul(x, y)
            elif (self.emb_format == 'concatx1' or self.emb_format == 'concatx2'):
                x = torch.cat((x, y), dim=1)


        x = self.relu9(x)     

        x = self.avg(x)            
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
            
        return F.softmax(x, dim=1) 



class DSCNNL(torch.nn.Module):
    def __init__(self, n_classes=12, use_bias=True, emb_format='add', n_user_envs = None, n_noise_envs = None):
        super(DSCNNL, self).__init__()

        self.emb_format = emb_format
        self.n_user_envs = n_user_envs
        self.n_noise_envs = n_noise_envs

        if (n_user_envs):
            if (self.emb_format == 'concatx2'):
                self.emb   = torch.nn.Embedding(n_user_envs, 276)
            elif (self.emb_format == 'concatx1'):
                self.emb   = torch.nn.Embedding(n_user_envs, 138)
            else:
                self.emb   = torch.nn.Embedding(n_user_envs, 276)

            if (self.emb_format == 'add' or self.emb_format == 'concatx1' or self.emb_format == 'concatx2'):
                nn.init.zeros_(self.emb.weight)
            elif (self.emb_format == 'mult'):
                nn.init.ones_(self.emb.weight)       
            else:
                raise ValueError("Integration mode not supported.")

        if (self.n_noise_envs):
            raise ValueError("Noise embeddings not integrated.")

        self.pad1  = nn.ConstantPad2d((1, 1, 5, 5), value=0.0)
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 276, kernel_size = (10, 4), stride = (2, 2), bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(276)
        self.relu1 = torch.nn.ReLU()

        self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv2 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (3, 3), stride = (1, 1), groups = 276, bias = use_bias)
        self.bn2   = torch.nn.BatchNorm2d(276)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn3   = torch.nn.BatchNorm2d(276)
        self.relu3 = torch.nn.ReLU()

        self.pad4  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv4 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (3, 3), stride = (1, 1), groups = 276, bias = use_bias)
        self.bn4   = torch.nn.BatchNorm2d(276)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn5   = torch.nn.BatchNorm2d(276)
        self.relu5 = torch.nn.ReLU()

        self.pad6  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv6 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (3, 3), stride = (1, 1), groups = 276, bias = use_bias)
        self.bn6   = torch.nn.BatchNorm2d(276)
        self.relu6 = torch.nn.ReLU()
        self.conv7 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn7   = torch.nn.BatchNorm2d(276)
        self.relu7 = torch.nn.ReLU()

        self.pad8  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv8 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (3, 3), stride = (1, 1), groups = 276, bias = use_bias)
        self.bn8   = torch.nn.BatchNorm2d(276)
        self.relu8 = torch.nn.ReLU()
        self.conv9 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn9   = torch.nn.BatchNorm2d(276)
        self.relu9 = torch.nn.ReLU()


        if (self.emb_format == 'concatx1'):
            self.pad10  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
            self.conv10 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (3, 3), stride = (1, 1), groups = 276, bias = use_bias)
            self.bn10   = torch.nn.BatchNorm2d(276)
            self.relu10 = torch.nn.ReLU()
            self.conv11 = torch.nn.Conv2d(in_channels = 276, out_channels = 138, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
            self.bn11   = torch.nn.BatchNorm2d(138)
            self.relu11 = torch.nn.ReLU()
        else:
            self.pad10  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
            self.conv10 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (3, 3), stride = (1, 1), groups = 276, bias = use_bias)
            self.bn10   = torch.nn.BatchNorm2d(276)
            self.relu10 = torch.nn.ReLU()
            self.conv11 = torch.nn.Conv2d(in_channels = 276, out_channels = 276, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
            self.bn11   = torch.nn.BatchNorm2d(276)
            self.relu11 = torch.nn.ReLU()

        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)

        if (self.emb_format == 'concatx2' and self.n_user_envs):
            self.fc1   = torch.nn.Linear(552, n_classes, bias=use_bias)
        elif (self.emb_format == 'concatx1' and self.n_user_envs):
            self.fc1   = torch.nn.Linear(276, n_classes, bias=use_bias)
        else:  
            self.fc1   = torch.nn.Linear(276, n_classes, bias=use_bias)

        
    def forward(self, x, y = None):

        if (self.n_user_envs):
            y = self.emb(y)
            y = y.unsqueeze(2)
            y = torch.repeat_interleave(y, 25, dim=2)
            y = y.unsqueeze(3)
            y = torch.repeat_interleave(y, 5, dim=3)


        x = self.pad1 (x)
        x = self.conv1(x)       
        x = self.bn1  (x)         
        x = self.relu1(x)
        
        x = self.pad2 (x)
        x = self.conv2(x)           
        x = self.bn2  (x)            
        x = self.relu2(x)            
        x = self.conv3(x)            
        x = self.bn3  (x)            
        x = self.relu3(x)
        
        x = self.pad4 (x)
        x = self.conv4(x)            
        x = self.bn4  (x)            
        x = self.relu4(x)            
        x = self.conv5(x)            
        x = self.bn5  (x)            
        x = self.relu5(x)            

        x = self.pad6 (x)
        x = self.conv6(x)          
        x = self.bn6  (x)            
        x = self.relu6(x)          
        x = self.conv7(x)            
        x = self.bn7  (x)            
        x = self.relu7(x)
        
        x = self.pad8 (x)            
        x = self.conv8(x)            
        x = self.bn8  (x)            
        x = self.relu8(x)            
        x = self.conv9(x)            
        x = self.bn9  (x)            
        x = self.relu9(x)        

        x = self.pad10 (x)
        x = self.conv10(x)
        x = self.bn10  (x)
        x = self.relu10(x)   
        x = self.conv11(x)
        x = self.bn11  (x)

        if (self.n_user_envs):
            if (self.emb_format == 'add'):
                x = torch.add(x, y)
            elif (self.emb_format == 'mult'):
                x = torch.mul(x, y)
            elif (self.emb_format == 'concatx1' or self.emb_format == 'concatx2'):
                x = torch.cat((x, y), dim=1)

        x = self.relu11(x)   

        x = self.avg(x)            
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
            
        return F.softmax(x, dim=1) 
        