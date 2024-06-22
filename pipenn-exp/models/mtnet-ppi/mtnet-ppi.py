# # ### PYTHONPATH FOR LOCAL RUNNING
# import sys
# print("PYTHONPATH:", sys.path)
# # List of directories to append
# new_directory = "/Users/kikivangerwen/Documents/PIPENN/pipenn-exp/utils"
# # Append multiple directories to sys.path
# sys.path.append(new_directory)
# # Print the updated sys.path
# print("Updated sys.path:", sys.path)

# make sure comet_ml is the first import (before all other Machine learning lib)
from comet_ml import Experiment

import datetime, time, os, sys, inspect
import numpy as np

from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, he_uniform, he_normal, lecun_uniform, lecun_normal, \
                                                 Orthogonal, TruncatedNormal, VarianceScaling
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Reshape, Input, concatenate, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Activation, \
                                           BatchNormalization, Conv2DTranspose, ZeroPadding2D, Dropout, UpSampling2D, UpSampling1D, \
                                           LSTM, GRU, TimeDistributed, Bidirectional, Dense, Add, ConvLSTM2D, Flatten, AveragePooling1D, \
                                           ReLU, LeakyReLU, PReLU, ELU, AlphaDropout, Lambda
from tensorflow.keras.models import Model

from keras.layers import Concatenate
from PPILogger import PPILoggerCls
from PPILoss import PPILossCls, LossParams
from PPIDataset import PPIDatasetCls, DatasetParams
from PPITrainTest import PPITrainTestCls, TrainingParams, MyPReLU
from PPIParams import PPIParamsCls
from PPIParams import PPIParamsCls
from PPIExplanation import PPIExplanationCls, ExplanationParams
from tensorflow.keras.initializers import glorot_normal
import tensorflow as tf


class AlgParams:

    ### Kiki
    ALGRITHM_NAME = "mtnet-ppi"
    DatasetParams.USE_COMET = False

    ### Kiki
    datasetLabel = 'Biolip_P'
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_NOWIN
    
    ONLY_TEST = False
    USE_EXPLAIN = False #True
    
    USE_2D_MODEL = False
    USE_POOLING = False
    USE_BN = True
    USE_DROPOUT = True

    ### Kiki
    DatasetParams.ONE_HOT_ENCODING = False
    #DatasetParams.USE_VAR_BATCH_INPUT = True
    
    INPUT_SHAPE = None  #must be determined in init.
    LABEL_SHAPE = None

    ## kiki: multi-task: 'p_interface', 'rel_surf_acc', 'prob_sheet', 'prob_helix', 'prob_coil'
    LABEL_DIM = 5
    if USE_2D_MODEL:
        PROT_IMAGE_H, PROT_IMAGE_W = 32, 32 #28, 28 #16, 16
        CNN_KERNEL_SIZE = 3
        CNN_POOL_SIZE = 4 #2

    else:
        # The receptive filed size, for a kernel-size of k and dilation rate of r can be calculated by: k+(k-1)(r-1). 
        if DatasetParams.USE_VAR_BATCH_INPUT:
            PROT_IMAGE_H,PROT_IMAGE_W = None,1

        else:
            PROT_IMAGE_H, PROT_IMAGE_W = 1170,1 #1170,1 #1024,1 #256,1 #2048,1 #576,1 #768,1 #384,1 #512,1
        
        CNN_KERNEL_SIZE = 7 #15 #4 #7
        CNN_POOL_SIZE = 2
        PROT_LEN = PROT_IMAGE_H     #PROT_LEN must be divisible by PAT_LEN.
        PAT_LEN = 8
    
    INIT_CNN_DILATION_SIZE = 1 #2 #1
    INIT_CNN_CHANNEL_SIZE = 128 #64 #32 #8 #4 #16 #128 #256
    
    if DatasetParams.USE_VAR_BATCH_INPUT:
        LossParams.USE_DELETE_PAD = True #False #True
        DatasetParams.MAX_PROT_LEN_PER_BATCH = [64,128,256,320,448,576,1170]  #False(65.13%)#True(68.57%)
        #DatasetParams.MAX_PROT_LEN_PER_BATCH = [128,512,1170] 
        #DatasetParams.MAX_PROT_LEN_PER_BATCH = [1170] 
        #DatasetParams.MAX_PROT_LEN_PER_BATCH = None
        TrainingParams.ACTIVATION_FUN = ReLU #MyPReLU(68.57%) #ReLU(68.84%)
        #TrainingParams.CUSTOM_OBJECTS = {'MyPReLU': MyPReLU}
    
    #DatasetParams.USE_MC_RESNET = True
    #DatasetParams.setMCResnetParams(PROT_LEN, PAT_LEN)
    #LossParams.setLossFun(LossParams.MC_RESNET)
    
    @classmethod
    def setShapes(cls):
        if cls.USE_2D_MODEL:
            cls.INPUT_SHAPE = (cls.PROT_IMAGE_H, cls.PROT_IMAGE_W, DatasetParams.getFeaturesDim())
            cls.LABEL_SHAPE = (cls.PROT_IMAGE_H, cls.PROT_IMAGE_W, cls.LABEL_DIM)

        ### Kiki: LABEL_SHAPE is dependend on LABEL_DIM
        else:
            cls.INPUT_SHAPE = (cls.PROT_IMAGE_H, DatasetParams.getFeaturesDim())   
            cls.LABEL_SHAPE = (cls.PROT_IMAGE_H, cls.LABEL_DIM)
            print('LABEL_DIM', cls.LABEL_DIM)
        PPIParamsCls.setShapeParams(cls.INPUT_SHAPE, cls.LABEL_SHAPE)   
        return
      
    @classmethod
    def initAlgParams(cls, dsParam=dataset, dsLabelParam=datasetLabel, ensembleTesting=False):
        if ensembleTesting:
            cls.ONLY_TEST = True
        else:
            PPIParamsCls.setLoggers(cls.ALGRITHM_NAME, dsLabelParam)
        
        EX_COLUMNS2 = [
                #'normalized_hydropathy_index',
                #'3_wm_normalized_hydropathy_index','5_wm_normalized_hydropathy_index','7_wm_normalized_hydropathy_index','9_wm_normalized_hydropathy_index',
                 ]
        EX_COLUMNS = ['domain',
                    # 'normalized_length',
                    'normalized_abs_surf_acc',
                    'rel_surf_acc',
                    'prob_sheet',
                    'prob_helix',
                    'prob_coil',
                    'pssm_A','pssm_R','pssm_N','pssm_D','pssm_C','pssm_Q','pssm_E','pssm_G','pssm_H','pssm_I',
                    'pssm_L','pssm_K','pssm_M','pssm_F','pssm_P','pssm_S','pssm_T','pssm_W','pssm_Y','pssm_V']
        IN_COLUMNS = [
                    #'glob_stat_score',
                    ]
        
        #PPIParamsCls.setInitParams(ALGRITHM_NAME,dsParam=dsParam, dsExParam=EX_COLUMNS,dsInParam=IN_COLUMNS,dsLabelParam=dsLabelParam)   
        PPIParamsCls.setInitParams(cls.ALGRITHM_NAME, dsParam=dsParam, dsExParam=EX_COLUMNS, dsLabelParam=dsLabelParam)
        cls.setShapes()
        return 
 

def make1DBlock(x, channelSize,  dilationRate):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    #kr = l2(TrainingParams.REG_LAMDA)
    #br = l2(TrainingParams.REG_LAMDA)
    #kr = l1_l2(TrainingParams.REG_LAMDA, TrainingParams.REG_LAMDA)
    #br = l1_l2(TrainingParams.REG_LAMDA, TrainingParams.REG_LAMDA)
    x = Conv1D(channelSize, AlgParams.CNN_KERNEL_SIZE, dilation_rate=dilationRate, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    
    if AlgParams.USE_DROPOUT:
        x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    if AlgParams.USE_BN:
        x = BatchNormalization()(x)    
    
    if DatasetParams.USE_VAR_BATCH_INPUT:    
        x = TrainingParams.ACTIVATION_FUN(shared_axes=[1])(x)    #low performance
    else:
        x = TrainingParams.ACTIVATION_FUN()(x)
    
    if AlgParams.USE_POOLING:
        x = MaxPooling1D(pool_size=AlgParams.CNN_POOL_SIZE, strides=1, padding='same')(x)
    
    return x   

def makeDenseBlock(x, denseSize, doDropout):
    x = TimeDistributed(Dense(denseSize))(x)
    if doDropout:
        x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)
    x = TrainingParams.ACTIVATION_FUN()(x)

    return x

def make1DModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE, name='input')
    print('inputshape', AlgParams.INPUT_SHAPE)
    x = protImg

    # shared dilated layers 
    channelSize = 64 
    dilationRate = AlgParams.INIT_CNN_DILATION_SIZE
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 64
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 64
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)


    # Task specific layers task 1 and 2 and 3 
    #x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    dilationRate = dilationRate * 2
    channelSize = 64
    x1 = make1DBlock(x, channelSize, dilationRate)
    x2 = make1DBlock(x, channelSize, dilationRate)
    x3 = make1DBlock(x, channelSize, dilationRate)
    x4 = make1DBlock(x, channelSize, dilationRate)
    x5 = make1DBlock(x, channelSize, dilationRate)
    
    dilationRate = dilationRate * 2
    channelSize = 32
    x1 = make1DBlock(x1, channelSize, dilationRate)
    x2 = make1DBlock(x2, channelSize, dilationRate)
    x3 = make1DBlock(x3, channelSize, dilationRate)
    x4 = make1DBlock(x4, channelSize, dilationRate)
    x5 = make1DBlock(x5, channelSize, dilationRate)

    x1 = TimeDistributed(Dense(32, activation='relu'))(x1)
    x1 = Dropout(TrainingParams.DROPOUT_RATE)(x1)
    x1 = BatchNormalization()(x1)

    x2 = TimeDistributed(Dense(32, activation='relu'))(x2)
    x2 = Dropout(TrainingParams.DROPOUT_RATE)(x2)
    x2 = BatchNormalization()(x2)

    x3 = TimeDistributed(Dense(32, activation='relu'))(x3)
    x3 = Dropout(TrainingParams.DROPOUT_RATE)(x3)
    x3 = BatchNormalization()(x3)

    x4 = TimeDistributed(Dense(32, activation='relu'))(x4)
    x4 = Dropout(TrainingParams.DROPOUT_RATE)(x4)
    x4 = BatchNormalization()(x4)

    x5 = TimeDistributed(Dense(32, activation='relu'))(x5)
    x5 = Dropout(TrainingParams.DROPOUT_RATE)(x5)
    x5 = BatchNormalization()(x5)

    x1 = TimeDistributed(Dense(32, activation='relu'))(x1)
    x1 = BatchNormalization()(x1)

    x2 = TimeDistributed(Dense(32, activation='relu'))(x2)
    x2 = BatchNormalization()(x2)

    x3 = TimeDistributed(Dense(32, activation='relu'))(x3)
    x3 = BatchNormalization()(x3)

    x4 = TimeDistributed(Dense(32, activation='relu'))(x4)
    x4 = BatchNormalization()(x4)

    x5 = TimeDistributed(Dense(32, activation='relu'))(x5)
    x5 = BatchNormalization()(x5)

    x1 = TimeDistributed(Dense(16, activation='relu'))(x1)
    x1= BatchNormalization()(x1)

    # kiki gewijzigd, de loss moet nu in de vorm van een dictionary
    # ppi = Dense(1, activation='sigmoid', name='ppi')(x1)
    # rsa = Dense(1, activation='relu', name='rsa')(x2)
    # sheet = Dense(1, activation='linear', name='sheet')(x3)
    # helix = Dense(1, activation='linear', name='helix')(x4)
    # coil = Dense(1, activation='linear', name='coil')(x5)

    ppi = TimeDistributed(Dense(1, activation='sigmoid'), name='ppi')(x1)
    rsa = TimeDistributed(Dense(1, activation='softplus'), name='rsa')(x2)

    # Multi-class output with softmax activation
    concatenated_output = concatenate([x3, x4, x5], axis=-1)
    SS3 = TimeDistributed(Dense(3, activation='softmax'), name='SS3')(concatenated_output)

    # Define the model
    model = Model(inputs=protImg, outputs=[ppi, rsa, SS3])
    model.summary()

    # # Define the model
    # model = Model(inputs=protImg, outputs=[ppi, rsa, sheet, helix, coil])
    # model.summary()
    
    return model


def performTraining():
    ##USE_2D_MODEL = True
    #LossParams.setLossFun(LossParams.MEAN_SQUARED)
    #no-padding
    #AlgParams.USE_DROPOUT = False
    #AlgParams.USE_BN = False
    ##AlgParams.USE_POOLING = True
    #TrainingParams.KERNAL_INITIALIZER = GlorotNormal
    #TrainingParams.ACTIVATION_FUN = ReLU
    """
    TrainingParams.KERNAL_INITIALIZER = he_uniform
    TrainingParams.ACTIVATION_FUN = PReLU
    #TrainingParams.BATCH_SIZE = 8 
    #TrainingParams.USE_EARLY_STOPPING = False    #if the training set is Serendip.
    
    TrainingParams.USE_BIAS = True
    LossParams.USE_WEIGHTED_LOSS = True
    LossParams.USE_PAD_WEIGHTS_IN_LOSS = True
    LossParams.LOSS_ONE_WEIGHT = 90.0 #0.90
    LossParams.LOSS_ZERO_WEIGHT = 10.0 #0.10
    LossParams.LOSS_PAD_WEIGHT = 0.0 #0.001
    """
    if AlgParams.USE_2D_MODEL: 
        model = make2DModel()
    else:
        model = make1DModel()
    TrainingParams.persistParams(sys.modules[__name__])
    PPITrainTestCls().trainModel(model)
    
    return

def performTesting():
    print(':) performTesting')
    TrainingParams.GEN_METRICS_PER_PROT = True
    tstResults = PPITrainTestCls().testModel()
    return tstResults

def performExplanation():
   ExplanationParams.NUM_BKS = 5
   ExplanationParams.NUM_TSTS = 3
   ExplanationParams.MAX_FEATURES = 10 
   PPIExplanationCls.explainModel()
        
if __name__ == "__main__":
    AlgParams.initAlgParams()
    if AlgParams.USE_EXPLAIN:
        performExplanation()
        sys.exit(0)
    if not AlgParams.ONLY_TEST:
        performTraining()
    performTesting()
    
   
