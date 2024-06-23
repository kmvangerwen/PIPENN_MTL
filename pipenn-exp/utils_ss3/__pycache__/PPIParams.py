import numpy as np

from tensorflow.python.keras.layers import ReLU, LeakyReLU, PReLU, ELU
from tensorflow.python.keras.initializers import GlorotUniform, GlorotNormal, he_uniform, he_normal, lecun_uniform, lecun_normal, \
                                                 Orthogonal, TruncatedNormal, VarianceScaling
from PPILogger import PPILoggerCls
from PPIDataset import PPIDatasetCls, DatasetParams
from PPILoss import PPILossCls, LossParams
from PPITrainTest import PPITrainTestCls, TrainingParams

class PPIParamsCls(object):
    FEATURE_COLUMNS_PROP = [
                       'RSA_q', 
                       'ASA_q', 
                       'length', 
                       'AliSeq',
                       'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 
                       ]
    EX_COLUMNS = [
                    #'normalized_length',
                    #'domain',
                    ]
        
    #EXPR_FEATURE_COLUMNS = FEATURE_COLUMNS_PROP
    #EXPR_FEATURE_COLUMNS = DatasetParams.FEATURE_COLUMNS_ENH_WIN
    #EXPR_FEATURE_COLUMNS = DatasetParams.FEATURE_COLUMNS_ENH_NOWIN
    #EXPR_FEATURE_COLUMNS = DatasetParams.FEATURE_COLUMNS_STD_NOWIN
    EXPR_FEATURE_COLUMNS = DatasetParams.FEATURE_COLUMNS_BIOLIP_NOWIN
    #EXPR_FEATURE_COLUMNS = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    
    #datasetLabel = 'Homo_Hetro'
    #datasetLabel = 'Hetro'  
    datasetLabel = 'Biolip_N'
    
    @classmethod
    def setInitParams(cls, algorithmName, dsParam=EXPR_FEATURE_COLUMNS, dsExParam=EX_COLUMNS, dsLabelParam=datasetLabel):
        TrainingParams.initKeras('float64')
        logger = PPILoggerCls.initLogger(algorithmName)
        DatasetParams.EXPR_DATASET_LABEL = dsLabelParam
        dsParam = list(np.setdiff1d(np.array(dsParam), np.array(dsExParam), True))
        if DatasetParams.USE_COMET:
            TrainingParams.initExperiment(algorithmName)
        
        DatasetParams.setExprFeatures(dsParam)

        PPIDatasetCls.setLogger(logger)
        PPILossCls.setLogger(logger)
        PPITrainTestCls.setLogger(logger)
        DatasetParams.setExprDatasetFiles()
        TrainingParams.setOutputFileNames(algorithmName)
        
        TrainingParams.KERNAL_INITIALIZER = he_uniform
        TrainingParams.ACTIVATION_FUN = PReLU
        TrainingParams.USE_BIAS = False
        TrainingParams.BATCH_SIZE = 8 #8 > 16 > 32 > 64
        TrainingParams.USE_EARLY_STOPPING = True
        TrainingParams.NUM_EPOCS = 300 #200 #150
        TrainingParams.MODEL_CHECK_POINT_MODE = 1
        
        DatasetParams.FEATURE_PAD_CONST = 0.11111111
        #DatasetParams.SKIP_SLICING = True
        #DatasetParams.USE_VAR_LEN_INPUT = True
        
        LossParams.USE_WEIGHTED_LOSS = True
        LossParams.USE_PAD_WEIGHTS_IN_LOSS = True
        LossParams.LOSS_ONE_WEIGHT = 0.90
        LossParams.LOSS_ZERO_WEIGHT = 0.10
        #LossParams.setLossFun(LossParams.MEAN_SQUARED)
        LossParams.setLossFun(LossParams.CROSS_ENTROPY)
        #LossParams.setLossFun(LossParams.TVERSKY)
        
        return
    
    @classmethod
    def setShapeParams(cls, inputShape, labelShape):    
        TrainingParams.INPUT_SHAPE = inputShape
        TrainingParams.LABEL_SHAPE = labelShape
        
        return
   
