# Protein Interface Prediction Using Multi-Task Learning and ProtT5-XL Embeddings

## Overview
This repository contains the code and data for our research on enhancing protein-protein interface (PPI) prediction using multi-task learning and ProtT5-XL embeddings. Proteins are complex molecules essential for biological functions, and understanding their interactions is crucial but often challenging due to limited annotations.

## Objective
The main goal of this project is to improve the accuracy of PPI prediction by leveraging multi-task learning and ProtT5-XL embeddings. This approach helps to overcome the limitations posed by sparsely annotated datasets.

## Dataset
- **Training Dataset:** PIPENN training dataset (BioDL_P_TR), which is sparsely annotated with only 11.9% interface class annotations.
- **Test Datasets:** Independent test datasets (BioDL_P_TE and ZK448_P).

## Methodology
### Multi-Task Learning
We employ multi-task learning to share information between the main task (PPI prediction) and two auxiliary tasks:
- **Relative Solvent Accessibility (RSA)**
- **Secondary Structure (SS3)**: prob_sheet, prob_helix, prob_coil

### Model Configurations
Three models with different loss-weight configurations for PPI, RSA, and SS3 were trained. The final model integrates optimized architectural features and the best-performing loss-weight configurations.

## Results
- **Performance:** The optimized model achieves competitive results on the ZK448_P benchmark set.
- **Auxiliary Tasks:** While SS3 shows promising results, integrating RSA and SS3 does not significantly improve PPI prediction performance.

## Insights
This research provides valuable insights into combining multi-task learning with ProtT5-XL embeddings, offering a robust approach for enhancing PPI predictions in datasets with limited annotations.


## Running existent multi-task algorithms 

- **Step 1:** Go to ~/pipenn-exp/mtnet-ppi and choose the Model architecture you want to use
```bash

# In this case dsLabel is 'Biolip_P': choose the target columns 
    def setExprFeatures(cls, featureColumns, dsLabel):
        ## labels are assigned here
        ## Epitope is changed, but in the future we will change other dataset and its labels
        keyFeatureDict = {
            'Cross_BL_A_HH': ['AliSeq', ['Interface1']],
            'Cross_BL_P_HH': ['AliSeq', ['Interface1']],
            'Cross_HH_BL_P': ['sequence', ['p_interface']],
            'HH_Combined': ['AliSeq', ['Interface1']],
            'Homo_Hetro': ['AliSeq', ['Interface1']],
            'Homo': ['AliSeq', ['Interface1']],
            'Hetro': ['AliSeq', ['Interface1']],
            'Biolip_P': ['sequence', ['p_interface', 'rel_surf_acc', 'prob_sheet', 'prob_helix', 'prob_coil']],
            'Biolip_S': ['sequence', ['s_interface']],
            'Biolip_N': ['sequence', ['n_interface']],
            'Biolip_A': ['sequence', ['any_interface']],
            'UserDS_A': ['sequence', ['any_interface']],
            'UserDS_P': ['sequence', ['any_interface']],
            'UserDS_N': ['sequence', ['any_interface']],
            'UserDS_S': ['sequence', ['any_interface']],
            'UserDS_E': ['sequence', ['any_interface']],
            'Epitope': ['AliSeq', ['Interface1']],
            'COVID_P': ['sequence', ['p_interface']],
        }
        cls.EXPR_DATASET_LABEL = dsLabel
        keyFeatures = keyFeatureDict.get(cls.EXPR_DATASET_LABEL)
        cls.SEQ_COLUMN_NAME = keyFeatures[0]

        ## kiki
        cls.LABEL_COLUMNS = keyFeatures[1]
        logger.info('The chosen labels are: {}'.format(keyFeatures[1]))
        cls.FEATURE_COLUMNS = featureColumns
        cls.setCondParams()
        return
```

- **Step 2:** Choose the belonging utils folder and change the name to 'utils'
```bash
mv source_folder_name utils
```

- **Step 3:** Go to the ~/pipenn-exp/jout directory and run:
```bash
python mtnet-ppi.py
```

## Running own multi-task algorithm 
- **Step 1:** ~/pipenn-exp/utils/PPIDataset.py
```bash
DatasetParams.MULTI_TASK_MODEL = True #False

# To use embeddings as input
DatasetParams.ONE_HOT_ENCODING = False #True
```
- **Step 2:** ~/pipenn-exp/utils/PPIDataset.py
```bash
DatasetParams.MULTI_TASK_MODEL = True #False

# To use embeddings as input
DatasetParams.ONE_HOT_ENCODING = False #True
```
- **Step 3:** ~/pipenn-exp/utils/PPIDataset.py
```bash
# If you use SS3 as third task. Your tasks are [ppi, rsa, SS3]
DatasetParams.SS3_USED = True #False
```
- **Step 4:** ~/pipenn-exp/utils/PPIDataset.py
```bash
# If you use three tasks and only the first task is binary classification [ppi, rsa, SS3]
DatasetParams.BINARY_MASK = [1, 0, 0]

# This is something you could print
DatasetParams.OUTPUT_COLUMNS = ['ppi', 'rsa', 'ss3']
```
- **Step 5:** ~/pipenn-exp/mtnet-ppi/mtnet-ppi.py and ~/pipenn-exp/jout/mtnet-pi.py
```bash
# If one label has 3 columns (like in SS3) count this into your total dimension. For [ppi, rsa, SS3] it is 5
Algparams.LABEL_DIM = 5 
```











## Keywords
- Protein interface prediction
- PPI
- Embedding
- Multi-task learning

