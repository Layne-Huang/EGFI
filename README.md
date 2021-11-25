# EGFI
EGFI: Drug-Drug Interaction Extraction and Generation with Fusion of Enriched Entity and Sentence Information  
Cite as:	
Lei Huang, Jiecong Lin, Xiangtao Li, Linqi Song, Zetian Zheng, Ka-Chun Wong, EGFI: drug–drug interaction extraction and generation with fusion of enriched entity and sentence information, Briefings in Bioinformatics, 2021;, bbab451, https://doi.org/10.1093/bib/bbab451
## Abstract
The rapid growth in literature accumulates diverse and yet comprehensive biomedical knowledge hidden to be mined such as drug interactions. However, it is difficult to extract the heterogeneous knowledge to retrieve or even discover the latest and novel knowledge in an efficient manner. To address such a problem, we propose EGFI for extracting and consolidating drug interactions from large-scale medical literature text data. Specifically, EGFI consists of two parts: classification and generation. In the classification part, EGFI encompasses the language model BioBERT which has been comprehensively pre-trained on biomedical corpus. In particular, we propose the multihead self-attention mechanism and packed BiGRU to fuse multiple semantic information for rigorous context modeling. In the generation part, EGFI utilizes another pre-trained language model BioGPT-2 where the generation sentences are selected based on filtering rules. We evaluated the classification part on “DDIs 2013” dataset and “DDTs” dataset, achieving the F1 scores of 0.842 and 0.720 respectively. Moreover, we applied the classification part to distinguish high-quality generated sentences and verified with the existing growth truth to confirm the filtered sentences. The generated sentences that are not recorded in DrugBank and DDIs 2013 dataset also demonstrated the potential of EGFI to identify novel drug relationships.

![image](https://github.com/Layne-Huang/EGFI/blob/main/Classification_part.png)
![image](https://github.com/Layne-Huang/EGFI/blob/main/Generation_part.png)
<!-- ## Model Structure
<div align="center">
<p><img src="Classification Part.pdf" width="800" /></p>
</div>

<div align="center">
<p><img src="GenerationPart.pdf" width="800" /></p>
</div> -->
## Environment
The test was conducted in the linux server with GTX2080Ti and the running environment is as follows:
* python 3.7
* pytorch 1.6.0
* transformers 3.2.0
* sklearn 0.23.1
* Cuda 10.0.130
## Data
### DDI data
The DDIs 2013 datset is avilable in ./Classification Part/data/ddi file.
### DTI data
Download the complete DTI dataset from https://portland-my.sharepoint.com/:f:/g/personal/lhuang93-c_my_cityu_edu_hk/EuM6bN22cpRBj1qUuEpHTCQBDDQqFLpBUqlNBt4M4_N8OQ?e=7P2TsI and put it in ./Classification Part/data/dti/.
## How to run
EGFI consists of classification part and generation part.
### Classification Part
#### DDI data
1. Run ./Classification Part/data/ddi/data_prepare_tsv.py to preprocess the DDI dataset.
2. Run ./Classification Part/train/ddi/train_ddi_epoch.py to train the classification part of EGFI with different learning rates and get the results on test dataset.
#### DTI data
1. Run ./Classification Part/data/dti/data_prepare.py to preprocess the DDI dataset.
2. Run ./Classification Part/train/ddi/train_dti_epoch.py to train EGFI with different learning rates and get the results on test dataset.
### Generation Part
1. Run ./Generation Part/train.py to train the generation model(BioGPT-2) of EGFI.
2. Run ./Generation Part/generate.py to generate sentences.
3. Run ./Classification Part/data/ddi/data_prepare_tsv.py to preprocess the generated senetnces.
4. Run ./Generation Part/test_textgeneration.py to test the generated sentences. The sentences that have high scores may contain potential meaningful DTIs.  
