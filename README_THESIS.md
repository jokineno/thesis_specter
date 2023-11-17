# Training scripts and data 

### Data 
- Folder `thesis_data` includes all training files
- Under `thesis_data` there are following main folders 
```bash
ls /home/olli/UniHY/thesis_specter/thesis_data/training/ 
data.json  metadata.json  README.md  test.txt  train.txt  val.txt
```
```bash
ls -R /home/olli/UniHY/thesis_specter/thesis_data/inference/ 
inference:
models  results

inference/models:
finnish_bert_base_cased_v1  README.md  sbert-cased-finnish-paraphrase

inference/models/finnish_bert_base_cased_v1:
finnish_bert.tar.gz

inference/models/sbert-cased-finnish-paraphrase:

inference/results:
```
`models` files for baseline models 
`results` are results from baseline models prediction. These are later used in scidocs 

```bash
ls /home/olli/UniHY/thesis_specter/thesis_data/preprocessed/ 
data-metrics.json  data-test.p  data-train.p  data-val.p
```
=> These are training files required for script `4-run-training-script-thesis.sh ` 



### Create training files
When creating training files you need to provide three files 
-  metadata.json 
- data.json
- train.txt, val.txt and test.txt 

If you're creating only samples use filtered citation data but full metadata.
It's important that the cited articles are found from the metadata file. 

```bash
Demo version using only samples.
./3-create-training-files-thesis.sh --demo --install_packages

Full version install packages: 
./3-create-training-files-thesis.sh --install_packages

Full version 
./3-create-training-files-thesis.sh
```


### Run training script 


## TODOs 
- In SPECTER the data/vocab/ dir contains tokens and other text files. 
- tokens.txt length is around 240k 
- Instead scibert vocab.txt length is around 30k => Why there is a mismatch. 
- Check that finnish_bert.tar.gz can be used as model weights 
- Check that correct vocabulary is used 
- tokens vs vocab.txt
- 2023-04-17 07:22:11,419 - INFO - allennlp.common.params - dataset_reader.token_indexers.bert.do_lowercase = True
=> Check if do_lowercase should be False in case of finnish bert 

```bash
Demo version
./4-run-training-script-thesis.sh --demo

Full version 
./4-run-training-script-thesis.sh
```
