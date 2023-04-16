# Training scripts and data 


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

```bash
Demo version
./4-run-training-script-thesis.sh --demo

Full version 
./4-run-training-script-thesis.sh
```