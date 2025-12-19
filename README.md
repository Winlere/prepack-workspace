# Prepack-workspace

A workspace for CS 498 Machine Learning System Course Project.

## Usage

```
git clone --recursive git@github.com:Winlere/prepacking-workspace.git
conda env create -f prepack/environment.yml -n prepack
conda activate prepack
pushd dataset
bash setup.sh
popd
```

## Reproduce Experiment

```
cd prepack_dynamic
bash run_exp.sh ../assets/ ../dataset/realdata_downsample/*.csv
```

If you see the following output, the experiment has started

```
Usage: run_exp.sh output_prefix xxx.csv yyy.csv ...                                        
Remaining argument: /u/wzhan/prepack-workspace/dataset/combined/mmlu_azure_ts.csv          
Output path: /u/wzhan/prepack-workspace/assets/raw_log/ONE_CLICK_A100/mmlu_azure_ts.txt    
Remaining argument: /u/wzhan/prepack-workspace/dataset/combined/mmlu_azure_ts_scaled.csv   
Output path: /u/wzhan/prepack-workspace/assets/raw_log/ONE_CLICK_A100/mmlu_azure_ts_scaled.txt                                                                    ```