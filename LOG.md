# project log

- [toc]

## TODO

- remove try-except block

## event type:

- format: `event`: `where`: `what`
	- ignore `event` case

- `update`: update **existed** object
	- ex: rename file or variable
- `new`: **new** features
- `fixed`: **bug fixed**
- `bug`: bug found but **yet fix**
- `doc`: add comment
- `other`: TBD

## Version 7

### 11/29 - jefflin

- Add sample YAML, update README.md and fix `demo_post` bug
    - `new`: Add `sample_trainV7.yaml`
    - `update`: Update `demo/README.md`
    - `update`: In `train.py`: Fix `demo_post` bug

- Update README.md

- Update traing envirnoment
    - `new`: In `train.py`: add `--demo-post` argument
    - `fixed`: in `postprocessor.py`: fix input_path bug
    - `update`: In `README.md`: update training description
    - `new`: In `.gitignore`: Add `logs/`

### 11/27 - fan

- doc: modified and detailed `README.md`

### 11/26 - fan

- refactor: reconstruct project
	- `update` divide project into: `demo`, `src`, `doc`, `dataset`
		- `src`: source code
		- `demo`: demo scripts
		- `doc`: documents
		- `dataset`: dataset for training
	- `remove`: remove some file in `doc`

### 11/26 - jefflin

- Update initial position
    - `update`: In `demo_utils.py`: update initial position

- Add combine input and output txt files
    - `new`: In `demo.py` and `demo_utils`:
        - add combine input and output txt files
    - `update`: In `project.py` and `demo.py`:
        - move construction env to reduce execution time
    - `update`: In `demo.py` and `demo_utils`:
        - improve demo post-processing
    - `other`: In `calligraphy_transform.py`: comment print

- Add translation to demo processing
    - `update`: In `demo.py` and `demo_utils.py`: update parameters
    - `new`: In `demo.py` and `demo_utils.py`:
        - add `translation` and add initial positon to the end of txt file

### 11/24 - jefflin
- Add copy2usb and update argument setting
    - `fixed`: In `project.py` and `demo.py`: merge argument_setting
    - `new`: In `demo.py` and `demo_utils.py`: add copy2usb and `--usb-path` argument
    - `update`: In `demo.py`: update `demo_main` parameters

### 11/18 - jefflin
- fix efficient demo bug
    - `fixed`: In `demo.py`: fix efficent demo bug about get test path in `data_env`

### 11/18 - fan

- feat: improve execution time and keep model in gpu
	- `new`: In `eval.py`: add `demo_eval` for demo only
		- remove unnecessary functions and features for demo
	- `new`: In `demo.py`: now can call `efficient_demo` with argument `--efficient`
		- `--efficient`: build envirnoment first in order to avoiding to re-instantiate model
		- `--content-loss`: compute content loss or not


### 11/17 - fan

- feat: compute execution time and then analyze them.
	- `new`: In `demo_utils.py`: add `timer` function to compute execution time
		- add `--timer` argument to activate timer
	- `new`: In `demo.py`: now can analyze the execution time with `--timer`
	- `doc`: In `.gitignore`: add `*.tar.gz` avoid to push dataset to github

### 11/17 - jefflin

- Improve the demo process
    - `new`: In `demo/demo_utils.py`: 
        - add `--demo-post` argument to switch to demo post-processing
    - `update`: In `demo/demo.py`:
        - remove useless libraries
        - fixed test_path bug
        - improve output message
    - `update`: In `demo/README.md`: improve content
    - `update`: In `postprocessor.py`, `post_utils.py` and `README.md` in `postprocessing/`:
        - add `demo_post` argument to switch to demo post-processing

- fix out2csv bug
    - `fixed`: In `out2csv` in `utils.py`:
        - fix np.squeeze bugs when the shape of output is [1, 1, 150, 6]

### 11/16 - jefflin
- Solve demo bugs
  
- Add demo/README.md

- Fix input_path bug

### 11/16 - fan

- Solve compatible problem
	- `rename`: `test.py` -> `eval.py`
	- `fix`: in `demo_util.py`: add parent path and import `utlis.StorePair`

### 11/16 - jefflin

- Automatic pre-processing, testing and post-processing
    - `new`: In `demo/`: add demo program
    - `update`: In `preprocessing/`: compatible
    - `update`: In `postprocessing/':
        - compatible
        - Group files into folders

## Version 6

### 10/14 - fan

- fix early stop argument missed
	- `fixed`: add `checkpoint` to early stop `__call__()`

### 10/5 - jefflin

- fix postprocessing bug about test_all
    - `fixed`: In `postprocessor.py`: fix bug about test_all

### 10/3 - fan

- Now can load pickle file in train normally
	- `fixed`: In `train.py`: add torch.load before try-except block and commented load_state_dict
	- `fixed`: In `utils.py`: pass checkpoint to `EarlyStop` to save iteration info

### 9/22 - jefflin

- fix preprocessing bug about np.load
    - `fixed`: In `postprocessor.py`: fix bug about test_all
    - `fixed`: In `verification.py`: get char number by abspath
    - `fixed`: In `preprocess.py`:
        - fix np.load bug
        - comment saving target data
    - `update`: In `utils.py`:
        - change default input path
        - delete `--test-target-path`

### 9/20 - fan

- Now can sort by numerical suffix and fix a tiny bug
	- `new`: In `utils.py`: add `_key_func` to sort integer key
	- `update`: In `utils.py`: replace `sorted` by `.sort()`
	- `fixed`: In `train.py`: now can get version 0
	- `doc`: In `doc/sampleV4.yaml`: add hint about arugment `load` and remove error message
		- **DO NOT declare `version` in doc file, modified `load` instead**

- Make dataset more memory efficiency
	- `update`: In `dataset.py`: now target dictionary only store overlapped file

### 9/19 - jefflin

- fix postprocessing bug about test whole character output length
  - `update`: In postprocessor.py:
      - adapt the test directory name generally
      - update the process to speed up
  - `update`: In preprocess.py and utils.py:
      - fix the test directory path bug
      - TODO: 解決225字以外的 testing data 的 target，長度問題及存放位置
  - `update`: In verification.py and README.md in preprocessing/:
      - Update explanation

### 9/19 - Angelowen

- fix test_error.py bug
  - `new`: In postprocessing/postprocessing.py: inverse the length of the data when combine strokes into a word after using interpolate adding way 
  - `update`: .gitigonre add doc/*.yaml
  - `update`: test_error.py storing path

### 9/15 - jefflin

- fix postprocessing bug about test_all txt file
  - `fixed`: In test.py: add inputs inverse_transform when test normalized
  - `fixed`: In postprocessing.py: fix bug about test_all txt file
  - `update`: In verification.py: update
  - `other`: In postprocessing.py: add TODO to delete the uselee line about test_all file

- fix test out2csv and postprocessing bug
  - `fixed`: In test.py: add `--test-num` args
  - `fixed`: In test.py: fix out2csv bug about batch size
  - `update`: In postprocessing.py: add axis2img and csv2txt to test files
  - `fixed`: In stroke2char.py: fix test file stroke order bug
  - `new`: In verification.py: verify the all `test_all_target.txt` is correct or not

### 9/14 - Angelowen

- fix test_error.py bug
  - `update`: In `test_error.py`: add content loss
  - `fixed`: In `test_error.py`: fix args.load bug

### 9/14 - jefflin

- fix test.py bug, fix out2csv bug
  - `fixed`: In `test.py`: fix args.load bug
  - `fixed`: In `test.py` and `utils.py`: update out2csv
  - `fixed`: In `utils.py`: fix keepdim bug
  - `new`: In `postprocessor.py`: add axis2img to test all files

### 9/14 - fan

- make model compatible temporarily
	- `fixed`: In `train.py`, `test.py` add try-except block to load model
	- `fixed`: In `test.py` add `--stroke-length` args
		- same as `train.py --stroke-length`

### 9/13 - jefflin

- fix dataset bug and preprocess bug
    - `fixed`: In dataset.py: fix np.load bug
    - `fixed`: In preprocess.py: fix bug when only one line data
    - `update`: In preprocessing/readme.md: update new feature

### 9/13 - fan

- config_loader can load doc now
	- `fixed`: In utils.py, `config_loader` will set `--load` to True when execute `test.py --doc doc_path`

- make `test.py` avaliable to current environment
	- `update`: Add `writer_builder` and `criterion_builder` to `test.py`
		- **`--criterition` defaults to MSE**
	- `remove`: remove NormScaler which means data is uncompression
	- `remove`: remove cross_validation


- Add inverse func which convert npy back to csv file
- `new`: In `preprocessing/preprocess.py`, add `_npy2csv` to convert npy to csv file
	- `--root-path` to control root of npy directories
- `update`: In `preprocessing/utils.py`, add `--keep` args to keep original data while converting or inverse converting
- `update`: In `preprocessing/preprocess.py`, add tqdm to `_npy2csv` to show progress bar while converting
- `rename`: In `preprocessing/preprocess.py`, rename `csv2npy` to `_csv2npy`

### 9/12 - fan

- **This commit is unstable**
- `update`: In `preprocessing/preprocess.py`, save inputs file as npy file to improve performance
	- inputs, target, test to npy file
	- **output remain csv file**
- `update`: In `dataset.py`, load and search npy file, instead of csv file
- `new`: In `preprocessing/preprocess.py`, add function `csv2npy` convert 
- `new`: In `preprocessing/utils.py`, add argv `--convert` to call `csv2npy`
	- `--root-path` to control `csv2npy` input path

### 9/12 - jefflin

- Add extend stroke lenght by interpolation in preprocessing
    - `new`: Add extend stroke lenght by interpolation in preprocessing, 
        - and add noise to the tail
    - `update`: In .gitignore: delete `stroke_statistics` function

### 9/11 - fan

- Add new mechanic in ADBPN:
	- `new`: Add weighted features
	- `update`: update DBPN series coding style
	- `update`: ADBPN now is use channel feature instead of col features
		- to call col features, set `atten=col`

### 9/3 - Angelowen

- adding the content loss to new model training
	- `update`: In train_error.py: add `content_loss` to MSE loss value 
	- `update`: In train_error.py: add `pbar_postfix['Content loss']` 
	- `update`: In train_error.py: add `args.scheduler` and `pbar_postfix['lr']` to record learning rate

### 9/2 - jefflin

- Change the strategy of Early-Stop
	- `update`: In train.py and train_error.py: change `threshold` argument default value and change the order of parameters in `EarlyStopping`
	- `update`: In `EarlyStopping` in utils.py: Change the strategy of Early-Stop. eg, val_loss decrease less than 0.1% (default)
	- `update`: In utils.py: Delete the comment of `csv2txt`

### 8/31 - fan

- Now config will save as original document file when call args.doc
	- `new`: In utils.py: now config will save in version directory with original format
		`.json` -> `.json`, `.yaml` -> `.yaml`
	- `update`: In README.md, add default lr which is based on paper

### 8/30 - Angelowen

- fix fix DBPN/init.py bug and train_error bug
  - `fixed`: train_error.py: fix out2csv bug
  - `fixed`: In DBPN/init.py add ADBPN to call 
  - `update`: In .gitignore: Add `output-v5-1c779cab/` folder

### 8/30 - fan

- Merge to master branch
- Add new model ADBPN and remove output dir
	- `new`: In model/DBPN/models.py: add new model ADBPN
	- `new`: In utils.py: now can call new model by `--model-name ADBPN`
		- model args almost same as DBPN but add `col_slice`, `stroke_len`
	- `removed`: remove output directory
	- `update`: add model args in readme.md

### 8/27 - jeff

- fix lr_scheduler bug and move out2csv place to decrease the I/O times
  - `fixed`: In train.py and train_error.py: fix lr_scheduler bug
  - `update`: In train.py: move out2csv to the end of epoch loop to decrease the I/O times
  - `fixed`: In utils.py in `writer_builder`: check the log_root exists, or create a new one
  - `update`: In .gitignore: Add `output*/` folder

### 8/23 - angelo

- fix line39 bug
  - `fixed`: In train_error.py line39, fix bug

### 8/24 - fan

- Fix iteration bug
	- `fixed`: In train.py, fixed iteration count bug

- Now can view current learing rate with lr_scehduler
	- `fixed`: In utils.py, in `writer_builder`, sort variable `version` now
	- `fixed`: In train.py, add pbar_postfix to control postfix value

### 8/22 - fan

- Now can call dbpn by `--model-name dbpn`
	- `new`: In model/DBPN.py, Add DBPN 
	- `new`: In utils.py, add `dbpn` in model_builder
- `update`: In model/DBPN.py, DBPN series can instantiate normally when scale factor is greater than 1

```python
DBPN-S:
	stages=2,
	n0=64,
	nr=18

DBPN-SS:
	stages=2,
	n0=128,
	nr=32
```

### 8/17 - fan

- Now model parameter and config would be save in log/\*/version\*/
	- `update`: In utils.py: `writer_builder` now would return writer and store path
		- return value: `writer` -> `(writer, model_path)`
	- `update`: In train.py: model_path would be define by `writer_builder`
	- `update`: In utils.py: `model_config` will save config to version directory
	- `update`: In train.py: progress bar can show current learning rate
	- `new`: In loss.py: add rmse loss function
	- `fixed`: In train.py: iteration will show value normaly

### 8/14 - fan

- Now can store model without early stopping
- Add learning rate scheduler and add huber loss function
- modified writer every iteration and epoch
	- `new`: In train.py: add new argument to adjust new builder
	- `fixed`: In train.py: now can store without earlystopping
	- `new`: In utils.py: add criterion builder
		- `--criterion`: choose loss function
	- `new`: In utils.py: add scheduler builder
		- `--scheduler`: to choose step or multiple step scheduler
		- `--lr-method`: to choose scheduler
		- `--step`: set update step(epoch)
		- `--factor`: set decreate factor

- TODO: modified out2csv timing

### 8/11 - jeff

- Add getting more model data from each epoch.
Update: In train.py:
		Comment target scaler and uncomment pred denormalize
Update: In sampleV3.yaml: Update arguments value
Add: In .gitignore: Add output_tmp/ for testing data
Add: In train.py:
		Add argument 'out-num' to set the number of model data to get
Add: In train.py and out2csv in utils.py:
		Add getting more model data from each epoch

- Add getting more model data from each epoch.

Update: In train.py:
		Comment target scaler and uncomment pred denormalize
Update: In sampleV3.yaml: Update arguments value
Add: In .gitignore: Add output_tmp/ for testing data

Add: In train.py:
		Add argument 'out-num' to set the number of model data to get
Add: In train.py and out2csv in utils.py:
		Add getting more model data from each epoch
		
### 8/11 - fan

- Dataset cross validation big update
	- New: In dataset.py: add new function 'cross_validation', split data by randomsplit instead sampler
	- Rename: In dataset.py: 'cross_validation' -> '_cross_validation'
	- Update: In train*: update loading dataset to fit in new CV function
- now different dataset have own progress bar but leave train set only. Update doc
	- Update: In train.py progress modified
	- Update: update doc parameters

### 8/10 - fan

- compression tensor to range `[-1, 1]`

### 8/09 - jeff

- Add modularization of postporcessing, early-stop threshold argument, out2csv taking specific data

1. Modularize postporcessing and add into train process
2. Add Early-Stop threshold argument
3. Add out2csv take specific valid data
4. Fix out2csv bug about output path

New: In train.py and train_error.py: Add postprocessing in the end
New: In train.py and EarlyStopping in utils.py: Add argument threshold
Update: In train.py: Move out2csv from train section to valid section
Update: In utils.py: Fix bug about output path
Update: In dataset.py: Change valid Sampler to SequentialSampler
Comment: In utils.py: Comment csv2txt function and move to the end
Update: In postprocessing/: Modularization

### 8/08 - fan

- Model-args now can be assign value by pairs
	- Update: In doc: update sample files which are available to fit in current version
	- New: In utils.py: Add StorePair to custom argparse.action object
	- Update: In train.py and test.py: modified model-args from list to dict, namely pass kwargs to model_builder
	- Comment: In dataset.py: Add todo Comment

### 8/07 - jeff

- Fixed bugs in postprocessing about stroke2char
	1. Fix bugs in postprocessing about stroke2char
	2. Optimize some process
