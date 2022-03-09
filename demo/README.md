# Automatic processing for 6axis_revision Project
> <span style="color:red">**Please see the new verion in `6axis_revision/README.md` . Here is the old and WRONG version.**</span>

> 開發及維護人員: jefflin

> 最新更新時間: 2020/11/17

- [Automatic processing for 6axis_revision Project](#automatic-processing-for-6axis_revision-project)
  - [Quik Run](#quik-run)
  - [Arguments Setting](#arguments-setting)
      - [Example](#example)
  - [Program Running Rule](#program-running-rule)
    - [**注意事項**](#注意事項)
  - [Folder Structure](#folder-structure)

## Quik Run
- **請在 `demo/` 資料夾裡執行**
- Run the following command for running demo process without any arguments
	```
	python project.py
	```
- 可加入參數`--efficient`加速執行
- 加入參數`--gui`以GUI介面執行(EX:`python project.py --efficient --gui`)
- 介面打開後請選擇參數後按執行

## Arguments Setting
Merge the arguments of pre-processing, testing and post-processing. The details please see ```python demo.py -h``` message.

#### Example
```
python demo.py --version 0
```

## Program Running Rule
自動化修正過程，包含以下三部分
1. 前處理
   - 輸入字元編號及誤差範圍，就可以生成數量僅一筆的測試資料
2. 修正
   - 修正六軸 
3. 後處理
   - 合併筆畫，及檢測輸出長度是否正確

### **注意事項**
- 初次執行，請把 `demo.tar.gz` 裡的 `dataset/`、`logs/` 兩資料夾移到 `demo/` 資料夾底下。
- **請維持相同的資料夾環境**，否則容易報錯。 ( 詳見 [Folder Structure](#Folder-Structure) )
- 程式最後會驗證 target 輸出檔是否正確。如 cmd 輸出有顯示 **`Verification All Correct!!!`**，則代表正確，否則代表輸出檔有錯。
- 為有效降低執行時間，後處理只產生 Demo 過程所需之資料，並存放於 `test_char/`。其中 Demo 過程必需之資料，分別為以下四項
  - test_all_compare.png
  - test_all_input.txt
  - test_all_output.txt
  - test_all_target.txt

## Folder Structure

- `test.py`
- preprocessing/
- postprocessing/
- doc/
    - sample.yaml          ————————————————— doc
- demo/
    - `demo.py`
    - `demo_util.py`
    - dataset/             ——————————————————— root-path
        - 6axis/           ————————————————— input-path
            - char_00436_stroke.txt ...
        - test/             —————————————————— test-path
        - target/         ————————————————— target-path
    - logs/                  ———————————————————— logs-path
        - FSRCNN/
            - version_0/
                - FSRCNN_1x.pt
    - output/              ——————————————————— save-path
        - test_1_input.csv ...
        - test_char/
            - test_all_compare.png
            - test_all_input.txt
            - test_all_output.txt
            - test_all_target.txt
