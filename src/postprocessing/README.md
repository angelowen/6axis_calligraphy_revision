# Postprocessing for 6axis_revision Project
> 開發及維護人員: jefflin

> 最新更新時間: 2020/11/16

- [Postprocessing for 6axis_revision Project](#postprocessing-for-6axis_revision-project)
	- [Quik Run](#quik-run)
	- [Arguments Setting](#arguments-setting)
			- [Example](#example)
	- [Program Running Rule](#program-running-rule)
		- [**注意事項**](#注意事項)

## Quik Run
- Run the following command for running postprocess without any arguments
	```
	python postprocessor.py
	```
- Run the following command for verify all test_all_targe.txt is same as original 6axis txt file or not.
	```
	python verification.py
	```

## Arguments Setting

```
--input-path INPUT_PATH (default: ./home/jefflin/6axis/)
--save-path SAVE_PATH (default : `./output` )
--demo-post DEMO_POST (default: False)
```

#### Example
```
python postprocessor.py --save-path ./output_path --input-path ./6axis/
```

## Program Running Rule
採遞迴方式取出預設資料夾底下所有層的 csv 檔，完成以下三個功能:
1. stroke2char: 把 `test_` 開頭的 csv 檔，依筆畫順序合併成單一完整書法字的 csv 檔，輸出檔名為 `test_all_(target|input|output).csv`
2. axis2img: 把 target, input, output 三部分生成 2D 細線化圖示，以進行比較
3. csv2txt: 把所有 csv 檔轉成機器手臂可執行的 txt 指令檔

### **注意事項**
- 所有輸出檔之書法筆畫長度皆被還原成原本的長度，長度不再是 150
- test data 的檔名須為 `test_` 開頭
- 檔名須以 `_target.csv`, `_input.csv` 或 `_output.csv` 為結尾形式命名
- 因此，同一層資料夾下的 csv 檔個數須為 3 的倍數，不然會報錯
- 存放 test 輸出資料的資料夾名稱的結尾須為`字元編號`，且長度須為 **3**
  - eg. `test_042`、`char00042`、`771`
- 輸出檔會照檔案類型分類到 `pic/`、`txt/`、`test_char/` 三個資料夾。如果參數 `--demo-post` 設為 `False`，則只產生 Demo 過程所需之資料，並存放於 `test_char/`，不會產生單一筆畫的執行檔及比較圖檔。其中 Demo 過程所需之資料分別為以下四項:
  - test_all_compare.png
  - test_all_input.txt
  - test_all_output.txt
  - test_all_target.txt