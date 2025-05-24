# Comparing The Performance Of Different Image Inpainting Models

比較不同Inpainting Models在小型資料與有限訓練輪次下的效果

# 目錄

- [資料來源](#資料來源)
- [專案結構](#專案結構)
- [Requirements](#Requirements)
- [MADF](#MADF)
- [AOT-GAN](#AOT-GAN)
- [TFill](#TFill)
- [評估指標](#評估指標 )
- [Complexity](#Complexity)

# 資料來源

[Models](https://drive.google.com/drive/folders/1P4SREbA9FmBw9qNcfxGZJ381uwRHo8Nb?usp=sharing)


[Training & Validation set](https://drive.google.com/file/d/1pwkuX5Oy2IRfFY-JooWwhex8yBVQJZGO/view?usp=drive_link)


[Testing set](https://drive.google.com/file/d/1-5iaqRzkkjg0Uv2bt-eRJML5iJS7wGIl/view?usp=drive_link)


# 專案結構

```
├── 2-MADF/				# MADF模型與測試儲存位置 
├── 4_AOT_GAN/			# AOT-GAN模型與測試儲存位置 
├── 5-TFill/			# TFill模型與測試儲存位置 
├── 2-MADF.ipynb 		# MADF訓練與測試程式 
├── 4_AOT_GAN.ipynb 	# AOT-GAN訓練與測試程式 
├── 5-TFill.ipynb 		# TFill訓練與測試程式 
├── Loss計算.ipynb		# 評估指標程式
├── README.md 			# 專案說明文件
```

其他不相關檔案為測試用，無須處理

# Requirements

本專案提供的四個程式檔案皆為jupiter notebook形式，運行環境使用**Google Colab**執行，各檔案的依賴皆在程式內部提供。

# MADF

## 初始化設定

上傳資料集部分請將來源路徑替換成你的資料夾路絕對路徑，目的路徑請勿更動，後續程式依照此路徑執行、無須改動。

```
# 複製data資料集
source_path = "/content/drive/MyDrive/中興_深度學習/HW4/data" # 請替換成你的data資料集的絕對路徑
...

# 複製test資料集
source_path = "/content/drive/MyDrive/中興_深度學習/HW4/test" # 請替換成你的test資料集的絕對路徑
...
```

## Training

請更改**保存測試結果**區塊中的目的路徑，確保測試結果被正確保存

```
# 將目標路徑替換成你的儲存路徑
!cp -r /content/Pytorch-MADF/output/snapshot/default/ckpt your/saving/path
```

## Testing

請更改**保存訓練後的模型**區塊中的目的路徑，確保模型被正確保存

```
# 將目標路徑替換成你的儲存路徑
!cp -r /content/Pytorch-MADF/results your/saving/path
!cp -r /content/Pytorch-MADF/gt_results your/saving/path
```

## Over All

上述修正完成後即可依照順序正常執行

# AOT-GAN

## 初始化設定

上傳資料集部分請將來源路徑替換成你的資料夾路絕對路徑，目的路徑請勿更動，後續程式依照此路徑執行、無須改動。

```
# 複製data資料集
source_path = "/content/drive/MyDrive/中興_深度學習/HW4/data" # 請替換成你的data資料集的絕對路徑
...

# 複製test資料集
source_path = "/content/drive/MyDrive/中興_深度學習/HW4/test" # 請替換成你的test資料集的絕對路徑
...
```

## Training

請用下方程式替換src/trainer/trainer.py程式的第147行，確保訓練結束時至少保存一次模型參數：

```
if self.args.global_rank == 0 and ((self.iteration % self.args.save_every) == 0 or (self.iteration == self.args.iterations)):
```

請用下方程式替換src/trainer/trainer.py程式的第73行整個save()函數，確保儲存邏輯不會發生錯誤：

```
def save(self,):
	if self.args.global_rank == 0:
		print(f"\nsaving {self.iteration} model to {self.args.save_dir} ...")

		if (self.args.distributed == True):
			model_to_save_G = self.netG.module
			model_to_save_D = self.netD.module

		else:
			model_to_save_G = self.netG
			model_to_save_D = self.netD

		torch.save(
		model_to_save_G.state_dict(), os.path.join(self.args.save_dir, f"G{str(self.iteration).zfill(7)}.pt")
		)

		torch.save(
		model_to_save_D.state_dict(), os.path.join(self.args.save_dir, f"D{str(self.iteration).zfill(7)}.pt")
		)

		torch.save(
		{"optimG": self.optimG.state_dict(), "optimD": self.optimD.state_dict()},
		os.path.join(self.args.save_dir, f"O{str(self.iteration).zfill(7)}.pt"),
		)
```

請更改**保存測試結果**區塊中的目的路徑，確保測試結果被正確保存

```
# 將目標路徑替換成你的儲存路徑
!cp -r /content/experiments your/saving/path
```

## Testing

請更改**保存訓練後的模型**區塊中的目的路徑，確保模型被正確保存

```
# 將目標路徑替換成你的儲存路徑
!cp -r /content/Pytorch-MADF/results your/saving/path
!cp -r /content/Pytorch-MADF/gt_results your/saving/path
```

## Over All

上述修正完成後即可依照順序正常執行

# TFill

## 初始化設定

上傳資料集部分請將來源路徑替換成你的資料夾路絕對路徑，目的路徑請勿更動，後續程式依照此路徑執行、無須改動。

```
# 複製data資料集
source_path = "/content/drive/MyDrive/中興_深度學習/HW4/data" # 請替換成你的data資料集的絕對路徑
...

# 複製test資料集
source_path = "/content/drive/MyDrive/中興_深度學習/HW4/test" # 請替換成你的test資料集的絕對路徑
...
```

## Training

請用以下程式替換 dataloader/data_loader.py 91行到118行的elif內邏輯，取消mask預處理步驟：

```
elif mask_type == 3:
	# external mask from "Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV18)"

	if self.opt.isTrain:
		mask_index = random.randint(0, self.mask_size-1)

	else:
	mask_index = item
	mask_transform = transforms.Compose(
		[
			transforms.Resize([h, w])
		]
	)
	mask_pil = Image.open(self.mask_paths[mask_index]).convert('L')
	mask = mask_transform(mask_pil) 

	if self.opt.isTrain:
		mask = self._mask_dilation(mask)

	else:
		mask = np.array(mask) < 128
		mask = torch.tensor(mask).view(1, h, w).float()
		mask_pil.close()

	return mask, mask_type
```

訓練時用提供的範例替換train.sh的內容，並執行shell指令即可

請更改**保存測試結果**區塊中的目的路徑，確保測試結果被正確保存

```
# 將目標路徑替換成你的儲存路徑
!cp -r /content/checkpoints/place2 your/saving/path
```

## Testing

請更改**保存訓練後的模型**區塊中的目的路徑，確保模型被正確保存

```
for data_test in test_data_names:
	print("Saving: ", data_test)
	source_path = f"/content/results/{data_test}/{NAME}/test_latest/img_ref_out" # Colab 中的執行目錄
	dest_path = f"/content/drive/MyDrive/中興_深度學習/HW4/5-TFill/place2/result/{data_test}" # 請替換成你的預測結果的絕對路徑
	!cp -r $source_path $dest_path
```

## Over All

上述修正完成後即可依照順序正常執行

# 評估指標

在Loss計算.ipynb中，依序執行**載入必要套件**與**定義loss計算方法**區塊完成計算函數的定義，接著在三種模型各自的區塊中的**計算loss**區塊將生成圖片資料夾與對照答案資料夾的根路徑替換成你的儲存位置即可開始執行

```
#定義路徑相關變數

# 請替換成你的生成圖片來源路徑
set1 = '/content/drive/MyDrive/中興_深度學習/HW4/5-TFill/place2/result'
set2 = '/content/drive/MyDrive/中興_深度學習/HW4/5-TFill/celeba/result'

test_datasets = ["FFHQ_test", "Shunghaitech_test", "celeba_test", "paris_test", "place2_chruch_indoor_test", "place2_chruch_ourdoor_test"]

# 請替換成你的對照答案來源路徑
path_ans = '/content/drive/MyDrive/中興_深度學習/HW4/test'
```

# Complexity

在三個模型的訓練程式中的**Complexity**區塊，依照說明更改作者提供的程式，執行shell指令時會在開始推理前使用ptflops.get_model_complexity_infow透過模擬輸入計算模型的參數數量與FLOPs，TFill模型本身執行時也會顯示各類別的參數數量供參考
