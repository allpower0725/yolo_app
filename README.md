# YOLO Object Detection App

這是一個基於 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 的即時物件偵測應用程式。即使影片解析度高於螢幕，程式也能透過可調整大小的視窗正常顯示。

## ✨ 功能特色

- **即時偵測**：使用輕量級的 `yolo11n` 模型進行快速推論。
- **視窗控制**：支援調整視窗大小 (`cv2.WINDOW_NORMAL`)，避免高解析度影片超出螢幕。
- **循環播放**：影片播放完畢後會自動重頭開始，方便持續觀察。
- **WSL 支援**：特別針對 Windows Subsystem for Linux (WSL2/WSLg) 環境進行了相容性處理。
- **自訂偵測目標**：可透過文字檔指定要標示的物件，忽略無關資訊。
- **偵測日誌**：自動將偵測到的物件及其最高/最低準確率記錄至檔案中。

## 🛠️ 環境需求

- **Python**: >= 3.12
- **套件管理器**: [uv](https://github.com/astral-sh/uv) (推薦)
- **作業系統**: Linux (Ubuntu 24.04 測試通過), WSL2, 或 Windows/macOS

### Linux / WSL2 特別需求 (重要)

若您在 Linux 或 WSL2 環境下執行，且遇到 `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` 錯誤，請執行以下指令安裝必要的系統套件：

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libsm6 libice6 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 libxcb-shape0
```

## 🚀 安裝與執行

1. **安裝 Python 依賴**
   ```bash
   uv sync
   ```

2. **執行程式**
   ```bash
   uv run main.py
   ```

3. **操作方式**
   - **調整視窗**：使用滑鼠拖曳視窗邊緣即可調整大小。
   - **退出程式**：點擊視窗並按下鍵盤上的 `q` 鍵。

## ⚙️ 設定說明

主要設定位於 `main.py` 檔案中：

### 更改輸入來源 (WebCam / 影片)
預設為讀取專案目錄下的 `example.mp4`。若要使用電腦的 WebCam，請修改 `source` 參數：

```python
# main.py
results = model.predict(
    source="0",    # 改成 "0" (字串) 或 0 (整數) 代表預設攝影機
    # source="example.mp4", # 原始設定
    # ...
)
```

### 更改偵測模型
預設使用 `yolo11n.pt` (Nano 版本，速度最快)。您也可以改用其他模型，程式會在第一次執行時自動下載：
- `yolo11s.pt` (Small)
- `yolo11m.pt` (Medium)
- `yolo11l.pt` (Large)
- `yolo11x.pt` (Extra Large)

```python
# main.py
model = YOLO("yolo11s.pt") # 改用 Small 模型
```

### 指定偵測目標 (`target_classes.txt`)
您可以在 `target_classes.txt` 檔案中列出希望程式關注的物件名稱（一行一個）。程式執行時，只有清單內的物件才會被標框顯示並記錄。
如果該檔案不存在或內容為空，程式將會顯示所有偵測到的物件。

**範例內容：**
```text
person
car
dog
```

### 檢視偵測紀錄 (`detection_log.txt`)
程式執行時會在目錄下自動生成 `detection_log.txt`，即時更新每個物件的統計數據：

**範例內容：**
```text
Object          Max Conf   Min Conf  
-----------------------------------
person          0.95       0.72      
car             0.88       0.65      
```

## ❓ 常見問題排除

**Q: 按下 'q' 後程式沒有反應？**
A: 請確保您的焦點是在「影片視窗」上（點一下影片畫面），而不是在終端機視窗，然後再按 `q`。

**Q: 視窗一片全黑或沒有畫面？**
A: 如果是使用 WebCam，請檢查是否有其他程式正在佔用相機。如果是 WSL2 環境，請確認已正確設定 USB 裝置透傳 (透過 `usbipd-win`)。
