# PEDiT-inference-100M

Легковесный сервер и CLI-инференс для диффузионной модели **PEDiT-100M** (Phase EdgeFlow Diffusion Transformer).

Модели размещены на Hugging Face: [**AIT-FT/PEDiT-100M**](https://huggingface.co/AIT-FT/PEDiT-100M).

Все модели (FP16, FP8, INT8), текстовый энкодер mT5 и VAE (TAESD) **скачиваются автоматически при первом запуске**.

---

## 🚀 Быстрый старт

### Windows
1. Клонируйте репозиторий:
   ```cmd
   git clone https://github.com/AIT-FT/PEDiT-inference-100M.git
   cd PEDiT-inference-100M
   ```
2. Запустите Web UI:
   ```cmd
   run_server.bat
   ```
   Откройте в браузере: `http://localhost:8000`
3. Или запустите генерацию через консоль:
   ```cmd
   run_inference.bat
   ```

### Linux / macOS
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/AIT-FT/PEDiT-inference-100M.git
   cd PEDiT-inference-100M
   chmod +x *.sh
   ```
2. Запустите Web UI:
   ```bash
   ./run_server.sh
   ```
   Откройте в браузере: `http://localhost:8000`
3. Или запустите CLI:
   ```bash
   ./run_inference.sh
   ```

---

## 🎛️ Возможности Web панели

В веб-панели доступно интерактивное управление генерацией:
* **Выбор модели**:
  * `PEDiT-100M-FP16.pt` (Рекомендованная, высокое качество, ~206 МБ)
  * `PEDiT-100M-FP8.pt` (Быстрая и легкая, ~103 МБ)
  * `PEDiT-100M-INT8.pt` (Максимальная скорость, ~103 МБ)
* **CFG Scale** (рекомендуется 1.0 - 1.5)
* **Количество шагов генерации** (Euler sampler, рекомендуется 8)
* **Разрешение изображения** (рекомендуется 256x256 или 512x512)
* **Выбор устройства** (GPU CUDA или CPU)
* **Зерно (Seed)** (рекомендуется -1 для случайного)
* **Просмотр шагов** (показ каждого промежуточного шага или только финального)

---

## 📥 Загрузка моделей с Hugging Face

Вы можете скачать модели заранее через удобный скрипт:
* **Windows**: `download_models.bat`
* **Linux**: `./download_models.sh`

Или через CLI:
```bash
# Скачать нужную версию
python download_utils.py --model fp16
python download_utils.py --model fp8
python download_utils.py --model int8

# Скачать всё сразу
python download_utils.py --all
```

---

## 📁 Структура файлов проекта

* `server.py` — FastAPI + WebSocket сервер для веб-интерфейса
* `inference.py` — консольный скрипт генерации изображений
* `model.py` — архитектура нейросети EdgeFlowPhaseV2
* `encoders.py` — загрузчик текстового энкодера mT5 и VAE (TAESD)
* `rectified_flow.py` — сэмплер Phase Rectified Flow
* `check_deps.py` — автоматическая проверка и доустановка зависимостей
* `download_utils.py` — менеджер моделей и загрузок с Hugging Face
* `export_tensorrt.py` — экспорт и оптимизация под NVIDIA TensorRT
* `static/` — фронтенд веб-интерфейса (`index.html`, `script.js`, `style.css`)
* `run_server.bat` / `run_server.sh` — запуск веб-сервера (Windows / Linux)
* `run_inference.bat` / `run_inference.sh` — запуск консольного инференса (Windows / Linux)
* `download_models.bat` / `download_models.sh` — скачивание моделей (Windows / Linux)

---

## 📄 Лицензия
Apache License 2.0
