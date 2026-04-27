# Keyboard Sound Classifier

Учебный Python-проект для классификации коротких звуков нажатий клавиш по собственным аудиозаписям. Проект показывает полный MVP-пайплайн: запись датасета, предобработка, извлечение признаков, обучение baseline-моделей и предсказание через CLI или Streamlit.

## Ограничения и этика

Этот проект предназначен только для:

- анализа своих собственных записей;
- демонстрации audio classification;
- экспериментов с безопасным локальным ML-пайплайном.

Проект **не** предназначен для скрытой записи, фонового захвата аудио, перехвата паролей, слежки или любых других неэтичных сценариев.

## Структура проекта

```text
keyboard_sound_classifier/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── record.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── app.py
├── requirements.txt
└── README.md
```

## Установка

Требуется Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Базовые клавиши

MVP настроен на 9 классов:

- `A`
- `S`
- `D`
- `F`
- `J`
- `K`
- `L`
- `Space`
- `Enter`

## 1. Запись датасета

Запись идёт только в явном foreground-режиме. Перед сессией скрипт предложит выбрать микрофон, затем нужно один раз нажать Enter и после этого нажать выбранную клавишу много раз подряд. Скрипт сам определит отдельные клики и сохранит их как отдельные `.wav`-файлы.

Пример:

```bash
python -m src.record --label A --samples 20 --duration 0.4
python -m src.record --label Space --samples 20 --duration 0.4
python -m src.record --label Enter --samples 50 --duration 0.4 --max-session-seconds 40
```

Если `--label` не указан, скрипт предложит выбрать метку из списка. Если `--device` не указан, скрипт покажет список доступных микрофонов для выбора.

Файлы сохраняются в формате:

```text
data/raw/<label>/<label>_<index>.wav
```

Полезные параметры:

- `--samples 50` для записи 50 кликов за одну сессию;
- `--device 3` чтобы сразу выбрать конкретный микрофон;
- `--max-session-seconds 40` чтобы дать себе больше времени на серию кликов.

## 2. Предобработка

Скрипт выполняет:

- загрузку `.wav`;
- приведение к `22050 Hz`;
- обрезку тишины;
- нормализацию громкости;
- сохранение результата в `data/processed/`.

Запуск:

```bash
python -m src.preprocess
```

## 3. Извлечение признаков

В `src/features.py` реализован baseline-признаковый пайплайн:

- MFCC;
- delta MFCC;
- delta-delta MFCC;
- RMS energy;
- spectral centroid;
- zero crossing rate.

Для фиксированной длины используются статистики `mean + std` по временным кадрам.

## 4. Обучение

Скрипт `src.train`:

- обходит `data/processed/`;
- извлекает признаки;
- делит выборку на train/test;
- обучает `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`;
- сравнивает `accuracy` и `macro F1`;
- сохраняет лучшую модель и `LabelEncoder`.

Запуск:

```bash
python -m src.train
```

После обучения артефакты появятся в папке `models/`:

- `best_model.joblib`
- `label_encoder.joblib`
- `evaluation_data.joblib`
- `model_metrics.csv`

## 5. Оценка

Скрипт выводит:

- accuracy;
- macro F1;
- classification report;
- confusion matrix.

Запуск:

```bash
python -m src.evaluate
```

PNG-файл confusion matrix сохраняется в `models/confusion_matrix.png`.

## 6. Предсказание по одному файлу

Пример:

```bash
python -m src.predict path/to/file.wav
```

Если модель поддерживает `predict_proba`, скрипт дополнительно покажет confidence и вероятности по классам.

## CNN на Mel-Spectrogram

В `src.train` теперь есть более сильный маршрут обучения:

- классические признаки MFCC + `RandomForest` / `SVC` / `KNeighbors`;
- mel-spectrogram;
- компактная CNN на `tensorflow.keras`.

Во время обучения `train.py` сравнивает все доступные модели по `accuracy` и `macro F1`, после чего сохраняет действительно лучшую модель. Если CNN показывает лучший результат, она будет использоваться и в `predict.py`, и в `evaluate.py`, и в `app.py`.

## 6a. Безопасный тестовый режим

Для проверки модели на одной заранее выбранной клавише есть отдельная команда. Она:

- предлагает выбрать микрофон;
- запускается только после явного `Enter`;
- пишет foreground-сессию;
- останавливается после нажатия `End` в активной консоли;
- если `End` перехватывается терминалом, позволяет остановить запись через `/stop` и `Enter`;
- режет запись на отдельные клики;
- показывает предсказания по каждому клику и итоговую строку `Predicted sequence`.

Пример:

```bash
python -m src.test_single_label --label A --max-session-seconds 20 --save-session
python -m src.test_single_label --max-session-seconds 20
```

Параметр `--label` теперь необязателен. Если он указан, скрипт дополнительно посчитает accuracy для этой одной ожидаемой клавиши.

По умолчанию тест использует лучшую сохранённую модель. При желании можно выбрать конкретную:

```bash
python -m src.test_single_label --model-name RandomForest
python -m src.test_single_label --model-name SVC
python -m src.test_single_label --model-name MelSpectrogramCNN
```

Чтобы выбор моделей работал, нужно сначала заново выполнить `python -m src.train`, потому что теперь обучение сохраняет все кандидаты отдельно.

## 7. Streamlit-приложение

Запуск:

[Сайт](https://keyboard-sound-classifier-5ciwmtbk4bcgk9nzlgvhnd.streamlit.app/)

```bash
streamlit run app.py
```

В приложении можно:

- загрузить `.wav` файл или записать напрямую через микрофон;
- получить предсказанный класс клавиши;
- увидеть confidence или вероятности классов, если модель их поддерживает.

## Пример пайплайна целиком

```bash
python -m src.record --label A --samples 20 --duration 0.4
python -m src.record --label S --samples 20 --duration 0.4
python -m src.record --label D --samples 20 --duration 0.4
python -m src.record --label F --samples 20 --duration 0.4
python -m src.record --label J --samples 20 --duration 0.4
python -m src.record --label K --samples 20 --duration 0.4
python -m src.record --label L --samples 20 --duration 0.4
python -m src.record --label Space --samples 20 --duration 0.4
python -m src.record --label Enter --samples 20 --duration 0.4
python -m src.preprocess
python -m src.train
python -m src.evaluate
streamlit run app.py
```

## Замечания по качеству

- Для baseline лучше записывать все клавиши на одном и том же устройстве и расстоянии до микрофона.
- Желательно собирать минимум 20–50 примеров на класс.
- Простые признаки MFCC + RandomForest/SVM хорошо подходят для MVP, но не гарантируют высокую точность на новых клавиатурах и в шумной среде.
- Если в папках нет файлов или загружен неподдерживаемый формат, скрипты завершатся с понятной ошибкой.
