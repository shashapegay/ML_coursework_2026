# SECOM Python Analysis

Скрипт `secom_analysis.py` выполняет полный цикл работы:

- загрузка `secom.data`, `secom_labels.data`, `secom.names`;
- разведочный анализ данных и графики;
- обработка пропусков, нормализация, feature engineering и отбор признаков;
- перебор гипотез через `GridSearchCV`;
- hold-out валидация лучшей модели;
- суррогатное дерево для интерпретации;
- генерация русского отчета `secom_report.md`.

## Установка зависимостей

```bash
pip install -r requirements_secom.txt
```

## Запуск

```bash
python secom_analysis.py
```

Если данные лежат в другом месте:

```bash
python secom_analysis.py ^
  --data "C:\path\to\secom.data" ^
  --labels "C:\path\to\secom_labels.data" ^
  --names "C:\path\to\secom.names" ^
  --out-dir secom_python_outputs ^
  --n-jobs -1
```

После запуска результаты будут в папке `secom_python_outputs`.
