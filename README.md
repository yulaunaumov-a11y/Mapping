# Модель маппинга
Репозиторий представляет из себя короткое исследование на тему получения матрицы гомографии для мапиинга точек из одной картинки ('top', 'bottom') к 'door2'. 

## Запуск обучения
Для запуска обучения модели через консоль:
```python solution/train.py CAMERA_TYPE --USE_ML ```

где:
* CAMERA_TYPE: тип камеры ('bottom', 'top')
* USE_ML: использовать ML или нет.

#### Пример:

```python solution/train.py top --use_ml```

```python solution/train.py bottom```

## Поулчение предикта
В консоли:

```python solution/tpredict.py --model --x --y --source ```

где:
* model: путь до json файл с матрицекй гомографии или путь до pkl файла, для использования ML модели
* X: координата X
* Y: координата Y
* source: тип камеры ('bottom', 'top')

#### Пример:

```python predict.py --model ./solution/Homography.json --x 100 --y 200 --source top```

```python predict.py --model ./solution/ml_model.pkl --x 580.76 --y 416.21 --source bottom```


## Установки зависимостей
Пройти по пути './test-task', запустить:

``` make install_all_requirements ```

Если какой-то зависимости не хватает, дописать в в файл 'requirements.txt'

## Содержание репозитория
* 0_EDA: EDA иследование датасета;
* 1_Research Homography: поиск наилучшего метода для маппинга; 
* 2_Test_Method: тестирование .py скриптов;
* result: в папке храняться изображения по результату обучения модели/матрицы;
* solution/train.py: скрипт для оубчения модели;
* solution/predict.py: скрипт для поулчения предикта модели/матрицы.