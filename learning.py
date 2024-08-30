# Импорт необходимых библиотек
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import gdown
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.regularizers import l2

# Проверка, существует ли папка и есть ли в ней файлы
DATA_PATH = 'car_brands1'

if os.path.exists(DATA_PATH) and len(os.listdir(DATA_PATH)) > 0:
    print("Папка существует и содержит файлы, загрузка не требуется.")
else:
    print("Папка пуста или не существует, загружаем и распаковываем данные.")

    # Загрузка zip-архива с датасетом из облака на диск виртуальной машины colab
    gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l5/middle_fmr.zip', None, quiet=True)

    # Распаковка архива в нужную директорию
    with zipfile.ZipFile("middle_fmr.zip", 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH)

    print("Данные успешно загружены и распакованы.")

# Задание гиперпараметров
TRAIN_PATH = './car_brands1'       # Папка для обучающего набора данных
IMG_WIDTH = 128                    # Ширина изображения для нейросети
IMG_HEIGHT = 64                    # Высота изображения для нейросети
IMG_CHANNELS = 3                   # Количество каналов (для RGB равно 3)
CLASS_LIST = sorted(os.listdir(TRAIN_PATH))  # Список классов
CLASS_COUNT = len(CLASS_LIST)      # Количество классов
EPOCHS = 100                        # Число эпох обучения
BATCH_SIZE = 24                    # Размер батча для обучения модели
OPTIMIZER = Adam(0.0001)           # Оптимизатор

# Параметры аугментации
ROTATION_RANGE = 8                 # Пределы поворота
WIDTH_SHIFT_RANGE = 0.15           # Пределы сдвига по горизонтали
HEIGHT_SHIFT_RANGE = 0.15          # Пределы сдвига по вертикали
ZOOM_RANGE = 0.15                  # Пределы увеличения/уменьшения
BRIGHTNESS_RANGE = (0.7, 1.3)      # Пределы изменения яркости
HORIZONTAL_FLIP = True             # Горизонтальное отражение разрешено
VAL_SPLIT = 0.2                    # Доля проверочной выборки в обучающем наборе

# Генераторы изображений
train_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    rotation_range=ROTATION_RANGE,
    width_shift_range=WIDTH_SHIFT_RANGE,
    height_shift_range=HEIGHT_SHIFT_RANGE,
    zoom_range=ZOOM_RANGE,
    brightness_range=BRIGHTNESS_RANGE,
    horizontal_flip=HORIZONTAL_FLIP,
    validation_split=VAL_SPLIT
)

# Изображения для тестового набора только нормализуются
test_datagen = ImageDataGenerator(rescale=1. / 255.)

# Обучающая выборка генерируется из папки обучающего набора
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

# Проверочная выборка также генерируется из папки обучающего набора
validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

# Функция для оценки модели и вывода матрицы ошибок
def eval_model(model,
               x,                # данные для предсказания модели (вход)
               y_true,           # верные метки классов в формате OHE (выход)
               class_labels=[],  # список меток классов
               cm_round=3,       # число знаков после запятой для матрицы ошибок
               title='',         # название модели
               figsize=(10, 10)  # размер полотна для матрицы ошибок
               ):
    # Вычисление предсказания сети
    y_pred = model.predict(x)
    # Построение матрицы ошибок
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')
    # Округление значений матрицы ошибок
    cm = np.around(cm, cm_round)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Нейросеть {title}: матрица ошибок нормализованная', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    ax.images[-1].colorbar.remove()       # Стирание ненужной цветовой шкалы
    fig.autofmt_xdate(rotation=45)        # Наклон меток горизонтальной оси
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    plt.show()

    print('-'*100)
    print(f'Нейросеть: {title}')

    # Для каждого класса:
    for cls in range(len(class_labels)):
        # Определяется индекс класса с максимальным значением предсказания (уверенности)
        cls_pred = np.argmax(cm[cls])
        # Формируется сообщение о верности или неверности предсказания
        msg = 'ВЕРНО :-)' if cls_pred == cls else 'НЕВЕРНО :-('
        # Выводится текстовая информация о предсказанном классе и значении уверенности
        print('Класс: {:<20} {:3.0f}% сеть отнесла к классу {:<20} - {}'.format(class_labels[cls],
                                                                               100. * cm[cls, cls_pred],
                                                                               class_labels[cls_pred],
                                                                               msg))

    # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
    print('\nСредняя точность распознавания: {:3.0f}%'.format(100. * cm.diagonal().mean()))

# Функция компиляции и обучения модели нейронной сети
# По окончанию выводит графики обучения и сохраняет модель
def compile_train_model(model,                  # модель нейронной сети
                        train_data,             # обучающие данные
                        val_data,               # проверочные данные
                        optimizer=OPTIMIZER,    # оптимизатор
                        epochs=EPOCHS,          # количество эпох обучения
                        batch_size=BATCH_SIZE,  # размер батча
                        figsize=(20, 5)):       # размер полотна для графиков

    # Компиляция модели
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Вывод сводки
    model.summary()

    # Обучение модели с заданными параметрами
    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=val_data)

    # Сохранение весов модели после завершения обучения
    model.save_weights('./models/car_brand_model.weights.h5')
    print("Веса модели сохранены в файл 'car_brand_model.weights.h5'")

    # Также можно сохранить всю модель для дальнейшего использования
    model.save('./models/car_brand_model.h5')
    print("Модель сохранена в файл 'car_brand_model.h5'")

    # Вывод графиков точности и ошибки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('График процесса обучения модели')
    ax1.plot(history.history['accuracy'],
               label='Доля верных ответов на обучающем наборе')
    ax1.plot(history.history['val_accuracy'],
               label='Доля верных ответов на проверочном наборе')
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('Эпоха обучения')
    ax1.set_ylabel('Доля верных ответов')
    ax1.legend()

    ax2.plot(history.history['loss'],
               label='Ошибка на обучающем наборе')
    ax2.plot(history.history['val_loss'],
               label='Ошибка на проверочном наборе')
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlabel('Эпоха обучения')
    ax2.set_ylabel('Ошибка')
    ax2.legend()
    # Сохранение графика в файл
    plt.savefig(f'./graphs/training_process.png')
    plt.close()  # Закрываем график, чтобы избежать отображения

# Совместная функция обучения и оценки модели нейронной сети
def compile_train_eval_model(model,                    # модель нейронной сети
                             train_data,               # обучающие данные
                             val_data,                 # проверочные данные
                             test_data,                # тестовые данные
                             class_labels=CLASS_LIST,  # список меток классов
                             title='',                 # название модели
                             optimizer=OPTIMIZER,      # оптимизатор
                             epochs=EPOCHS,            # количество эпох обучения
                             batch_size=BATCH_SIZE,    # размер батча
                             graph_size=(20, 5),       # размер полотна для графиков обучения
                             cm_size=(10, 10)          # размер полотна для матрицы ошибок
                             ):

    # Компиляция и обучение модели на заданных параметрах
    compile_train_model(model,
                        train_data,
                        val_data,
                        optimizer=optimizer,
                        epochs=epochs,
                        batch_size=batch_size,
                        figsize=graph_size)

    # Оценка модели на тестовой выборке
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f'Точность на тестовой выборке: {test_accuracy:.2f}')

    # Вывод результатов оценки работы модели на тестовых данных
    eval_model(model, test_data[0][0], test_data[0][1],
               class_labels=class_labels,
               title=title,
               figsize=cm_size)

# Создание модели сверточной нейронной сети
model = Sequential()

# Первый сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
model.add(BatchNormalization())

# Второй сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

# Третий сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Четвертый сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))

# Пятый сверточный слой
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

# Шестой сверточный слой
model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))

# Слой преобразования многомерных данных в одномерные
model.add(Flatten())

# Промежуточный полносвязный слой
model.add(Dense(2048, activation='relu'))

# Промежуточный полносвязный слой
model.add(Dense(4096, activation='relu'))

# Выходной полносвязный слой с количеством нейронов по количеству классов
model.add(Dense(CLASS_COUNT, activation='softmax'))

# Вызов функции для компиляции, обучения и оценки модели
compile_train_eval_model(model,
                         train_generator,
                         validation_generator,
                         validation_generator,  
                         class_labels=CLASS_LIST,
                         title='Car Brand Classifier')
