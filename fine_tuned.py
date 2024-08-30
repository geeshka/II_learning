import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import shutil
import zipfile
from datetime import datetime

# Параметры
TEST_SPLIT = 0.15  # 15% данных выделяем для теста

# Пути к данным
TRAIN_PATH = 'car_brands1'
TEST_PATH = 'car_brands_test'
IMG_HEIGHT = 64  # Высота изображения
IMG_WIDTH = 128  # Ширина изображения
BATCH_SIZE = 16  # Размер батча
INITIAL_LEARNING_RATE = 1e-5  # Начальная скорость обучения

# Создание папки для тестовых данных и разделение данных
CLASS_LIST = os.listdir(TRAIN_PATH)
try:
    os.mkdir(TEST_PATH)
except:
    pass

train_count = 0
test_count = 0

for class_name in CLASS_LIST:
    class_path = f'{TRAIN_PATH}/{class_name}'
    test_path = f'{TEST_PATH}/{class_name}'
    class_files = os.listdir(class_path)
    class_file_count = len(class_files)

    try:
        os.mkdir(test_path)
    except:
        pass

    test_file_count = int(class_file_count * TEST_SPLIT)
    test_files = class_files[-test_file_count:]
    for f in test_files:
        shutil.move(f'{class_path}/{f}', f'{test_path}/{f}')  # Используем shutil.move вместо os.rename
    train_count += class_file_count - test_file_count
    test_count += test_file_count

    print(f'Размер класса {class_name}: {class_file_count} машин, для теста выделено файлов: {test_file_count}')

print(f'Общий размер базы: {train_count + test_count}, выделено для обучения: {train_count}, для теста: {test_count}')

# Генераторы изображений с аугментацией
train_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.5, 1.5),
    horizontal_flip=True,
    validation_split=0.2  # 20% для валидации
)

test_datagen = ImageDataGenerator(rescale=1. / 255.)

# Генераторы данных
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Загрузка сохраненной модели
model = load_model('./models/car_brand_model.h5')
# model = load_model('./models/car_brand_model_Car_Brand_Classifier.h5')

# Компиляция модели с новым оптимизатором
OPTIMIZER = Adam(learning_rate=INITIAL_LEARNING_RATE)
model.compile(optimizer=OPTIMIZER,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Дообучение модели
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    # callbacks=[early_stopping, reduce_lr]
)

# Оценка модели после дообучения на тестовом наборе данных
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Точность на тестовой выборке после дообучения: {test_accuracy:.2f}')

# Оценка модели на валидационной выборке
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Точность на валидационной выборке после дообучения: {val_accuracy:.2f}')

# Построение графиков точности и потерь и их сохранение
def plot_history(history, save_path='./graphs/training_history.png'):
    plt.figure(figsize=(12, 5))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучающей выборке')
    plt.plot(history.history['val_accuracy'], label='Точность на валидационной выборке')
    plt.title('Точность модели')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучающей выборке')
    plt.plot(history.history['val_loss'], label='Потери на валидационной выборке')
    plt.title('Потери модели')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)  # Сохранение графиков в файл
    plt.close()  # Закрытие графика, чтобы не пытаться отобразить его

# Вызываем функцию для построения и сохранения графиков
plot_history(history)

# Условие для сохранения модели, скрипта и их упаковки
if test_accuracy >= 0.85:
    # Генерация имени файла с текущей датой и временем
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    model_filename = f'car_brand_model_finetuned_{timestamp}.h5'
    weights_filename = f'car_brand_model_weights_{timestamp}.weights.h5'
    
    # Сохранение модели и весов
    model.save(f'./models/{model_filename}')
    model.save_weights(f'./models/{weights_filename}')
    
    # Создание скрипта для использования модели
    with open(f'./models/car_brand_model_script_{timestamp}.py', 'w') as f:
        f.write(f"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Пути к данным
model_path = ''

# Загрузка изображения для предсказания (укажите свой путь к изображению)
new_image_path = ''

# Параметры изображения
IMG_HEIGHT = 64  
IMG_WIDTH = 128  

# Загрузка изображения и приведение его к нужному формату
img = image.load_img(new_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  
img_array /= 255.0  

# Загрузка обученной модели
model = load_model(model_path)

# Предсказание
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print(f"Модель предсказала класс: {predicted_class}")
""")

    # Упаковка в ZIP архив
    with zipfile.ZipFile(f'car_brand_model_package_{timestamp}.zip', 'w') as zipf:
        zipf.write(f'./models/{model_filename}')
        zipf.write(f'./models/{weights_filename}')
        zipf.write(f'./models/car_brand_model_script.py')

    print(f"Модель, веса и скрипт упакованы в архив car_brand_model_package_{timestamp}.zip.")
else:
    print("Точность модели ниже 85%, модель не сохранена.")