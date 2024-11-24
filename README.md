# qrobot
Программы для робота на PyQt

## Подготовка к работе

### Создание и активация окружения
```
python -m venv .venv
source .venv/bin/activate
```

### Установка пакетов на компьютере

```
pip install -r requirements.txt --upgrade
```

### Установка пакетов на Raspberry PI 4B

```
pip install -r requirements_rpi4b.txt --upgrade
```

## Обучение моделей

### Редактирование пользовательского интерфейса

```
pyqt6-tools designer
```

### Обновление пользовательского интерфейса

```
cd train
pyuic6 train_window.ui -o train_window_ui.py 
```

### Запуск

```
cd train
./train_app.py 
```