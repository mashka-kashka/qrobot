import csv


def save_csv(filename, data):
    # 'utf-8-sig' автоматически добавляет BOM, чтобы Excel сразу открывал на русском
    with open(filename, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print(f"Файл успешно создан: {filename}")


# 1. Генерация чисел текстом (0 - 59)
units = [
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
]
teens = [
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
]
tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят"]

numbers_data = []
for i in range(60):
    if i < 10:
        text = units[i]
    elif i < 20:
        text = teens[i - 10]
    else:
        t = i // 10
        u = i % 10
        text = tens[t] + (" " + units[u] if u != 0 else "")
    numbers_data.append([i, text])


# Функция для определения правильного склонения
def get_declension(n, form1, form2, form5):
    if n % 100 in [11, 12, 13, 14]:
        return form5
    if n % 10 == 1:
        return form1
    if n % 10 in [2, 3, 4]:
        return form2
    return form5


# 2. Генерация часов (0 - 23)
hours_data = [[i, get_declension(i, "час", "часа", "часов")] for i in range(24)]

# 3. 0 - 59
minutes_data = [
    [i, get_declension(i, "минута", "минуты", "минут")] for i in range(60)
]

# 4. 0 - 59
seconds_data = [
    [i, get_declension(i, "секунда", "секунды", "секунд")] for i in range(60)
]

# Запуск создания файлов
save_csv("numbers_0_59.csv", numbers_data)
save_csv("hours_declension.csv", hours_data)
save_csv("minutes_declension.csv", minutes_data)
save_csv("seconds_declension.csv", seconds_data)
