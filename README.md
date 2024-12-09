
# Установка

Для начала работы с проектом, выполните следующие шаги:

1. **Клонируйте репозиторий `stable-audio-tools`:**

   ```bash
   git clone https://github.com/Stability-AI/stable-audio-tools.git
   cd stable-audio-tools
   ```

2. **Создайте и активируйте виртуальное окружение:**

   - **Conda:**

     ```bash
     conda create --name audio_proj python=3.8
     conda activate audio_proj
     ```

   - **Virtualenv:**

     ```bash
     python3 -m venv audio_proj
     source audio_proj/bin/activate  # Для Linux и macOS
     audio_proj\Scripts\activate  # Для Windows
     ```

3. **Установите зависимости из исходников:**

   Перейдите в директорию `stable-audio-tools` и выполните:

   ```bash
   pip install .
   ```

4. **Установите дополнительные зависимости для вашего проекта:**

   Перейдите в корневую директорию вашего проекта и выполните:

   ```bash
   pip install -r requirements.txt
   ```

Теперь ваше окружение готово для работы с проектом. Убедитесь, что все зависимости установлены корректно, и вы можете запускать скрипты из вашего проекта.
