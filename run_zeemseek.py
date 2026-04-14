#!/usr/bin/env python3
"""
ZeemSeek Ultra - Запуск приложения
"""

import subprocess
import sys
import os
import time
import webbrowser

def check_dependencies():
    """Проверка установленных пакетов"""
    try:
        import torch
        import fastapi
        import uvicorn
        print("✓ Все зависимости установлены")
        return True
    except ImportError as e:
        print(f"✗ Отсутствует зависимость: {e}")
        print("Установка: pip install -r requirements.txt")
        return False

def main():
    print("🐍" * 50)
    print("ZeemSeek Ultra - Суперинтеллектуальная модель")
    print("🐍" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\nЗапуск сервера на http://localhost:8000")
    print("Интерфейс откроется автоматически\n")
    
    time.sleep(1)
    webbrowser.open("http://localhost:8000")
    
    # Запуск сервера
    from zeemseek_server import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
