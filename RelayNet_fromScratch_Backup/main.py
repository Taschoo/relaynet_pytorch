import os
import scripts.train as train
import scripts.test as test

# Bereinigen der Kommandozeile
if os.name == 'nt':  # Windows
    os.system('cls')
else:  # Unix-basierte Systeme (Linux, macOS)
    os.system('clear')

if __name__ == '__main__':
    train.main()
    test.main()
