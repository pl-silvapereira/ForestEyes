import os
from dotenv import load_dotenv

load_dotenv()
ROOT = os.getenv('PROJECT_ROOT')

def configurar_ambiente():
    pastas = [
        os.path.join(ROOT, 'data', 'CBERS4A-WPM', 'Downloads'),
        os.path.join(ROOT, 'data', 'MapBiomas'),
        os.path.join(ROOT, 'data', 'Output')
    ]
    
    for p in pastas:
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"📁 Pasta criada: {p}")
    
    print("✅ Ambiente ForestEyes pronto.")

if __name__ == "__main__":
    configurar_ambiente()