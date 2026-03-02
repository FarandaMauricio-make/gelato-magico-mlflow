import subprocess

def run():
    print("1) Rodando treinamento com MLflow...")
    subprocess.run(["python", "src/train.py"], check=True)

    print("\n2) Testando previsão rápida...")
    # exemplo com 30°C
    p = subprocess.run(
        ["python", "src/predict.py"],
        input="30\n",
        text=True,
        capture_output=True
    )
    print(p.stdout)

if __name__ == "__main__":
    run()
