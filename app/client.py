import requests

BASE_URL = "http://127.0.0.1:8880"

if __name__ == "__main__":


    print("Ping")
    response = requests.get(f"{BASE_URL}/ping")
    print(f"Ответ: {response.json()}")


    response = requests.get(f"{BASE_URL}/model_info")
    model_info = response.json()
    print(f"Коэффициенты: {model_info['coef']}")
    print(f"R2 score: {model_info['R2']}")


    features = {
        "x1": 0.4,
        "x2": 0.55,
        "x3": -0.2,
        "x4": -0.1,
        "x5": 0.88
    }
    
    response = requests.post(f"{BASE_URL}/prediction", json=features)
    prediction = response.json()
    print(f"Предсказание: {prediction['Y_pred']}")