def test_predict_json_success(client):
    payload = {
        "date": "2026-03-01",
        "prix": 100,
        "Category": "Kurta",
        "Style": "Casual"
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)

def test_predict_json_missing_field(client):
    payload = {
        "date": "2026-03-01",
        "prix": 100,
        "Category": "Kurta"
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422

def test_predict_json_invalid_type(client):
    payload = {
        "date": "2026-03-01",
        "prix": "abc",
        "Category": "Kurta",
        "Style": "Casual"
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422