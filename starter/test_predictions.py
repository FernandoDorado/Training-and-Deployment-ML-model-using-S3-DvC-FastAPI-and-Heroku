from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_hello_world():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World!"}


def test_predict():
    request_body = {
        "age": 49,
        "workclass": "Private",
        "fnlgt": 187454,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 99999,
        "capital-loss": 0,
        "hours-per-week": 65,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    assert response.json() == {"Model Prediction (Salary): ": ">50K"}

test_predict