from fastapi import FastAPI
import ml_tools
from pydantic import BaseModel


app = FastAPI()


# пример модели: описывает, что клиент должен отправить.
class UserRequest(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float


# описывает, что сервер вернёт
class UserResponse(BaseModel):
    Y_pred: float


class ModelInfoResponse(BaseModel):
    coef: list[float] 
    R2: float


# get-запрос для корневого эндпоинта, используется в качестве health-check
@app.get("/")
async def root():
    """health-check; в норме выдаёт Ok"""
    # docstring будет виден в /doc
    return {"status": "Ok"}


@app.get("/ping")
async def number():
    return {"status": "Ok"}


@app.post("/prediction", response_model=UserResponse,  summary="Предсказание целевой переменной")
async def prediction(features: UserRequest) -> UserResponse:
    Y_pred = ml_tools.predict_Y(features.x1, features.x2, features.x3, features.x4, features.x5)
    return UserResponse(Y_pred=Y_pred)


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(coef=ml_tools.model_info['coefficients'], R2=ml_tools.R2)


# # специальные параметры (summary и description) декоратора станут частью документации
# @app.get("/number_with_params", summary="тут короткое описание эндпоинта", description="а тут детальное")
# async def number(min:int, max:int):
#     """Выдаёт случайное число от min до max включительно"""
#     # todo: проверить: min < max
#     return {"number":  randint(min,max)}
