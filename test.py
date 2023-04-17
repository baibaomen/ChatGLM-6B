import asyncio
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI! - After 10 seconds"}

@app.get("/calculate")
async def calculate(a: int, b: int, operation: str):
    await asyncio.sleep(10)  # 模拟耗时操作，暂停 10 秒
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"error": "Division by zero is not allowed."}
        result = a / b
    else:
        return {"error": "Invalid operation."}

    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30081)