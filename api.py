import asyncio
from fastapi import FastAPI, WebSocket, Request, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor

import asyncio
from transformers import AutoModel, AutoTokenizer
import mdtex2html
import uvicorn, json, datetime
import torch

import secrets
from threading import Thread
from time import sleep

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
TOKEN_EXPIRATION = 24 * 60 * 60  # 24 hours
CLEANUP_INTERVAL = 60 * 60  # 1 hour

# Store temporary access tokens
temp_tokens = {}

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def cleanup_tokens():
    while True:
        now = datetime.datetime.now()
        for token, timestamp in list(temp_tokens.items()):
            if (now - timestamp).total_seconds() > TOKEN_EXPIRATION:
                print(f"[{datetime.datetime.now()}] - " + 'Delete token:' + token + ', create at ' + temp_tokens[token])
                del temp_tokens[token]
        sleep(CLEANUP_INTERVAL)


def read_auth_keys(file_path: str):
    with open(file_path, "r") as file:
        keys = [line.strip() for line in file.readlines()]
    return keys


queue = asyncio.Queue(maxsize=100)

from fastapi.middleware.cors import CORSMiddleware

# 在FastAPI实例化之后，添加CORS中间件
app = FastAPI()

# 添加CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有站点访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    print(f"[{datetime.datetime.now()}] - scheme: {request.url.scheme}, url path: {request.url.path}")

    with open("static/api-demo.html", "r", encoding="utf-8") as f:
        content = f.read()
    return content


@app.get("/token")
async def get_token(appkey: str, request: Request):
    print(f"[{datetime.datetime.now()}] - scheme: {request.url.scheme}, url path: {request.url.path}, calling parameters: {appkey[:50]}")
    auth_keys = read_auth_keys("key.txt")
    if appkey in auth_keys:
        token = secrets.token_hex(16)
        temp_tokens[token] = datetime.datetime.now()
        print(f"[{datetime.datetime.now()}] - " + 'Token generated for ' + appkey + ':' + token)
        return {"token": token}
    else:
        raise HTTPException(status_code=403, detail="Invalid appkey")


@app.post("/")
async def create_item(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid Authorization header")
    
    now = datetime.datetime.now()

    token = authorization.split(" ")[1]
    if not token or token not in temp_tokens:
        raise HTTPException(status_code=403, detail="Invalid token")

    # Original code
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    print(f"[{now}] - scheme: {request.url.scheme}, url path: {request.url.path}, calling parameters: {str(json_post_list)[:50]}")


    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    # Measure model execution time
    start_time = datetime.datetime.now()
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    end_time = datetime.datetime.now()
    time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    exec_duration = (end_time - start_time).total_seconds()

    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = f"[{datetime.datetime.now()}]" + ' - prompt:"' + prompt + '", response:"' + repr(response) + '"' + ' - Model execution time: {exec_duration} seconds'
    print(log)
    torch_gc()
    return answer


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

async def predict(websocket, input, history, max_length, top_p, temperature):
    async def send_response(response, last_response):
        diff = response[len(last_response):]
        if diff:
            await websocket.send_text(diff)

    history.append((parse_text(input), ""))
    last_response = ""
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        history[-1] = (parse_text(input), parse_text(response))
        await send_response(parse_text(response), last_response)
        last_response = parse_text(response)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print(f"[{datetime.datetime.now()}] - scheme: ws")
    await websocket.accept()
    history = []

    while True:
        try:
            data = await websocket.receive_json()
            token = data.get("token")

            if not token or token not in temp_tokens:
                await websocket.close(code=4001)
                break

            input = data.get("prompt")
            max_length = data.get("max_length", 2048)
            top_p = data.get("top_p", 0.7)
            temperature = data.get("temperature", 0.95)
            
            start_time = datetime.datetime.now()
            await predict(websocket, input, history, max_length, top_p, temperature)
            end_time = datetime.datetime.now()
            exec_duration = (end_time - start_time).total_seconds()
            print(f"[{datetime.datetime.now()}] - ws request finished in {exec_duration} seconds.")
            await websocket.send_text("{{BBMCPLT}}")
        except Exception as e:
            print(f"[{datetime.datetime.now()}] - Error: {e}")
            break

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model.eval()

cleanup_thread = Thread(target=cleanup_tokens)
cleanup_thread.start()

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=40080, workers=1)