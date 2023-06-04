from io import BytesIO
from retinaface.pre_trained_models import get_model
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import PlainTextResponse
from recog import retinaface, facenet_predict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from aiofiles import open as a_open
from os import makedirs, getcwd
from os.path import join
from datetime import datetime

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/hello_image_recog")
@limiter.limit('30/minute')
async def face_recognition(request: Request, file: UploadFile = File()):
    if file.size > 20_000_000:
        return PlainTextResponse('Too large image file', status_code=413)
    print("here!")
    image_file = BytesIO(await file.read())
    faces = retinaface(image_file)
    facenet_predict(res=faces, image=image_file)

    return {'len': image_file.getbuffer().nbytes, "count": faces.__len__(), "faces": faces}


@app.post("/wrong_report")
@limiter.limit('10/minute')
async def wrong_report(request: Request, file: UploadFile = File()):
    print(file.size)
    if file.size > 10_000_000:
        return PlainTextResponse('Too large image file', status_code=413)
    makedirs(join(getcwd(), 'wrong_images'), exist_ok=True)
    async with a_open(
            file=join(getcwd(), 'wrong_images', f'{request.client.host}_{datetime.now().timestamp()}_{file.filename}'),
            mode='wb') as f:
        await f.write(await file.read())
