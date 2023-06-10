from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from recog import retinaface, facenet_predict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from aiofiles import open as a_open
from os import makedirs, getcwd
from os.path import join
from datetime import datetime, tzinfo, timezone, timedelta

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/hello_image_recog")
@limiter.limit('30/minute')
async def face_recognition(request: Request, file: UploadFile = File()):
    if file.size > 20_000_000:
        return PlainTextResponse('Too large image file', status_code=413)
    print(file.size, datetime.now(tz=timezone(timedelta(hours=9))), end=' ')
    if 'x-real-ip' in request.headers.keys():
        print(request.headers.get('x-real-ip'))
    else:
        print()
    image_file = BytesIO(await file.read())
    faces = retinaface(image_file)
    facenet_predict(res=faces, image=image_file)
    makedirs(join(getcwd(), 'uploaded', 'proceed'), exist_ok=True)
    async with a_open(
            file=join(getcwd(), 'uploaded', 'proceed',
                      f'{request.client.host}_{datetime.now().timestamp()}_{file.filename}'),
            mode='wb') as f:
        await f.write(image_file.getbuffer())

    return {'len': image_file.getbuffer().nbytes, "count": faces.__len__(), "faces": faces}


@app.post("/wrong_report")
@limiter.limit('10/minute')
async def wrong_report(request: Request, file: UploadFile = File()):
    print(file.size)
    if file.size > 10_000_000:
        return PlainTextResponse('Too large image file', status_code=413)
    makedirs(join(getcwd(), 'uploaded', 'wrong_images'), exist_ok=True)
    async with a_open(
            file=join(getcwd(), 'uploaded', 'wrong_images',
                      f'{request.client.host}_{datetime.now().timestamp()}_{file.filename}'),
            mode='wb') as f:
        await f.write(await file.read())


app.mount("/hp-face-recognizer", StaticFiles(directory=join(getcwd(), 'html'), html=True), name="html")
