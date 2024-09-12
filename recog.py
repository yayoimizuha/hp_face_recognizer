from functools import cache
from io import BytesIO
from pprint import pprint
from numpy import pi, sqrt
from PIL import Image
from numpy import array, arctan2
from retinaface.pre_trained_models import get_model
from torch.cuda import is_available
from torchvision.transforms.functional import to_tensor
from torch import stack, load, no_grad, topk, device, mean
from torch.nn import Module, Sequential, Softmax
from PIL.ImageOps import exif_transpose
from itertools import combinations_with_replacement

dev = device(device='cuda') if is_available() else device(device='cpu')
print(f'device: {dev}')

retinaface_model = get_model("resnet50_2020-07-20", max_size=512, device=dev)
retinaface_model.eval()


# register_heif_opener()

@cache
def class_text(x: int):
    return ['段原瑠々', '中山夏月姫', '横山玲奈', '八木栞', '金澤朋子', '後藤花', '小関舞', '石山咲良',
            '弓桁朱琴', '米村姫良々', '広本瑠璃', '道重さゆみ', '前田こころ', '橋迫鈴', '秋山眞緒', 
            '勝田里奈', '佐々木莉佳子', '斉藤円香', '田口夏実', '中島早貴', '川村文乃', '浅倉樹々', 
            '上國料萌衣', '岡村美波', '太田遥香', '清水佐紀', '豫風瑠乃', '宮本佳林', '植村葉純',
            '川名凜', '一岡伶奈', '田代すみれ', '和田桜子', '室田瑞希', '川嶋美 楓', '真野恵里菜',
            '田村芽実', '生田衣梨奈', '広瀬彩海', '平井美葉', '西﨑美空', '石栗奏美', '加賀楓',
            '植村あかり', '鈴木愛理', '江端妃咲', '平山遊季', '藤井梨央', '岸本ゆめの', '小川麗奈',
            '吉田姫杷', '井上春華', '萩原舞', '岡村ほまれ', '谷本安美', '為永幸音', '相川茉穂',
            '相馬優芽', '小野瑞歩', '尾形春水', '鈴木香音', '山岸理子', '山﨑夢羽', '村越彩菜',
            '石田亜佑美', '笠原桃奈', '小野田紗栞', '鞘師里保', '野村みな美', '松永里愛','山﨑愛生',
            '高木紗友希', '夏焼雅', '伊勢鈴蘭', '松本わかな', '橋田歩果', '船木結', '有澤一華',
            '入江里咲', '須藤茉麻', '井上玲音', '高瀬くるみ', '熊井友理奈', '山木梨沙', '稲場愛香',
            '菅谷梨沙子', '工藤遥', '北川莉央', '野中美希', '河西結心', '筒井澪心', '徳永千奈美',
            '田中れいな', '北原もも', '島川波菜', '遠藤彩加里', '和田彩花', '佐藤優樹', '飯窪春菜', 
            '羽賀朱音', '櫻井梨央', '里吉うたの', '西田汐里', '浜浦彩乃', '矢島舞美', '宮崎由加',
            '中西香菜', '森戸知沙希', '福田真琳', '岡井千聖', '江口紗耶', '清野桃々姫', '梁川奈々美', 
            '小片リサ', '小林萌花', '松原ユリヤ', '牧野真莉愛', '上村麗菜', '嗣永桃子', '譜久村聖', 
            '下井谷幸穂', '竹内朱莉', '窪田七海', '小田さくら', '新沼希空', '島倉りか', '工藤由愛', '小野田華凜'][x]


@no_grad()
def retinaface(image_data: BytesIO):
    image_data.seek(0)
    try:
        image = Image.open(image_data)
        image = exif_transpose(image)

        if image.mode != 'RGB':
            print(image.mode)
            image = image.convert(mode='RGB')

        image_arr = array(image)
    except Exception as e:
        print("invalid Image")
        return []
    res = retinaface_model.predict_jsons(image=image_arr)
    if res.__len__() == 1 and res[0]['score'] == -1:
        return []
    return res


def truncate(landmark: list[tuple[float]]) -> tuple[tuple[int, int], float]:
    left_eye, right_eye, nose, left_mouth, right_mouth = landmark
    center_x = sum((left_eye[0], right_eye[0], left_mouth[0], right_mouth[0])) / 4
    center_y = sum((left_eye[1], right_eye[1], left_mouth[1], right_mouth[1])) / 4
    eye_center = (right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2
    mouth_center = (right_mouth[0] + left_mouth[0]) / 2, (right_mouth[1] + left_mouth[1]) / 2
    return (int(center_x), int(center_y)), arctan2(eye_center[0] - mouth_center[0], mouth_center[1] - eye_center[1])


facenet_model: Module = Sequential(load(f='model.pth'), Softmax(dim=1)).to(dev)
facenet_model.eval()


@no_grad()
def facenet_predict(res: list[dict], image: BytesIO):
    image = Image.open(image)
    image = exif_transpose(image)

    if image.mode != 'RGB':
        print(image.mode)
        image = image.convert(mode='RGB')

    for order, face in enumerate(res):
        bbox, score, landmarks = face.values()
        face['pred'] = dict()
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        trans = truncate(landmarks)
        face['rotate'] = trans[1]
        image_size = max(face_width, face_height) * sqrt(2) // 2

        if image_size < 70:
            face['pred']['stat'] = {'invalid': f'face too small. {(face_height, face_width)}'}
            continue
        cropped_list = []
        for gap_r, gap_w, gap_h in combinations_with_replacement([-1, 0, 1], r=3):
            gap_w, gap_h = map(lambda x: x * int(image_size / 20), (gap_w, gap_h))
            cropped_face = image.rotate(angle=(trans[1] * 360 / (2 * pi)) + gap_r * 5, center=trans[0]).crop(
                (trans[0][0] - image_size + gap_w, trans[0][1] - image_size + gap_h, trans[0][0] + image_size + gap_w,
                 trans[0][1] + image_size + gap_h)).resize(size=(224, 224))
            cropped_list.append(to_tensor(cropped_face).to(dev))

        pred_tensor = mean(facenet_model(stack(cropped_list)), dim=0)
        predict = topk(pred_tensor.to(device(device='cpu')), k=5)
        for rank, (value, index) in enumerate(zip(predict.values.tolist(), predict.indices.tolist())):
            face['pred']['stat'] = {'success': ''}
            face['pred'][rank] = {class_text(index): value}


if __name__ == '__main__':
    with open(file=r"C:\Users\tomokazu\Desktop\新しいフォルダー\伊勢鈴蘭=angerme-new=12785418621-3.jpg",
              mode='rb') as f:
        image_io = BytesIO(initial_bytes=f.read())
    pos = retinaface(image_io)
    pprint(pos)
    pred = facenet_predict(pos, image_io)
    pprint(pos)
