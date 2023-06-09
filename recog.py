from functools import cache
from io import BytesIO
from numpy import pi, sqrt
from PIL import Image
from numpy import asarray, array, arctan2
from retinaface.pre_trained_models import get_model
from torch.cuda import is_available
from torchvision.transforms.functional import to_tensor
from torch import stack, load, no_grad, topk, device
from torch.nn import Module, Sequential, Softmax

dev = device(device='cuda') if is_available() else device(device='cpu')
print(f'device: {dev}')

retinaface_model = get_model("resnet50_2020-07-20", max_size=512, device=dev)
retinaface_model.eval()


@cache
def class_text(x: int):
    return ['一岡伶奈', '上國料萌衣', '中山夏月姫', '中島早貴', '中西香菜', '井上玲音', '伊勢鈴蘭', '佐々木莉佳子',
            '佐藤優樹', '入江里咲', '八木栞', '前田こころ', '加賀楓', '勝田里奈', '北原もも', '北川莉央', '和田彩花',
            '和田桜子', '嗣永桃子', '夏焼雅', '太田遥香', '室田瑞希', '宮崎由加', '宮本佳林', '小川麗奈', '小林萌花',
            '小片リサ', '小田さくら', '小野瑞歩', '小野田紗栞', '小関舞', '尾形春水', '山岸理子', '山木梨沙', '山﨑夢羽',
            '山﨑愛生', '岡井千聖', '岡村ほまれ', '岡村美波', '岸本ゆめの', '島倉りか', '川名凜', '川村文乃', '工藤由愛',
            '工藤遥', '平井美葉', '平山遊季', '広本瑠璃', '広瀬彩海', '徳永千奈美', '斉藤円香', '新沼希空', '有澤一華',
            '松本わかな', '松永里愛', '梁川奈々美', '森戸知沙希', '植村あかり', '横山玲奈', '橋迫鈴', '櫻井梨央', '段原瑠々',
            '江口紗耶', '江端妃咲', '河西結心', '浅倉樹々', '浜浦彩乃', '清水佐紀', '清野桃々姫', '為永幸音',
            '熊井友理奈', '牧野真莉愛', '生田衣梨奈', '田中れいな', '田代すみれ', '田口夏実', '田村芽実', '相川茉穂',
            '矢島舞美', '石山咲良', '石栗奏美', '石田亜佑美', '福田真琳', '秋山眞緒', '稲場愛香', '窪田七海',
            '竹内朱莉', '笠原桃奈', '筒井澪心', '米村姫良々', '羽賀朱音', '船木結', '菅谷梨沙子', '萩原舞', '藤井梨央',
            '西田汐里', '西﨑美空', '譜久村聖', '谷本安美', '豫風瑠乃', '道重さゆみ', '遠藤彩加里', '里吉うたの', '野中美希',
            '野村みな美', '金澤朋子', '鈴木愛理', '鈴木香音', '鞘師里保', '須藤茉麻', '飯窪春菜', '高木紗友希',
            '高瀬くるみ'][x]


@no_grad()
def retinaface(image_data: BytesIO):
    image_data.seek(0)
    try:
        image = Image.open(image_data)
        image_arr = array(image)
    except Exception as e:
        print("invalid Image")
        return []
    res = retinaface_model.predict_jsons(image=image_arr)
    if res.__len__() == 1 and res[0]['score'] == -1:
        return []
    return res


if __name__ == '__main__':
    with open(file=r"C:\Users\tomokazu\Desktop\新しいフォルダー\伊勢鈴蘭=angerme-new=12741916432-2.jpg", mode='rb') as f:
        print(retinaface(BytesIO(f.read())))


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
    if image.mode != 'RGB':
        print(image.mode)
        image = image.convert(mode='RGB')
    print(image.mode)
    for order, face in enumerate(res):
        bbox, score, landmarks = face.values()
        face['pred'] = dict()
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        trans = truncate(landmarks)
        face['rotate'] = trans[1]
        rotated = image.rotate(angle=trans[1] * 360 / (2 * pi), center=trans[0])
        image_size = max(face_width, face_height) * sqrt(2) // 2

        cropped_face = rotated.crop((trans[0][0] - image_size, trans[0][1] - image_size, trans[0][0] + image_size,
                                     trans[0][1] + image_size)).resize(size=(224, 224))
        if image_size < 70:
            face['pred']['stat'] = {'invalid': f'face too small. {(face_height, face_width)}'}
            continue
        cropped_tensor = to_tensor(cropped_face).unsqueeze(0)

        predict = topk(facenet_model(cropped_tensor.to(dev)).to(device(device='cpu')), dim=1, k=5)
        for rank, (value, index) in enumerate(zip(predict.values[0].tolist(), predict.indices[0].tolist())):
            face['pred']['stat'] = {'success': ''}
            face['pred'][rank] = {class_text(index): value}
