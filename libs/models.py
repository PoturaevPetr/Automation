import tensorflow as tf
from keras.utils import plot_model
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
from keras.layers import Dropout, Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate 
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os, cv2
from typing import List, Tuple, Dict, Any

config: Dict[str, Dict[str, Any]] = {
    "focus": {
        "type": "classification",
        "model_type": "VGG16",
        "name": "Определение качества",
        "path": "frame_qua.h5",
        "labels": {
            0: "focus", 
            1: "not_focus",
            2: "part_focus"
        }
    },
    "matherial_type":{
        "type": "classification",
        "model_type": "VGG16",
        "name": "Определние типа материала",
        "path": "math_type.h5",
        "labels": {
            0: "AZ31_150",
            1: "AZ31_200",
            2: "AZ31_250",
            3: "AZ31_300",
            4: "AZ31_SRC",
            5: "Brass",
            6: "Cooper",
            7: "Mg",
            8: "Steel"
        }
    },
    "mark_steel":{
        "type": "classification",
        "model_type": "VGG16",
        "name": "Опеределение марки стали",
        "path": "steel_mark.h5",
        "labels": {
            0: "09Г2С",
            1: "12Х18Н10Т",
            2: "15ХМ",
            3: "20Х13НЛ",
            4: "20ХГ",
        }
    },
    "defects": {
        "type": "classification",
        "model_type": "VGG16",
        "name": "Определение типа дефекта",
        "path": "defects.h5",
        "labels": {
            0: "Crazing",
            1: "Inclusion",
            2: "Patches",
            3: "Pitted",
            4: "Rolled",
            5: "Scratches",
        }
    },
    "grains": {
        "type": "segmentation",
        "model_type": "Unet",
        "name": "Локализация зерен",
        "path": "grains.h5",
        "labels": {}
    }
}

# Определим интерфейс для моделей
class BaseModel:
    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings

    def predict(self, data: Any) -> Any:
        """Предсказание модели"""
        raise NotImplementedError


def initialize_models(config: Dict[str, Dict[str, Any]]) -> Dict[str, BaseModel]:
    models: Dict[str, BaseModel] = {}
    for model_name, settings in config.items():
        model_type = settings.get('type')
        if model_type == "classification":
            models[model_name] = ModelClassification(settings)
        elif model_type == "segmentation":
            models[model_name] = ModelSegmentation(settings)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type} для модели {model_name}")
    return models


class ModelClassification(BaseModel):
    def __init__(self, settings: Dict[str, Any]) -> None:
        self.img_width, self.img_height = 224, 224
        self.labels: Dict[int, str] = settings['labels']
        self.num_classes: int = len(self.labels)
        model_type: str = settings['model_type']
        self.model: Sequential

        if model_type == "MobileNet":
            from keras.applications import MobileNet
            base_model = MobileNet(
                input_shape=(self.img_width, self.img_height, 3),
                include_top=False,
                classes=self.num_classes
            )
        elif model_type == "VGG16":
            from keras.applications import VGG16
            base_model = VGG16(
                input_shape=(self.img_width, self.img_height, 3),
                include_top=False,
                classes=self.num_classes
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model = Sequential([
            base_model,
            Dropout(0.1),
            GlobalAveragePooling2D(name='global_average_pooling2d'),
            Dropout(0.1),
            Dense(self.num_classes, activation='softmax', name='predictions')
        ])

        weight_path = os.path.join(os.getcwd(), "../models", settings['path'])
        self.model.load_weights(weight_path, skip_mismatch=True)

    def predict(self, img_array: np.ndarray) -> Tuple[str, float]:
        x = img_array
        if x.shape != (self.img_width, self.img_height, 3):
            x = np.resize(x, (self.img_width, self.img_height, 3))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        preds = self.model.predict(x)
        pred_prob = np.max(preds)
        pred_class_idx = np.argmax(preds)
        pred_label = self.labels.get(pred_class_idx, str(pred_class_idx))
        return pred_label, pred_prob

class Preprocessing:
    @staticmethod
    def open_image(img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found at {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def fragmentation(image: np.ndarray, crop: bool = True, rows: int = 6, cols: int = 8) -> List[np.ndarray]:
        height, width = image.shape[:2]
        fragment_height = height // rows
        fragment_width = width // cols
        fragments: List[np.ndarray] = []

        if crop:
            for r in range(rows):
                for c in range(cols):
                    y1 = r * fragment_height
                    y2 = (r + 1) * fragment_height if r != rows - 1 else height
                    x1 = c * fragment_width
                    x2 = (c + 1) * fragment_width if c != cols - 1 else width
                    fragments.append(image[y1:y2, x1:x2])
        else:
            fragments.append(image)
        return fragments

    @staticmethod
    def defragmentation(fragments: List[np.ndarray], rows: int, cols: int, original_width: int, original_height: int) -> np.ndarray:
        reconstructed = np.zeros((original_height, original_width, 3), dtype=fragments[0].dtype)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                frag = fragments[idx]
                y1 = r * (original_height // rows)
                y2 = y1 + frag.shape[0]
                x1 = c * (original_width // cols)
                x2 = x1 + frag.shape[1]
                reconstructed[y1:y2, x1:x2] = frag
                idx += 1
        return reconstructed

class ModelSegmentation(BaseModel):
    def __init__(self, settings: Dict[str, Any]) -> None:
        self.sz = (256, 256, 3)
        self.model = self.unet()
        weight_path = os.path.join(os.getcwd(), "../models", settings['path'])
        self.model.load_weights(weight_path)

    @staticmethod
    def mean_iou(y_true, y_pred):
        yt0 = y_true[:, :, :, 0]
        yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
        inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
        union = tf.math.count_nonzero(tf.add(yt0, yp0))
        return tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, 'float32'))

    def unet(self) -> Model:
        inputs = Input(self.sz)
        x = inputs
        f = 8
        layers: List[tf.Tensor] = []

        # Downsampling
        for _ in range(6):
            x = Conv2D(f, 3, activation='relu', padding='same')(x)
            x = Conv2D(f, 3, activation='relu', padding='same')(x)
            layers.append(x)
            x = MaxPooling2D()(x)
            f *= 2

        # Bottleneck
        ff2 = 64
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, layers[-1]])

        # Upsampling
        for i in range(5):
            ff2 //= 2
            f //= 2
            x = Conv2D(f, 3, activation='relu', padding='same')(x)
            x = Conv2D(f, 3, activation='relu', padding='same')(x)
            x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
            x = Concatenate(axis=3)([x, layers[-(i+2)]])  # сосредоточиться на правильных слоях

        # Final conv
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        outputs = Conv2D(1, 1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def predict(self, img_path: str) -> List[np.ndarray]:
        src_image = cv2.imread(img_path)
        if src_image is None:
            raise ValueError(f"Image not found at {img_path}")
        original_shape = src_image.shape[:2]
        resized_img = cv2.resize(src_image, (self.sz[1], self.sz[0]), interpolation=cv2.INTER_LINEAR)
        normalized_img = resized_img / 255.0
        input_tensor = np.expand_dims(normalized_img, axis=0)
        pred_mask = self.model.predict(input_tensor)[0, :, :, 0]
        mask_binary = (pred_mask >= 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours


models = initialize_models(config)

class Cascade:
    def __init__(self, models: Dict[str, Any]) -> None:
        self.models = models

    def predict(self, img_path: str, crop: bool, params: Tuple[int, int]) -> None:
        image_obj = Preprocessing.open_image(img_path)
        frames = Preprocessing.fragmentation(image_obj, crop, params[0], params[1])
        grains_list: List[np.ndarray] = []
        areas: List[float] = []

        for i, frame in enumerate(frames):
            focus_label, _ = self.models['focus'].predict(frame)
            if focus_label in ['not_focus', 'part_focus']:
                matherial_type_label, _ = self.models['matherial_type'].predict(frame)
                try:
                    matherial_type_idx = int(matherial_type_label)  # ожидается числовая метка
                except ValueError:
                    matherial_type_idx = -1

                if matherial_type_idx == 8:
                    mark_steel = self.models['mark_steel'].predict(frame)

                if matherial_type_idx in list(range(0, 5)) + [7]:
                    grains = self.models['grains'].predict(frame)
                    grains_list.extend(grains)

                    result_img = frame.copy()
                    cv2.drawContours(result_img, grains, -1, (255, 0, 0), 2)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    filename = os.path.basename(img_path)
                    name, ext = os.path.splitext(filename)
                    cv2.imwrite(f'{name}/{name}_{i}{ext}', result_img)

                # Предполагается, что есть еще дефекты
                # self.models['defects'].predict(frame)
                # self.models['defects_local'].predict(frame)

        # Анализ объектов
        for grain in grains_list:
            area = cv2.contourArea(grain)
            if area > 100:
                areas.append(area)

# Вспомогательные функции для анализа
def calc_linear_size(contour: np.ndarray) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, float]]:
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    dist_horizontal = np.linalg.norm(np.array(rightmost) - np.array(leftmost))
    dist_vertical = np.linalg.norm(np.array(bottommost) - np.array(topmost))
    return (
        {'l': leftmost, 'r': rightmost, 't': topmost, 'b': bottommost},
        {'dh': dist_horizontal, 'dv': dist_vertical}
    )

def predict(img_path: str, crop: bool, params: Tuple[int, int]) -> None:
    img = Preprocessing.open_image(img_path)
    frames = Preprocessing.fragmentation(img, crop, params[0], params[1])
    areas: List[float] = []
    dh: List[float] = []
    dv: List[float] = []

    for i, frame in enumerate(frames):
        grains = models['grains'].predict(frame)
        result_img = frame.copy()
        cv2.drawContours(result_img, grains, -1, (255, 0, 0), 2)

        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        os.makedirs(name, exist_ok=True)
        cv2.imwrite(f'{name}/{name}_{i}{ext}', result_img)

        for grain in grains:
            area = cv2.contourArea(grain)
            if area > 100:
                areas.append(area)
                pp, ls = calc_linear_size(grain)
                dv.append(ls['dv'])
                dh.append(ls['dh'])
                color = (0, 0, 255)
                for point in pp.values():
                    cv2.circle(result_img, point, 3, color, -1)
                cv2.line(result_img, pp['l'], pp['r'], color, 2)
                cv2.line(result_img, pp['t'], pp['b'], color, 2)

        # Можно отображать результат
        # plt.imshow(result_img); plt.axis('off'); plt.show()

    # Визуализация гистограмм
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.hist(np.array(areas) / 100000, bins=40, edgecolor='black')
    plt.xlabel('Площадь')
    plt.ylabel('Количество объектов')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(np.array(dh + dv) / 10000, bins=40, edgecolor='black')
    plt.xlabel('Линейный размер, нм')
    plt.ylabel('Количество объектов')
    plt.show()