import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import numpy as np
import os
import argparse

from pathlib import Path, PosixPath
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


class HomographyTrain:
    """
    A class for teaching homography matrices with improved accuracy.
    Supports batch images as well as single images.
    """
    
    def __init__(self, camera_type: str, use_ml: bool = False) -> None:
        """
        A class constructor for obtaining a homography matrix.
        
        Args:
            camera_type: the type of camera for which the homography matrix will be calculated ('bottom'|'top')
            use_ml: if True, uses MLPRegressor instead of homography for non-linear mapping
        Returns:
            None
        """
        self.camera_type = camera_type
        self.main_camera = "door2"
        self.H = None
        self.use_ml = use_ml
        self.mlp_model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def _get_json_coordinate(self, path_json: str, preffix_image: str) -> Dict[str, List[str]]:
        """
        Get image coordinates.

        Args:
            image_path: path to images;
            camera_type: type of camera, "bottom" or "top".

        Returns:
            image_characteristic: Dictionary of image characteristics, points from a json file.
        """
        image_characteristic = {
            "image_main": [],
            "image_second": [],
            "points_main": [],
            "points_second": []
        }

        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for idx in range(len(data)):
            points_main = []
            points_second = []

            variable_main = data[idx]["file1_path"].split('/')
            variable_second = data[idx]["file2_path"].split('/')

            image_characteristic["image_main"].append(f"{preffix_image}/{variable_main[-3]}/{variable_main[-2]}/{variable_main[-1]}")
            image_characteristic["image_second"].append(f"{preffix_image}/{variable_second[-3]}/{variable_second[-2]}/{variable_second[-1]}")

            for p in zip(data[idx]["image1_coordinates"], data[idx]["image2_coordinates"]):
                points_main.append([p[0]["x"], p[0]["y"]])
                points_second.append([p[1]["x"], p[1]["y"]])

            image_characteristic["points_main"].append(np.array(points_main, dtype=np.float32))
            image_characteristic["points_second"].append(np.array(points_second, dtype=np.float32))

        return image_characteristic
    
    def _draw_points_on_image(self, image: np.array, points_list: list, color=(0, 0, 255), radius=5, thickness=20):
        """
        Draws points on an image by coordinates

        Args:
            image: image (numpy array)
            points_list: a list of dictionaries with the keys 'x' and 'y'
            color: the point color
            radius: the radius of a point in pixels
            thickness: outline thickness

        Returns:
            image_with_points: image with drawn dots
        """
        image_points = image.copy()

        for number, point in enumerate(points_list):
            x, y = int(point[0]), int(point[1])
            cv2.circle(image_points, (x, y), radius, color, thickness)

            cv2.putText(image_points, str(number + 1), 
                       (x + radius, y - radius), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image_points
    
    def _prepare_points(self, all_image_characteristic: Dict[str, Dict[str, List[str | float]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares the points that need to be converted to the target points of the camera directed at the refrigerator door.
        
        Args:
            all_image_characteristic: Dictionary of image characteristics, points from a json file.
            
        Returns:
            src_all: Set of points of the 'top' or 'bottom' camera;
            dst_all: Set of central camera points.
        """
        all_src = []
        all_dst = []

        for v in all_image_characteristic.values():
            for idx in range(len(v['points_second'])):
                src_pts = v['points_second'][idx]
                dst_pts = v['points_main'][idx]

                if len(src_pts) > 0 and len(dst_pts) > 0 and len(src_pts) == len(dst_pts):
                    all_src.append(src_pts)
                    all_dst.append(dst_pts)

        if not all_src:
            return np.empty((0, 2)), np.empty((0, 2))
        
        src_all = np.vstack(all_src)
        dst_all = np.vstack(all_dst)
        
        return src_all, dst_all
    
    def _normalize_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize points for better numerical stability.
        
        Args:
            points: array of shape (N, 2)
            
        Returns:
            normalized_points: normalized coordinates
            mean: mean of original points
            std: std of original points
        """
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        std[std == 0] = 1.0
        normalized = (points - mean) / std
        return normalized, mean, std
    
    def _denormalize_points(self, points_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Denormalize points back to original scale."""
        return points_norm * std + mean
    
    def _fit_homography_improved(self, src_all: np.ndarray, dst_all: np.ndarray) -> None:
        """
        Trains a homography matrix with improved accuracy using normalization and RANSAC.
        
        Args:
            src_all: Set of points of the 'top' or 'bottom' camera;
            dst_all: Set of central camera points.
        """
        print("Производится расчёт матрицы гомографии (улучшенный метод)...")
        
        src_norm, src_mean, src_std = self._normalize_points(src_all)
        dst_norm, dst_mean, dst_std = self._normalize_points(dst_all)
        
        methods = [
            (cv2.RANSAC, 3.0, "RANSAC (порог=3.0)"),
            (cv2.RANSAC, 5.0, "RANSAC (порог=5.0)"),
            (cv2.LMEDS, 0, "LMEDS"),
            (cv2.RHO, 3.0, "RHO (порог=3.0)")
        ]
        
        best_H = None
        best_inlier_ratio = 0
        best_method = ""
        
        for method, threshold, method_name in methods:
            try:
                if method == cv2.LMEDS:
                    H, mask = cv2.findHomography(src_norm, dst_norm, method, 0, maxIters=5000)
                else:
                    H, mask = cv2.findHomography(
                        src_norm, dst_norm, 
                        method=method,
                        ransacReprojThreshold=threshold,
                        maxIters=5000,
                        confidence=0.999
                    )
                
                if H is not None:
                    inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
                    print(f"  {method_name}: inlier_ratio = {inlier_ratio:.2%}")
                    
                    if inlier_ratio > best_inlier_ratio:
                        best_H = H
                        best_inlier_ratio = inlier_ratio
                        best_method = method_name
            except Exception as e:
                print(f"  {method_name}: ошибка - {e}")
        
        if best_H is None:
            print("Не удалось найти матрицу гомографии!")
            return
        
        print(f"Выбран метод: {best_method} (inlier_ratio={best_inlier_ratio:.2%})")
        
        T_src = np.array([[1/src_std[0], 0, -src_mean[0]/src_std[0]],
                          [0, 1/src_std[1], -src_mean[1]/src_std[1]],
                          [0, 0, 1]])
        T_dst = np.array([[dst_std[0], 0, dst_mean[0]],
                          [0, dst_std[1], dst_mean[1]],
                          [0, 0, 1]])
        
        self.H = T_dst @ best_H @ T_src
        self.H = self.H / self.H[2, 2]
        
        print(f"Матрица гомографии найдена: {self.H}")
    
    def _fit_ml_model(self, src_all: np.ndarray, dst_all: np.ndarray) -> None:
        """
        Trains an MLP regressor for non-linear mapping (better for fisheye).
        
        Args:
            src_all: Set of points of the 'top' or 'bottom' camera;
            dst_all: Set of central camera points.
        """
        print("Производится обучение MLP регрессора для нелинейного маппинга...")
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(src_all)
        y_scaled = self.scaler_y.fit_transform(dst_all)
        
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=(128, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        
        self.mlp_model.fit(X_scaled, y_scaled)
        
        print(f"MLP модель обучена. Итераций: {self.mlp_model.n_iter_}, потери: {self.mlp_model.loss_:.6f}")
    
    def _fit(self, src_all: np.ndarray, dst_all: np.ndarray) -> None:
        """
        Trains a model (homography or ML) on all points.
        
        Args:
            src_all: Set of points of the 'top' or 'bottom' camera;
            dst_all: Set of central camera points.
        """
        if self.use_ml:
            self._fit_ml_model(src_all, dst_all)
        else:
            self._fit_homography_improved(src_all, dst_all)
        
    def batch_euclidean_distance(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """
        Calculating an array of distances for each pair of points.

        Args:
            points1: Array of points of shape (N, 2) - first set of points;
            points2: Array of points of shape (N, 2) - second set of points.

        Returns:
            distances: Array of shape (N,) containing Euclidean distances for each pair.
        """
        return np.sqrt(np.sum((points1 - points2)**2, axis=1))

    def mean_euclidean_distance(self, pred_points: np.ndarray, true_points: np.ndarray) -> float:
        """
        Calculation of the average Euclidean distance.

        Args:
            pred_points: Array of predicted points of shape (N, 2);
            true_points: Array of true points of shape (N, 2).

        Returns:
            mean_dist: Average Euclidean distance between predicted and true points.
        """
        return np.mean(self.batch_euclidean_distance(pred_points, true_points))

    def median_euclidean_distance(self, pred_points: np.ndarray, true_points: np.ndarray) -> float:
        """
        Calculation of the median Euclidean distance.

        Args:
            pred_points: Array of predicted points of shape (N, 2);
            true_points: Array of true points of shape (N, 2).

        Returns:
            median_dist: Median Euclidean distance between predicted and true points.
        """
        return np.median(self.batch_euclidean_distance(pred_points, true_points))
    
    def _predict_points(self, points: np.ndarray) -> np.ndarray:
        """
        Predict transformed points using trained model.
        
        Args:
            points: array of shape (N, 2)
            
        Returns:
            transformed_points: array of shape (N, 2)
        """
        if self.use_ml and self.mlp_model is not None:
            X_scaled = self.scaler_X.transform(points)
            y_pred_scaled = self.mlp_model.predict(X_scaled)
            return self.scaler_y.inverse_transform(y_pred_scaled)
        elif self.H is not None:
            points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
            transformed = cv2.perspectiveTransform(points_reshaped, self.H)
            return transformed.reshape(-1, 2)
        else:
            raise RuntimeError("Модель не обучена")

    def _calculate_metric(self, src_all: np.ndarray, dst_all: np.ndarray) -> None:
        """
        Function for preparing data and calculating metrics, as well as displaying results.

        Args:
            pred_points: Array of predicted points of shape (N, 2);
            true_points: Array of true points of shape (N, 2).
        """
        points_transformed = self._predict_points(src_all)
        
        errors = self.batch_euclidean_distance(points_transformed, dst_all)
        
        print(f"Среднее евклидовое расстояние (MED): {np.mean(errors):.2f} px")
        print(f"Медианное евклидовое расстояние: {np.median(errors):.2f} px")
        print(f"Стандартное отклонение: {np.std(errors):.2f} px")
        print(f"95-й процентиль: {np.percentile(errors, 95):.2f} px")
        print(f"Максимальная ошибка: {np.max(errors):.2f} px")

    def _visualization_best_result(self, all_image_characteristic: Dict[str, Dict[str, List[str | float]]], 
                                    src_all: np.ndarray, dst_all: np.ndarray, preffix: str) -> None:
        """
        Visualizing the best result.

        Args:
            src_all: Set of points of the 'top' or 'bottom' camera;
            dst_all: Set of central camera points:
            all_image_characteristic: Dictionary of image characteristics, points from a json file.
            preffix: prefix for output filename
        """
        points_transformed = self._predict_points(src_all)
        
        metric = self.batch_euclidean_distance(points_transformed, dst_all)
        
        l = 0
        for jsn_pth in all_image_characteristic:
            length = len(all_image_characteristic[jsn_pth]['image_second'])
            all_image_characteristic[jsn_pth]['metrics'] = metric[l:l + length]
            l += length

        best_metric = 0
        main_path = ''
        second_path = ''
        best_point_second = None
        best_point_main = None
        
        for jsn_pth in all_image_characteristic:
            for idx in range(len(all_image_characteristic[jsn_pth]['metrics'])):
                if all_image_characteristic[jsn_pth]['metrics'][idx] > best_metric:
                    best_metric = all_image_characteristic[jsn_pth]['metrics'][idx]
                    main_path = jsn_pth.parent / "/".join(Path(all_image_characteristic[jsn_pth]['image_main'][idx]).parts[2:])
                    second_path = jsn_pth.parent / "/".join(Path(all_image_characteristic[jsn_pth]['image_second'][idx]).parts[2:])
                    best_point_second = all_image_characteristic[jsn_pth]['points_second'][idx]
                    best_point_main = all_image_characteristic[jsn_pth]['points_main'][idx]

        if main_path and second_path and best_point_second is not None:
            main_image = cv2.imread(str(main_path))
            second_image = cv2.imread(str(second_path))
            
            if main_image is None or second_image is None:
                print(f"Не удалось загрузить изображения: {main_path} или {second_path}")
                return
            
            points_transformed = self._predict_points(best_point_second)

            main_image = self._draw_points_on_image(main_image, best_point_main, color=(0, 0, 255), radius=15, thickness=10)
            main_image = self._draw_points_on_image(main_image, points_transformed, color=(0, 255, 0), radius=15, thickness=10)
            second_image = self._draw_points_on_image(second_image, best_point_second, color=(0, 0, 255), radius=15, thickness=10)
            
            os.makedirs('result', exist_ok=True)
            merged = np.hstack((main_image, second_image))
            cv2.imwrite(f'result/{self.camera_type}_{preffix}_merged.jpg', merged)
            print(f'Результат сохранён по пути: result/{self.camera_type}_{preffix}_merged.jpg')

    def _save_matrix(self) -> None:
        """
        Saving the result without overwriting existing data.
        """
        file_path = Path('./solution/Homography.json')
        
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        if self.use_ml:
            import joblib
            ml_path = Path('./solution/ml_model.pkl')
            joblib.dump({
                'model': self.mlp_model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'camera_type': self.camera_type
            }, ml_path)
            print(f"ML модель сохранена в {ml_path}")
            existing_data[f"{self.camera_type}_ml"] = True
        else:
            H_list = self.H.tolist() if hasattr(self.H, 'tolist') else self.H
            existing_data[self.camera_type] = H_list

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2)

        print(f"Модель сохранена для камеры '{self.camera_type}' в {file_path}")

    def train(self, train_json_list: List[str | PosixPath], val_json_list: List[str | PosixPath]) -> None:
        """
        Learning the homography matrix.
        
        Args:
            train_json_list: A list of json files with paths to images and dots.
            val_json_list: A list of json files for validation.
        
        Return:
            None
        """
        all_image_characteristic = {}
        all_image_characteristic_val = {}
        
        for jsn in train_json_list:
            if self.camera_type in jsn.as_posix():
                image_characteristic = self._get_json_coordinate(jsn, "train")
                all_image_characteristic[jsn] = image_characteristic

        for jsn in val_json_list:
            if self.camera_type in jsn.as_posix():
                image_characteristic_val = self._get_json_coordinate(jsn, "val")
                all_image_characteristic_val[jsn] = image_characteristic_val

        if not all_image_characteristic:
            raise ValueError(f"Нет тренировочных данных для камеры {self.camera_type}")
        
        src_all, dst_all = self._prepare_points(all_image_characteristic)
        val_points_src, val_points_dst = self._prepare_points(all_image_characteristic_val)
        
        print(f"\n{'='*50}")
        print(f"Обучение для камеры: {self.camera_type}")
        print(f"Тренировочных точек: {len(src_all)}")
        print(f"Валидационных точек: {len(val_points_src)}")
        print(f"{'='*50}\n")
        
        self._fit(src_all, dst_all)
        
        print("\n==================== Расчёт метрик для Train сплита ====================")
        self._calculate_metric(src_all, dst_all)
        
        if len(val_points_src) > 0:
            print("\n==================== Расчёт метрик для Val сплита ====================")
            self._calculate_metric(val_points_src, val_points_dst)
        
        print("\n==================== Визуализация результата для Train ====================")
        self._visualization_best_result(all_image_characteristic, src_all, dst_all, 'train')
        
        if all_image_characteristic_val:
            print("\n==================== Визуализация результата для Val ====================")
            self._visualization_best_result(all_image_characteristic_val, val_points_src, val_points_dst, 'val')
        
        self._save_matrix()


def get_image_in_folder(path_to_image: str, suffix: List[str] = [".png", ".jpeg", ".jpg", ".tif", ".tiff", ".json"]) -> List[PosixPath]:
    """
    Search for image paths and image names.

    param:
        directory: str
            Directory for image search.

    returns:
        image_paths: list[PosixPath]
            List of image paths.
    """
    return [file_name for file_name in Path(path_to_image).rglob("*")
                   if file_name.suffix in suffix and ".ipynb_checkpoints" not in file_name.as_posix()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_type", choices=["top", "bottom"])
    parser.add_argument("--use_ml", action="store_true", help="Use MLP regressor instead of homography")
    args = parser.parse_args()

    train_json = get_image_in_folder("train", suffix=[".json"])
    val_json = get_image_in_folder("val", suffix=[".json"])

    homogr = HomographyTrain(args.camera_type, use_ml=args.use_ml)
    homogr.train(train_json, val_json)