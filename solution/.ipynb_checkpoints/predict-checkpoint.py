import json
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
import joblib
import argparse


class HomographyMapper:
    """
    Class for loading and applying homography matrices or ML models.
    """
    
    def __init__(self, file_path: str) -> None:
        """
        Initializes mapper and loads model from file.
        
        Args:
            file_path: Path to JSON file with homography matrices or .pkl file with ML model.
        """
        self.H_bottom = None
        self.H_top = None
        self.use_ml = False
        self.mlp_model = None
        self.scaler_X = None
        self.scaler_y = None
        self._load_matrix(file_path)
    
    def _load_matrix(self, file_path: str) -> None:
        """Loads homography matrices or ML model from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == '.pkl':
            data = joblib.load(file_path)
            self.use_ml = True
            self.mlp_model = data['model']
            self.scaler_X = data['scaler_X']
            self.scaler_y = data['scaler_y']
            print(f"Loaded ML model")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "bottom" in data:
            self.H_bottom = np.array(data["bottom"], dtype=np.float32)
            print("Loaded homography for bottom camera")
        
        if "top" in data:
            self.H_top = np.array(data["top"], dtype=np.float32)
            print("Loaded homography for top camera")
    
    def predict(self, x: float, y: float, source: str) -> Tuple[float, float]:
        """
        Transforms point from source camera to door2 camera.
        
        Args:
            x: X coordinate on source camera
            y: Y coordinate on source camera
            source: Camera type ("top" or "bottom")
        
        Returns:
            (x_door, y_door): Transformed coordinates on door2 camera
        """
        if self.use_ml:
            point = np.array([[x, y]], dtype=np.float32)
            point_scaled = self.scaler_X.transform(point)
            pred_scaled = self.mlp_model.predict(point_scaled)
            pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 2))
            return float(pred[0, 0]), float(pred[0, 1])
        
        H = self.H_top if source == "top" else self.H_bottom
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, H)
        
        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homography Mapper CLI")
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.json or .pkl)")
    parser.add_argument("--x", type=float, required=True, help="X coordinate")
    parser.add_argument("--y", type=float, required=True, help="Y coordinate")
    parser.add_argument("--source", type=str, required=True, choices=["top", "bottom"], help="Camera source")
    
    args = parser.parse_args()
    
    mapper = HomographyMapper(args.model)
    x_pred, y_pred = mapper.predict(args.x, args.y, args.source)
    print(f"\nInput: ({args.x}, {args.y}) on {args.source}")
    print(f"Output: ({x_pred:.2f}, {y_pred:.2f}) on door2")