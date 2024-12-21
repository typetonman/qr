import cv2
import numpy as np
import time
import os
import sys
from pyzbar import pyzbar

class QRDetector:
    def __init__(self, input_source=0):
        self.cap = cv2.VideoCapture(input_source)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Параметры быстрого зума
        self.zoom_active = True
        self.zoom_speed = 3.0
        self.max_zoom = 2.0
        self.min_zoom = 0.5
        self.current_zoom = 1.0
        self.zoom_direction = 1
        self.last_time = time.time()
        self.frequency = 1/10.0
        
        # Параметры поворота
        self.angles = [0, 90, 180, 270]
        self.current_angle_index = 0
        
        # Параметры улучшения изображения
        self.min_qr_size = 20
        self.max_qr_size = 500
        self.border_size = 20
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def enhance_small_qr(self, frame):
        # Добавление белой рамки
        bordered = cv2.copyMakeBorder(
            frame,
            self.border_size, self.border_size,
            self.border_size, self.border_size,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        
        # Улучшение контраста
        gray = cv2.cvtColor(bordered, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        
        # Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary

    def apply_zoom_and_rotation(self, frame):
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= self.frequency:
            # Обновление зума
            if self.zoom_active:
                self.current_zoom += self.zoom_speed * self.zoom_direction
                if self.current_zoom >= self.max_zoom:
                    self.zoom_direction = -1
                elif self.current_zoom <= self.min_zoom:
                    self.zoom_direction = 1
            
            # Обновление угла поворота
            self.current_angle_index = (self.current_angle_index + 1) % len(self.angles)
            self.last_time = current_time
        
        # Применение зума
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        new_h = int(height / self.current_zoom)
        new_w = int(width / self.current_zoom)
        
        start_x = max(0, center_x - new_w//2)
        start_y = max(0, center_y - new_h//2)
        end_x = min(width, start_x + new_w)
        end_y = min(height, start_y + new_h)
        
        if end_x <= start_x or end_y <= start_y:
            return frame
            
        cropped = frame[start_y:end_y, start_x:end_x]
        if cropped.size == 0:
            return frame
            
        zoomed = cv2.resize(cropped, (width, height))
        
        # Применение поворота
        angle = self.angles[self.current_angle_index]
        if angle > 0:
            matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            zoomed = cv2.warpAffine(zoomed, matrix, (width, height))
        
        return zoomed

    def detect_qr_codes(self, frame):
        # Масштабирование для маленьких QR-кодов
        scales = [1.0, 1.5, 2.0, 2.5]
        detected_codes = []
        
        for scale in scales:
            # Масштабирование кадра
            if scale != 1.0:
                width = int(frame.shape[1] * scale)
                height = int(frame.shape[0] * scale)
                scaled = cv2.resize(frame, (width, height))
            else:
                scaled = frame
            
            # Улучшение изображения
            enhanced = self.enhance_small_qr(scaled)
            
            # Поиск QR-кодов
            qr_codes = pyzbar.decode(enhanced)
            
            for qr in qr_codes:
                points = np.array(qr.polygon, np.int32)
                if scale != 1.0:
                    points = (points / scale).astype(np.int32)
                points = points.reshape((-1, 1, 2))
                detected_codes.append((points, qr.data))
        
        return detected_codes

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed = self.apply_zoom_and_rotation(frame)
            qr_codes = self.detect_qr_codes(processed)
            
            # Отрисовка найденных QR-кодов
            for points, data in qr_codes:
                cv2.polylines(processed, [points], True, (0, 0, 255), 3)
                try:
                    text = data.decode('utf-8')
                    x, y = points[0][0]
                    cv2.putText(processed, text, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Ошибка декодирования QR: {str(e)}")
            
            cv2.imshow('QR Detection', processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='QR детектор с улучшенным распознаванием')
    parser.add_argument('input', help='Путь к видео файлу или номер камеры (0, 1, ...)')
    
    args = parser.parse_args()
    
    try:
        input_source = int(args.input)
    except ValueError:
        input_source = args.input
        if not os.path.exists(input_source):
            print(f"Ошибка: Файл {input_source} не найден")
            sys.exit(1)
    
    detector = QRDetector(input_source)
    detector.run()

