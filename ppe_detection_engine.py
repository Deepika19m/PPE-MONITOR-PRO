"""
PPE Detection Engine
Core detection engine for PPE compliance monitoring using YOLOv8 with integrated face recognition
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
import os
import time
import logging

# Keep Ultralytics config inside the project to avoid AppData permission issues.
_project_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_project_dir, "Ultralytics"), exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", _project_dir)

from ultralytics import YOLO

# Import face recognition engine
try:
    from face_recognition_engine import FaceRecognitionEngine
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("Face recognition engine not available")

# Detection Engine Constants
PROGRESS_UPDATE_INTERVAL = 0.5  # seconds between progress updates
DEFAULT_DETECTION_PERSISTENCE = 3  # frames to keep detections visible
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45


class PPEDetectionEngine:
    """Core PPE detection engine using YOLOv8 with integrated face recognition"""

    def __init__(self, model_path: str = "best.pt", enable_face_recognition: bool = True):
        """Initialize the PPE detection engine

        Args:
            model_path: Path to the YOLOv8 model file
            enable_face_recognition: Whether to enable face recognition features
        """
        self.model_path = model_path
        self.model = None
        self.class_names = {
            0: 'Hardhat',
            1: 'Mask',
            2: 'NO-Hardhat',
            3: 'NO-Mask',
            4: 'NO-Safety Vest',
            5: 'Person',
            6: 'Safety Cone',
            7: 'Safety Vest',
            8: 'machinery',
            9: 'vehicle'
        }
        self.compliance_classes = {
            'Hardhat': True,
            'Mask': True,
            'Safety Vest': True,
            'Gloves': True,
            'Goggles': True,
            'NO-Hardhat': False,
            'NO-Mask': False,
            'NO-Safety Vest': False,
            'NO-Gloves': False,
            'NO-Goggles': False
        }
        self.required_ppe_items = ['Hardhat', 'Mask', 'Safety Vest']
        self.hog_person_detector = cv2.HOGDescriptor()
        self.hog_person_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Initialize face recognition engine
        self.face_recognition_enabled = enable_face_recognition and FACE_RECOGNITION_AVAILABLE
        self.face_engine = None

        if self.face_recognition_enabled:
            try:
                self.face_engine = FaceRecognitionEngine()
                logging.info("Face recognition engine initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize face recognition engine: {e}")
                self.face_recognition_enabled = False

        self.load_model()

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize model class labels for robust matching."""
        return str(label).strip().lower().replace("_", "").replace("-", "").replace(" ", "")

    def _resolve_class_name(self, class_id: int) -> str:
        """Resolve class name from the loaded YOLO model when available."""
        if self.model is not None and hasattr(self.model, "names"):
            model_names = self.model.names
            if isinstance(model_names, dict):
                name = model_names.get(class_id)
                if name is not None:
                    return str(name)
            elif isinstance(model_names, (list, tuple)) and 0 <= class_id < len(model_names):
                return str(model_names[class_id])
        return self.class_names.get(class_id, f"Class_{class_id}")

    def _is_person_class(self, class_name: str) -> bool:
        normalized = self._normalize_label(class_name)
        return normalized in {"person", "people", "worker", "workers"}

    def _is_violation_class(self, class_name: str) -> bool:
        normalized = self._normalize_label(class_name)
        return normalized in {
            # Standard canonical names
            "nohardhat", "nomask", "nosafetyvest", "novest",
            "nogloves", "nogoggles", "nogoggle",
            # Actual model output labels (underscores stripped by normalize)
            "nohelmet", "noboots", "none",
            # With-prefix variants
            "withouthelmet", "withoutmask", "withoutvest",
            "withoutgloves", "withoutgoggles",
        }
    # PPE items that require specific body regions to be visible
    # Maps PPE item → which body region fraction of person bbox must be visible
    PPE_BODY_REGION = {
        'Hardhat':     'head',    # top 20% of person bbox
        'Mask':        'head',
        'Goggles':     'head',
        'Safety Vest': 'torso',   # middle 40% of person bbox
        'Gloves':      'hands',   # bottom 60% must be visible (arms/hands)
        'Boots':       'feet',    # bottom 20% of person bbox
    }

    def _estimate_visible_body_regions(self, person_bbox, image_shape):
        """Estimate which body regions are visible from person bbox geometry.

        Conservative rules:
          head, torso  -- always checked (helmet + vest always relevant)
          hands/Gloves -- only if person height >= 55% of image height
          feet/Boots   -- only if person height >= 75% of image height
                          AND bbox bottom is within 15% of image bottom
        """
        px1, py1, px2, py2 = [float(x) for x in person_bbox]
        img_h = float(image_shape[0])
        p_h = max(1.0, py2 - py1)

        # Head and torso are always checked
        visible = {'head', 'torso'}

        # Hands visible only when person occupies >= 55% of frame height
        # (need to see at least down to waist/arms level)
        if p_h >= img_h * 0.55:
            visible.add('hands')

        # Feet visible only in a full-body shot:
        # person height >= 75% of frame AND bottom edge near image bottom
        if p_h >= img_h * 0.75 and py2 >= img_h * 0.85:
            visible.add('feet')

        return visible


    def _canonicalize_class_name(self, class_name: str) -> str:
        """Map model-specific labels to the labels used throughout the UI."""
        normalized = self._normalize_label(class_name)
        canonical_names = {
            # Positive PPE classes
            "hardhat": "Hardhat",
            "helmet": "Hardhat",       # model: 'helmet'
            "mask": "Mask",
            "safetyvest": "Safety Vest",
            "vest": "Safety Vest",     # model: 'vest'
            "gloves": "Gloves",        # model: 'gloves'
            "goggles": "Goggles",      # model: 'goggles'
            "boots": "Boots",          # model: 'boots'
            "safetyglasses": "Goggles",
            # Person classes
            "person": "Person",
            "people": "Person",
            "worker": "Person",
            "workers": "Person",
            # Violation classes — canonical NO-* names
            "nohardhat": "NO-Hardhat",
            "nohelmet": "NO-Hardhat",      # model: 'no_helmet'
            "withouthelmet": "NO-Hardhat",
            "nomask": "NO-Mask",
            "withoutmask": "NO-Mask",
            "nosafetyvest": "NO-Safety Vest",
            "novest": "NO-Safety Vest",
            "withoutvest": "NO-Safety Vest",
            "nogloves": "NO-Gloves",       # model: 'no_gloves'
            "withoutgloves": "NO-Gloves",
            "nogoggle": "NO-Goggles",      # model: 'no_goggle'
            "nogoggles": "NO-Goggles",
            "withoutgoggles": "NO-Goggles",
            "noboots": "NO-Boots",
            "none": "NO-PPE",              # model: 'none' = no PPE at all
            # Equipment
            "safetycone": "Safety Cone",
            "machinery": "machinery",
            "vehicle": "vehicle",
        }
        return canonical_names.get(normalized, str(class_name))

    def _refresh_required_ppe_items(self):
        """Update required PPE items based on the loaded model's positive PPE classes."""
        default_required = ['Hardhat', 'Mask', 'Safety Vest']
        if self.model is None or not hasattr(self.model, "names"):
            self.required_ppe_items = default_required
            return

        if isinstance(self.model.names, dict):
            raw_names = self.model.names.values()
        else:
            raw_names = self.model.names

        detected_positive_items = []
        for raw_name in raw_names:
            canonical = self._canonicalize_class_name(str(raw_name))
            if canonical in {'Hardhat', 'Mask', 'Safety Vest', 'Gloves', 'Goggles', 'Boots'}:
                detected_positive_items.append(canonical)

        self.required_ppe_items = detected_positive_items or default_required

    def _face_to_person_bbox(self, face_bbox: List[float], image_shape) -> List[float]:
        """Expand a face box into an approximate person upper-body box."""
        x1, y1, x2, y2 = face_bbox
        face_w = x2 - x1
        face_h = y2 - y1
        img_h, img_w = image_shape[:2]

        expanded_x1 = max(0, int(x1 - face_w * 0.8))
        expanded_y1 = max(0, int(y1 - face_h * 0.4))
        expanded_x2 = min(img_w, int(x2 + face_w * 0.8))
        expanded_y2 = min(img_h, int(y2 + face_h * 3.2))
        return [expanded_x1, expanded_y1, expanded_x2, expanded_y2]

    def _detect_people_fallback(self, image: np.ndarray) -> List[Dict]:
        """Estimate visible people when the PPE model is unavailable."""
        fallback_people = []

        # Face-based fallback
        if self.face_engine:
            try:
                face_results = self.face_engine.detect_faces(image)
                for face in face_results:
                    fallback_people.append({
                        'bbox': self._face_to_person_bbox(face['bbox'], image.shape),
                        'confidence': 1.0,
                        'class_id': -1,
                        'class_name': 'Person',
                        'is_compliant': None
                    })
            except Exception as e:
                logging.error(f"Fallback face detection failed: {e}")

        # HOG-based fallback for non-face full-body detections
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image if len(image.shape) == 2 else image[:, :, 0]

            boxes, weights = self.hog_person_detector.detectMultiScale(
                gray, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
            for (x, y, w, h), weight in zip(boxes, weights):
                candidate = [int(x), int(y), int(x + w), int(y + h)]
                center_x = (candidate[0] + candidate[2]) / 2
                center_y = (candidate[1] + candidate[3]) / 2
                duplicate = False
                for existing in fallback_people:
                    ex = existing['bbox']
                    ex_center_x = (ex[0] + ex[2]) / 2
                    ex_center_y = (ex[1] + ex[3]) / 2
                    if abs(center_x - ex_center_x) < w * 0.4 and abs(center_y - ex_center_y) < h * 0.4:
                        duplicate = True
                        break
                if not duplicate:
                    fallback_people.append({
                        'bbox': candidate,
                        'confidence': float(weight),
                        'class_id': -1,
                        'class_name': 'Person',
                        'is_compliant': None
                    })
        except Exception as e:
            logging.error(f"Fallback HOG person detection failed: {e}")

        return fallback_people

    def create_fallback_analysis(self, image: np.ndarray, reason: str = "Model not loaded") -> Dict:
        """Return a useful non-model analysis when YOLO weights are unavailable."""
        detections = self._detect_people_fallback(image)
        total_people = len(detections)
        violations = []
        inconclusive = total_people == 0

        if total_people > 0:
            for idx, detection in enumerate(detections):
                violations.append({
                    'type': 'Missing Required PPE',
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'associated_person_idx': idx
                })

        compliance_stats = {
            'total_people': total_people,
            'compliant_people': 0 if total_people > 0 else 0,
            'violations': violations,
            'compliance_rate': 0.0 if total_people > 0 else None,
            'people_with_violations': total_people
        }

        face_results = []
        face_stats = {
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'recognized_people': [],
            'recognition_rate': 0.0
        }

        if self.face_engine:
            try:
                face_results = self.face_engine.recognize_faces(image)
                face_stats = self.face_engine.get_recognition_stats(face_results)
            except Exception as e:
                logging.error(f"Fallback face recognition error: {e}")

        return {
            'detections': detections,
            'compliance_stats': compliance_stats,
            'face_results': face_results,
            'face_stats': face_stats,
            'image_shape': image.shape,
            'analysis_mode': 'fallback',
            'inconclusive': inconclusive,
            'requirements': self.required_ppe_items.copy(),
            'supported_requirements': self.required_ppe_items.copy(),
            'advisory': (
                f"{reason}. Using fallback person visibility analysis. "
                "This fallback can only estimate whether a person is visible and does not verify unsupported PPE like gloves or safety glasses."
            )
        }

    def is_model_ready(self) -> bool:
        """Return True when the YOLO model weights are loaded."""
        return self.model is not None
    
    def load_model(self) -> bool:
        """Load the YOLOv8 model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self._refresh_required_ppe_items()
                return True
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.5,
                      iou_threshold: float = 0.45) -> Dict:
        """Detect objects in an image

        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS

        Returns:
            Dict containing detection results and compliance info
        """
        if self.model is None:
            self.load_model()
        if self.model is None:
            return self.create_fallback_analysis(image, "Model not loaded")

        try:
            # Ensure image has correct number of channels (RGB = 3 channels)
            if len(image.shape) == 3 and image.shape[2] == 4:
                # Convert RGBA to RGB by removing alpha channel
                image = image[:, :, :3]
            elif len(image.shape) == 2:
                # Convert grayscale to RGB
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Convert single channel to RGB
                image = np.repeat(image, 3, axis=2)

            # Run inference
            results = self.model(image, conf=conf_threshold, iou=iou_threshold)

            # Process results
            detections = []
            people_detections = []
            violation_detections = []

            compliance_stats = {
                'total_people': 0,
                'compliant_people': 0,
                'violations': [],
                'compliance_rate': 100.0,
                'people_with_violations': 0
            }

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    # First pass: collect all detections and separate people from violations
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        raw_class_name = self._resolve_class_name(class_id)
                        class_name = self._canonicalize_class_name(raw_class_name)

                        detection = {
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': class_name,
                            'is_compliant': self.compliance_classes.get(class_name, None)
                        }
                        detections.append(detection)

                        # Separate people and violations for proper association
                        if self._is_person_class(class_name):
                            people_detections.append(detection)
                        elif self._is_violation_class(class_name):
                            violation_detections.append(detection)

                    # Count total people
                    compliance_stats['total_people'] = len(people_detections)

                    # Associate violations with people using spatial proximity
                    people_with_violations = set()

                    for violation in violation_detections:
                        violation_bbox = violation['bbox']
                        violation_center = [
                            (violation_bbox[0] + violation_bbox[2]) / 2,
                            (violation_bbox[1] + violation_bbox[3]) / 2
                        ]

                        # Find closest person to this violation
                        min_distance = float('inf')
                        closest_person_idx = -1

                        for i, person in enumerate(people_detections):
                            person_bbox = person['bbox']
                            person_center = [
                                (person_bbox[0] + person_bbox[2]) / 2,
                                (person_bbox[1] + person_bbox[3]) / 2
                            ]

                            # Calculate distance between centers
                            distance = ((violation_center[0] - person_center[0]) ** 2 +
                                      (violation_center[1] - person_center[1]) ** 2) ** 0.5

                            # Check if violation is within reasonable proximity to person
                            person_width = person_bbox[2] - person_bbox[0]
                            person_height = person_bbox[3] - person_bbox[1]
                            max_distance = max(person_width, person_height) * 1.5  # 1.5x person size

                            if distance < min_distance and distance < max_distance:
                                min_distance = distance
                                closest_person_idx = i

                        # If we found a person close enough, associate the violation
                        if closest_person_idx >= 0:
                            people_with_violations.add(closest_person_idx)
                            compliance_stats['violations'].append({
                                'type': violation['class_name'],
                                'bbox': violation['bbox'],
                                'confidence': violation['confidence'],
                                'associated_person_idx': closest_person_idx
                            })
                        else:
                            # Violation without associated person (still count it)
                            compliance_stats['violations'].append({
                                'type': violation['class_name'],
                                'bbox': violation['bbox'],
                                'confidence': violation['confidence'],
                                'associated_person_idx': -1
                            })

                    # Visibility filter: drop NO-* violations whose body region is not visible
                    VIOL_REGION = {
                        'NO-Gloves': 'hands', 'NO-Goggles': 'head',
                        'NO-Boots': 'feet',   'NO-Hardhat': 'head',
                        'NO-Mask': 'head',    'NO-Safety Vest': 'torso',
                        'NO-PPE': 'torso',
                    }
                    kept = []
                    kept_pidx = set()
                    for v in compliance_stats['violations']:
                        vtype = v.get('type', '')
                        req_region = VIOL_REGION.get(vtype)
                        pidx = v.get('associated_person_idx', -1)
                        if req_region is None:
                            kept.append(v)
                            if pidx >= 0: kept_pidx.add(pidx)
                            continue
                        if 0 <= pidx < len(people_detections):
                            pbbox = people_detections[pidx]['bbox']
                        elif people_detections:
                            pbbox = people_detections[0]['bbox']
                        else:
                            kept.append(v)
                            if pidx >= 0: kept_pidx.add(pidx)
                            continue
                        if req_region in self._estimate_visible_body_regions(pbbox, image.shape):
                            kept.append(v)
                            if pidx >= 0: kept_pidx.add(pidx)
                        # else: body region not visible, drop violation
                    compliance_stats['violations'] = kept

                    # Calculate accurate compliance statistics
                    compliance_stats['people_with_violations'] = len(kept_pidx)

                    # Fallback for models that detect missing PPE but not a separate person box.
                    if compliance_stats['total_people'] == 0 and violation_detections:
                        compliance_stats['total_people'] = len(violation_detections)
                        compliance_stats['people_with_violations'] = len(violation_detections)

                    if compliance_stats['total_people'] > 0:
                        compliance_stats['compliant_people'] = compliance_stats['total_people'] - compliance_stats['people_with_violations']
                        compliance_stats['compliance_rate'] = (compliance_stats['compliant_people'] / compliance_stats['total_people']) * 100
                    else:
                        compliance_stats['compliant_people'] = 0
                        compliance_stats['compliance_rate'] = 100.0  # No people = 100% compliance

            # Perform face recognition if enabled
            face_results = []
            face_stats = {
                'total_faces': 0,
                'recognized_faces': 0,
                'unknown_faces': 0,
                'recognized_people': [],
                'recognition_rate': 0.0
            }

            if self.face_recognition_enabled and self.face_engine:
                try:
                    face_results = self.face_engine.recognize_faces(image)
                    face_stats = self.face_engine.get_recognition_stats(face_results)
                except Exception as e:
                    logging.error(f"Face recognition error: {e}")

            positive_ppe_detections = [
                det for det in detections if det['class_name'] in self.required_ppe_items
            ]
            detected_ppe_names = {det['class_name'] for det in positive_ppe_detections}

            # Fallback: if a face is visible but the model missed "Person", still count a person.
            if compliance_stats['total_people'] == 0 and face_results:
                for face in face_results:
                    pseudo_person_bbox = self._face_to_person_bbox(face['bbox'], image.shape)
                    detections.append({
                        'bbox': pseudo_person_bbox,
                        'confidence': float(face.get('recognition_confidence', 1.0) or 1.0),
                        'class_id': -1,
                        'class_name': 'Person',
                        'is_compliant': None
                    })

                compliance_stats['total_people'] = len(face_results)

            # ── Per-person PPE presence check ────────────────────────────────
            # For every detected person, determine which required PPE items are
            # spatially present (overlap with person bbox) vs missing.
            # This replaces the old global "infer missing" block which was
            # incorrectly marking people compliant when PPE was detected
            # anywhere in the frame (not necessarily on that person).

            person_detections_list = [d for d in detections if self._is_person_class(d['class_name'])]
            # Ensure total_people is always set from actual person detections
            if compliance_stats['total_people'] == 0 and person_detections_list:
                compliance_stats['total_people'] = len(person_detections_list)
            positive_ppe_detections_list = [
                d for d in detections if d['class_name'] in self.required_ppe_items
            ]

            # Also treat 'NO-PPE' (model class 'none') as a full violation
            no_ppe_detections = [
                d for d in detections if d['class_name'] == 'NO-PPE'
            ]

            if compliance_stats['total_people'] > 0 and not compliance_stats['violations']:
                # No explicit NO-* classes detected — do spatial + visibility-aware PPE check
                people_with_violations_set = set()

                for p_idx, person in enumerate(person_detections_list):
                    px1, py1, px2, py2 = person['bbox']
                    p_area = max(1, (px2 - px1) * (py2 - py1))

                    # Determine which body regions are visible for this person
                    visible_regions = self._estimate_visible_body_regions(
                        [px1, py1, px2, py2], image.shape
                    )

                    # Only check PPE items whose body region is visible
                    items_to_check = [
                        item for item in self.required_ppe_items
                        if self.PPE_BODY_REGION.get(item, 'torso') in visible_regions
                    ]

                    # Which required PPE items overlap this person's bbox?
                    present_items = set()
                    for ppe_det in positive_ppe_detections_list:
                        if ppe_det['class_name'] not in items_to_check:
                            continue  # skip PPE not relevant to visible regions
                        ex1, ey1, ex2, ey2 = ppe_det['bbox']
                        ix1 = max(px1, ex1); iy1 = max(py1, ey1)
                        ix2 = min(px2, ex2); iy2 = min(py2, ey2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            ppe_area = max(1, (ex2 - ex1) * (ey2 - ey1))
                            if inter_area / ppe_area >= 0.20 or inter_area / p_area >= 0.10:
                                present_items.add(ppe_det['class_name'])

                    # Check for 'none' / NO-PPE overlapping this person
                    for no_ppe in no_ppe_detections:
                        ex1, ey1, ex2, ey2 = no_ppe['bbox']
                        ix1 = max(px1, ex1); iy1 = max(py1, ey1)
                        ix2 = min(px2, ex2); iy2 = min(py2, ey2)
                        if ix2 > ix1 and iy2 > iy1:
                            present_items.clear()
                            break

                    # Only flag missing items for visible body regions
                    missing_items = [
                        item for item in items_to_check
                        if item not in present_items
                    ]

                    # Items not evaluated (body region not visible)
                    not_evaluated = [
                        item for item in self.required_ppe_items
                        if item not in items_to_check
                    ]

                    if missing_items:
                        people_with_violations_set.add(p_idx)
                        for item in missing_items:
                            compliance_stats['violations'].append({
                                'type': f'Missing {item}',
                                'bbox': person['bbox'],
                                'confidence': 1.0,
                                'associated_person_idx': p_idx,
                                'not_evaluated': not_evaluated
                            })
                    elif not_evaluated:
                        # Person has no violations for visible regions — record skipped items
                        compliance_stats['violations'].append({
                            'type': 'Not Evaluated',
                            'bbox': person['bbox'],
                            'confidence': 1.0,
                            'associated_person_idx': p_idx,
                            'not_evaluated': not_evaluated,
                            'skipped_only': True
                        })

                # Recalculate — only count real violations (not skipped-only entries)
                real_violations = [v for v in compliance_stats['violations']
                                   if not v.get('skipped_only', False)]
                compliance_stats['violations'] = real_violations  # remove skipped-only placeholders
                compliance_stats['people_with_violations'] = len(people_with_violations_set)
                compliance_stats['compliant_people'] = (
                    compliance_stats['total_people'] - compliance_stats['people_with_violations']
                )
                # Compliance based only on items that were actually checked
                total_checked = sum(
                    len([i for i in self.required_ppe_items
                         if self.PPE_BODY_REGION.get(i, 'torso') in
                         self._estimate_visible_body_regions(p['bbox'], image.shape)])
                    for p in person_detections_list
                )
                total_missing = len(real_violations)
                if total_checked > 0:
                    compliance_stats['compliance_rate'] = max(
                        0.0, (1 - total_missing / total_checked) * 100
                    )
                else:
                    compliance_stats['compliance_rate'] = 100.0

            elif compliance_stats['total_people'] > 0:
                # Explicit NO-* violations already found — recalculate stats
                compliance_stats['compliant_people'] = (
                    compliance_stats['total_people'] - compliance_stats['people_with_violations']
                )
                compliance_stats['compliance_rate'] = (
                    compliance_stats['compliant_people'] / compliance_stats['total_people']
                ) * 100

            return {
                'detections': detections,
                'compliance_stats': compliance_stats,
                'face_results': face_results,
                'face_stats': face_stats,
                'image_shape': image.shape
            }
            
        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            if hasattr(image, 'shape'):
                error_msg += f" (Image shape: {image.shape})"
            logging.error(error_msg)
            return {"error": error_msg}
    
    def draw_detections(self, image, detections, show_confidence=True,
                       face_results=None, attendance_info=None,
                       compliance_stats=None):
        """Draw detections with per-person violation labels (red tags) and compliant tag (green)."""
        result_image = image.copy()

        colors = {
            'Hardhat': (0, 255, 0), 'Mask': (0, 255, 0),
            'Safety Vest': (0, 255, 0), 'Gloves': (0, 255, 0), 'Goggles': (0, 255, 0),
            'NO-Hardhat': (0, 0, 255), 'NO-Mask': (0, 0, 255),
            'NO-Safety Vest': (0, 0, 255), 'NO-Gloves': (0, 0, 255), 'NO-Goggles': (0, 0, 255),
            'Person': (255, 255, 0), 'Safety Cone': (255, 165, 0),
            'machinery': (128, 0, 128), 'vehicle': (255, 192, 203)
        }
        violation_labels = {
            'NO-Hardhat': 'Helmet Missing', 'NO-Mask': 'Mask Missing',
            'NO-Safety Vest': 'Safety Jacket Missing', 'NO-Gloves': 'Gloves Missing',
            'NO-Goggles': 'Goggles Missing', 'Missing Hardhat': 'Helmet Missing',
            'Missing Mask': 'Mask Missing', 'Missing Safety Vest': 'Safety Jacket Missing',
            'Missing Gloves': 'Gloves Missing', 'Missing Goggles': 'Goggles Missing',
            'Missing Required PPE': 'PPE Missing',
        }

        # Build per-person violation map
        person_violations = {}
        if compliance_stats and 'violations' in compliance_stats:
            for v in compliance_stats['violations']:
                pidx = v.get('associated_person_idx', -1)
                lbl = violation_labels.get(v.get('type', ''), v.get('type', ''))
                person_violations.setdefault(pidx, []).append(lbl)

        person_idx = 0
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            color = colors.get(class_name, (128, 128, 128))
            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(result_image, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), 1)

            label = class_name + (f" {confidence:.2f}" if show_confidence else "")
            lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            lby1 = max(0, y1 - lsz[1] - 12)
            cv2.rectangle(result_image, (x1+1, lby1+1), (x1+lsz[0]+9, y1+1), (0,0,0), -1)
            cv2.rectangle(result_image, (x1, lby1), (x1+lsz[0]+8, y1), color, -1)
            cv2.putText(result_image, label, (x1+4, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Per-person violation / compliant tags
            if self._is_person_class(class_name):
                viols = person_violations.get(person_idx, [])
                tag_y = lby1 - 4
                if viols:
                    for vt in viols:
                        tag_txt = f"X {vt}"
                        tsz = cv2.getTextSize(tag_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
                        tag_y -= tsz[1] + 6
                        if tag_y < 0:
                            break
                        cv2.rectangle(result_image, (x1, tag_y),
                                      (x1+tsz[0]+8, tag_y+tsz[1]+6), (0,0,200), -1)
                        cv2.putText(result_image, tag_txt, (x1+4, tag_y+tsz[1]+2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)
                else:
                    ok_txt = "OK Compliant"
                    tsz = cv2.getTextSize(ok_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
                    tag_y -= tsz[1] + 6
                    if tag_y >= 0:
                        cv2.rectangle(result_image, (x1, tag_y),
                                      (x1+tsz[0]+8, tag_y+tsz[1]+6), (0,180,0), -1)
                        cv2.putText(result_image, ok_txt, (x1+4, tag_y+tsz[1]+2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)
                person_idx += 1

        if face_results and self.face_recognition_enabled and self.face_engine:
            result_image = self.face_engine.draw_face_detections(
                result_image, face_results, show_confidence, attendance_info)
        return result_image


    def get_face_engine(self) -> Optional[FaceRecognitionEngine]:
        """Get the face recognition engine instance

        Returns:
            Face recognition engine instance or None if not available
        """
        return self.face_engine if self.face_recognition_enabled else None

    def is_face_recognition_enabled(self) -> bool:
        """Check if face recognition is enabled and available

        Returns:
            True if face recognition is enabled and working, False otherwise
        """
        return self.face_recognition_enabled and self.face_engine is not None

    def process_video(self, video_path: str, output_path: str,
                     conf_threshold: float = 0.5, iou_threshold: float = 0.45,
                     progress_callback=None, skip_frames: int = 1,
                     max_resolution: int = 1280, stop_flag=None,
                     persistent_overlay: bool = True) -> Dict:
        """Process video file for PPE detection with optimizations and cancellation

        Args:
            video_path: Path to input video
            output_path: Path for output video
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            progress_callback: Callback function for progress updates
            skip_frames: Process every Nth frame (1 = all frames)
            max_resolution: Maximum resolution for processing (width)
            stop_flag: Threading event to stop processing

        Returns:
            Dict containing processing results and statistics
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"error": "Could not open video file"}

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate optimal resolution for processing
        if orig_width > max_resolution:
            scale_factor = max_resolution / orig_width
            process_width = max_resolution
            process_height = int(orig_height * scale_factor)
        else:
            scale_factor = 1.0
            process_width = orig_width
            process_height = orig_height

        # Setup video writer with original resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

        # Initialize statistics
        video_stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'skipped_frames': 0,
            'total_violations': 0,
            'frame_violations': [],
            'compliance_timeline': [],
            'processing_fps': 0,
            'ppe_violation_counts': {},   # e.g. {'NO-Hardhat': 3, 'NO-Mask': 2}
            'person_violation_log': []    # per-frame per-person detail
        }

        frame_count = 0
        processed_count = 0
        start_time = time.time()
        last_progress_update = start_time

        # For persistent overlay - store last detection results
        last_detections = []
        last_detection_frame = -1
        # Keep detections visible for multiple frames to prevent flickering
        detection_persistence = skip_frames * DEFAULT_DETECTION_PERSISTENCE if persistent_overlay else 0

        try:
            while True:
                # Check for stop signal
                if stop_flag and stop_flag.is_set():
                    video_stats['cancelled'] = True
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Determine which detections to use for this frame
                current_detections = []

                # Process every skip_frames frame for consistent speed
                should_process = (frame_count % skip_frames == 0)

                if should_process:
                    # Resize frame for processing if needed
                    if scale_factor != 1.0:
                        process_frame = cv2.resize(frame, (process_width, process_height))
                    else:
                        process_frame = frame

                    # Detect objects in frame
                    results = self.detect_objects(process_frame, conf_threshold, iou_threshold)

                    if 'error' not in results:
                        # Scale detection results back to original resolution
                        if scale_factor != 1.0:
                            scaled_detections = []
                            for det in results['detections']:
                                scaled_det = det.copy()
                                bbox = det['bbox']
                                scaled_det['bbox'] = [
                                    bbox[0] / scale_factor,
                                    bbox[1] / scale_factor,
                                    bbox[2] / scale_factor,
                                    bbox[3] / scale_factor
                                ]
                                scaled_detections.append(scaled_det)
                            results['detections'] = scaled_detections

                        # Store detections for persistence
                        last_detections = results['detections'].copy()
                        last_detection_frame = frame_count
                        current_detections = results['detections']

                        # Update statistics with improved accuracy
                        compliance_stats = results['compliance_stats']
                        # Count people with violations instead of total violations for better accuracy
                        video_stats['total_violations'] += compliance_stats['people_with_violations']
                        video_stats['compliance_timeline'].append(compliance_stats['compliance_rate'])

                        if compliance_stats['people_with_violations'] > 0:
                            frame_entry = {
                                'frame': frame_count,
                                'timestamp': round(frame_count / fps, 2),
                                'violations': compliance_stats['violations'],
                                'people_with_violations': compliance_stats['people_with_violations']
                            }
                            video_stats['frame_violations'].append(frame_entry)

                            # Accumulate per-PPE-type violation counts
                            for v in compliance_stats['violations']:
                                vtype = v.get('type', 'Unknown')
                                video_stats['ppe_violation_counts'][vtype] = (
                                    video_stats['ppe_violation_counts'].get(vtype, 0) + 1
                                )

                            # Build per-person violation log for this frame
                            person_map = {}
                            for v in compliance_stats['violations']:
                                pidx = v.get('associated_person_idx', -1)
                                person_map.setdefault(pidx, []).append(v.get('type', 'Unknown'))
                            for pidx, vtypes in person_map.items():
                                video_stats['person_violation_log'].append({
                                    'frame': frame_count,
                                    'timestamp': round(frame_count / fps, 2),
                                    'person_idx': pidx,
                                    'violations': vtypes
                                })

                        processed_count += 1
                    else:
                        # Use last detections if available for error frames
                        if persistent_overlay and last_detections and (frame_count - last_detection_frame) <= detection_persistence:
                            current_detections = last_detections
                else:
                    # Skip processing, but use persistent detections if enabled
                    video_stats['skipped_frames'] += 1
                    if persistent_overlay and last_detections and (frame_count - last_detection_frame) <= detection_persistence:
                        current_detections = last_detections

                # Draw detections on frame (either new or persistent)
                if current_detections:
                    processed_frame = self.draw_detections(frame, current_detections, compliance_stats=results.get('compliance_stats') if should_process and 'error' not in results else None)
                else:
                    processed_frame = frame

                # Write frame
                out.write(processed_frame)
                frame_count += 1
                video_stats['processed_frames'] = frame_count

                # Update progress more frequently for smoother display
                current_time = time.time()
                if progress_callback and (current_time - last_progress_update) >= PROGRESS_UPDATE_INTERVAL:
                    progress = (frame_count / total_frames) * 100
                    elapsed_time = current_time - start_time
                    if elapsed_time > 0:
                        # Calculate more accurate FPS
                        video_stats['processing_fps'] = frame_count / elapsed_time

                    progress_callback(progress)
                    last_progress_update = current_time

        finally:
            cap.release()
            out.release()

        # Calculate overall statistics
        if video_stats['compliance_timeline']:
            video_stats['average_compliance_rate'] = np.mean(video_stats['compliance_timeline'])
        else:
            video_stats['average_compliance_rate'] = 0.0

        # Add performance metrics
        total_time = time.time() - start_time
        video_stats['processing_time'] = total_time
        video_stats['final_fps'] = frame_count / total_time if total_time > 0 else 0
        video_stats['detection_frames'] = processed_count

        return video_stats
