import cv2
import numpy as np
from PIL import Image
from skimage import transform
import torch
from torchvision.ops import batched_nms

from src.face_detect.src.models.box_utils import get_image_boxes, calibrate_box, nms
from src.face_detect.src.models.get_nets import ONet
from src.face_detect.src.models.retinaface import RetinaFace, load_model, PriorBox, decode, decode_landm, py_cpu_nms, \
    face_bigger


class RetinaFaceExtract():
    def __init__(self, cfg):
        # init the parameters
        self.keep_size = cfg["keep_size"]
        self.target_size = cfg["target_size"]
        self.top_k = 5000
        self.keep_top_k = 10

        self.confidence_threshold = 0.9
        self.nms_threshold = 0.4
        self.cfg = cfg

        # set device
        device = cfg['device']
        if torch.cuda.is_available() and device == "cuda":
            self.device = torch.device("cuda:{}".format(cfg["gpu_id"]))
            # load_to_cpu = False
        else:
            self.device = torch.device("cpu")
            # load_to_cpu = True

        # init model & load weights
        print('Creating RetinaFace')
        self.net = RetinaFace(cfg=self.cfg)
        self.net = load_model(self.net, self.cfg['weights_path'], load_to_cpu=True)
        self.net = self.net.to(self.device)
        self.net.eval()
        print('RetinaFace created on {}!'.format(self.device))

        # for face align
        self.trans = transform.SimilarityTransform()
        self.REFERENCE_FACIAL_POINTS = [
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 87],
            [62.72990036, 87]
        ]
        self.DEFAULT_CROP_SIZE = (96, 112)
        self.ref_pts = self._get_reference_facial_points()
        self.ref_pts_for_ps_detect = self._get_reference_facial_points_for_ps_detect()
        # init onet model & load weights
        self.onet = ONet(self.cfg['onet_weights_path'])
        self.onet.eval()

    def _get_reference_facial_points(self, output_size=(112, 112)):
        tmp_5pts = np.array(self.REFERENCE_FACIAL_POINTS)
        tmp_crop_size = np.array(self.DEFAULT_CROP_SIZE)
        x_scale = output_size[0] / tmp_crop_size[0]
        y_scale = output_size[1] / tmp_crop_size[1]
        tmp_5pts[:, 0] *= x_scale
        tmp_5pts[:, 1] *= y_scale
        return tmp_5pts

    def _get_reference_facial_points_for_ps_detect(self, output_size=(400, 400)):
        tmp_5pts = np.array(self.REFERENCE_FACIAL_POINTS) * 3  # (96, 112)->(240, 280)
        tmp_crop_size = np.array(self.DEFAULT_CROP_SIZE) * 3
        center_point_tmp = tmp_crop_size // 2
        center_point = np.array(output_size) // 2
        offset = (center_point - center_point_tmp) // 2  # (w, h)
        tmp_5pts[:, 0] += offset[0]
        tmp_5pts[:, 0] += offset[1]
        return tmp_5pts

    def _make_dict(self, det):
        return {
            'box': [int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])],
            'confidence': float(det[4]),
            'keypoints': {
                'left_eye': [int(det[5]), int(det[6])],
                'right_eye': [int(det[7]), int(det[8])],
                'nose': [int(det[9]), int(det[10])],
                'mouth_left': [int(det[11]), int(det[12])],
                'mouth_right': [int(det[13]), int(det[14])],
            }
        }

    def run_data_batch(self, img_raw_batch):
        # img_raw_batch: ndarray, (n, h, w, 3)
        img_batch = np.float32(img_raw_batch)

        # get resize scale
        if self.keep_size:
            resize = 1
        else:
            _, height, width, _ = img_batch.shape
            size_max = max(height, width)
            resize = float(self.target_size) / float(size_max)

        # resize images
        if resize != 1:
            img_batch_resized = []
            for img in img_batch:
                img_resized = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                img_batch_resized.append(img_resized)
            img_batch_resized = np.array(img_batch_resized)
        else:
            img_batch_resized = img_batch

        _, height, width, _ = img_batch_resized.shape
        img_batch_resized -= (104, 117, 123)
        img_batch_resized = torch.from_numpy(img_batch_resized).permute(0, 3, 1, 2)
        img_batch_resized = img_batch_resized.to(self.device)

        scale = torch.Tensor([width, height]).repeat(2)
        scale = scale.to(self.device)
        scale1 = torch.Tensor([width, height]).repeat(5)
        scale1 = scale1.to(self.device)

        priors = PriorBox(self.cfg, image_size=(height, width))
        priors = priors.to(self.device)

        with torch.no_grad():
            batch_loc, batch_conf, batch_landms = self.net(img_batch_resized)  # forward pass

        n_images, n_boxes, _ = batch_loc.shape
        batch_idx = torch.arange(n_images).repeat_interleave(n_boxes).to(self.device)

        batch_scores = batch_conf[:, :, 1]
        batch_scores = batch_scores.reshape((-1))

        batch_boxes = decode(batch_loc, priors, self.cfg['variance'])
        batch_boxes = batch_boxes * scale / resize
        batch_boxes = batch_boxes.reshape(n_images * n_boxes, 4)

        batch_landms = decode_landm(batch_landms, priors, self.cfg['variance'])
        batch_landms = batch_landms * scale1 / resize
        batch_landms = batch_landms.reshape(n_images * n_boxes, 10)

        # ignore low scores
        mask_threshold = batch_scores.ge(self.confidence_threshold)
        batch_boxes = batch_boxes[mask_threshold]
        batch_idx = batch_idx[mask_threshold]
        batch_landms = batch_landms[mask_threshold]
        batch_scores = batch_scores[mask_threshold]

        # do NMS
        keep = batched_nms(batch_boxes, batch_scores, batch_idx, self.nms_threshold)
        batch_boxes = batch_boxes[keep]
        batch_idx = batch_idx[keep]
        batch_landms = batch_landms[keep]
        batch_scores = batch_scores[keep]

        batch_detections = []
        for idx in range(n_images):
            mask_current_image = batch_idx.eq(idx)
            # keep top-k
            scores = batch_scores[mask_current_image]
            order = scores.argsort(dim=0, descending=True)[:self.keep_top_k]
            scores = scores[order].data.cpu().numpy()
            boxes = batch_boxes[mask_current_image][order].data.cpu().numpy()
            landms = batch_landms[mask_current_image][order].data.cpu().numpy()
            batch_detections.append([boxes, landms, scores[:, np.newaxis]])
        return batch_detections

    def face_datas_for_EFFModel(self, img_raw, bbox, landms, scores):
        # bbox: (num, 4)
        # landms: (num, 10)
        # scores: (num, 1)
        dets = np.concatenate((bbox.astype(np.int), scores, landms.astype(np.int)), axis=1)
        count = 0
        faces = []
        landmarks = []
        ori_coordinates = []
        coordinates = []
        scores_all = []

        im_shape = img_raw.shape
        for det in dets:
            # expand bbox
            b = face_bigger(im_shape[0], im_shape[1], det)
            # bbox boundary processing
            width = max(b[3] - b[1], b[2] - b[0])
            xmin = max(0, b[0])
            ymin = max(0, b[1])
            xmax = min(b[0] + width, im_shape[1])
            ymax = min(b[1] + width, im_shape[0])
            ori_xmin = max(xmin, det[0])
            ori_ymin = max(ymin, det[1])
            ori_xmax = min(det[2], xmax)
            ori_ymax = min(det[3], ymax)

            coordinates.append([xmin, ymin, xmax, ymax])
            face = img_raw[int(ymin):int(ymax), int(xmin):int(xmax)]
            faces.append(face)
            scores_all.append(det[4])
            landmarks.append(list(det[5:]))
            ori_coordinates.append([ori_xmin, ori_ymin, ori_xmax, ori_ymax])
            count += 1
        return faces, np.array(scores_all), np.array(ori_coordinates), np.array(coordinates), np.array(landmarks)

    def inference(self, img):
        # img: shape:(h, w, 3)
        # faces, ori_coordinates, coordinates, bbox, landms, scores_all: ordered from larger score to small score
        bbox, landms, scores = self.run_data_batch(np.expand_dims(img, 0))[0]
        faces, scores_all, ori_coordinates, coordinates, landmarks = self.face_datas_for_EFFModel(img, bbox, landms,
                                                                                                  scores)
        return faces, ori_coordinates, coordinates, bbox, landms, scores_all

    def inference_batch(self, img_batch):
        # img_batch: shape:(batch, h, w, 3)
        # res is ordered by images' order
        # faces, ori_coordinates, coordinates, bbox, landms, scores_all: ordered from larger score to small score
        res = []
        for i, (bbox, landms, scores) in enumerate(self.run_data_batch(img_batch)):
            faces, scores_all, ori_coordinates, coordinates, landmarks = self.face_datas_for_EFFModel(img_batch[i],
                                                                                                      bbox, landms,
                                                                                                      scores)
            res.append([faces, ori_coordinates, coordinates, bbox, landms, scores_all])
        return res

    def face_tracker(self, img_raw, bounding_box_pre):
        # input: img BGR
        # bounding_box_pre: ndarray (n, 5) , bbox+score
        # get resize scale
        if self.keep_size:
            resize = 1
        else:
            height, width, _ = img_raw.shape
            size_max = max(height, width)
            # target_size = 320
            # resize = float(self.target_size) / float(size_max)
            resize = float(self.cfg['target_size_tracker']) / float(size_max)
        # resize images
        if resize != 1:
            img = cv2.resize(img_raw, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        else:
            img = img_raw

        bounding_box_pre[:, 0:4] = np.round(bounding_box_pre[:, 0:4] * resize)
        im = Image.fromarray(img[:, :, ::-1])  # RGB
        img_boxes = get_image_boxes(bounding_box_pre, im, size=48)
        if len(img_boxes) == 0:
            return [], []
        with torch.no_grad():
            img_boxes = torch.FloatTensor(img_boxes)
            output = self.onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.cfg['onet_threshold'])[0]
        bounding_boxes = bounding_box_pre[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, self.cfg['onet_threshold'], mode='min')
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, :4] = bounding_boxes[:, :4] / resize
        landmarks = landmarks[keep] / resize
        out_landmks = np.zeros_like(landmarks)
        out_landmks[:, ::2], out_landmks[:, 1::2] = landmarks[:, :5], landmarks[:, 5:]
        return bounding_boxes, out_landmks

    def draw(self, img, landmarks, save_path):
        landms = landmarks.astype(np.int32)
        for i in range(5):
            cv2.circle(img, (landms[2 * i], landms[2 * i + 1]), 1, (0, 0, 255), 4)
        cv2.imwrite(save_path, img)

    def detect_align(self, img, landmarks, output_size=(112, 112)):
        aligned_faces = []
        for src_pts in landmarks:
            self.trans.estimate(src_pts.reshape((5, 2)), self.ref_pts)
            face_img = cv2.warpAffine(img, self.trans.params[0:2, :], output_size)
            aligned_faces.append(face_img)
        return aligned_faces

    def detect_align_for_ps_detect(self, img, landmarks, output_size=(400, 400)):
        aligned_faces = []
        aligned_landmarks = []
        for src_pts in landmarks:
            aligned_pts = np.zeros((5, 2))
            self.trans.estimate(src_pts.reshape((5, 2)), self.ref_pts_for_ps_detect)
            face_img = cv2.warpAffine(img, self.trans.params[0:2, :], output_size)
            aligned_faces.append(face_img)
            M = self.trans.params[0:2, :]
            aligned_pts[:, 0] = src_pts.reshape((5, 2))[:, 0] * M[0, 0] + \
                                src_pts.reshape((5, 2))[:, 1] * M[0, 1] + \
                                M[0, 2]
            aligned_pts[:, 1] = src_pts.reshape((5, 2))[:, 0] * M[1, 0] + \
                                src_pts.reshape((5, 2))[:, 1] * M[1, 1] + \
                                M[1, 2]
            aligned_landmarks.append(aligned_pts.reshape(-1))
        return aligned_faces, aligned_landmarks
