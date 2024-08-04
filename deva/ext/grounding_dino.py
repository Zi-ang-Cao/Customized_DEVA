# Reference: https://github.com/IDEA-Research/Grounded-Segment-Anything
import os

from typing import Dict, List
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torchvision

try:
    from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
    # not sure why this happens sometimes
    from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel

from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor
from deva.ext.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
import numpy as np
import torch

from deva.inference.object_info import ObjectInfo



def get_parent_folder_of_package(package_name):
    # Import the package
    package = __import__(package_name)

    # Get the absolute path of the imported package
    package_path = os.path.abspath(package.__file__)

    # Get the directory of the package
    package_dir = os.path.dirname(package_path)

    # Get the parent directory
    parent_dir = os.path.dirname(package_dir)

    return parent_dir

PARENT_FOLDER = get_parent_folder_of_package('deva')

def get_grounding_dino_model(config: Dict, device: str) -> (GroundingDINOModel, SamPredictor):
    try:
        GROUNDING_DINO_CONFIG_PATH = config['GROUNDING_DINO_CONFIG_PATH']
        GROUNDING_DINO_CHECKPOINT_PATH = config['GROUNDING_DINO_CHECKPOINT_PATH']
        gd_model = GroundingDINOModel(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                  model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
                                  device=device)
    except:
        GROUNDING_DINO_CONFIG_PATH = os.path.join(PARENT_FOLDER, config['GROUNDING_DINO_CONFIG_PATH'])
        GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(PARENT_FOLDER, config['GROUNDING_DINO_CHECKPOINT_PATH'])
        gd_model = GroundingDINOModel(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                  model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
                                  device=device)


    # Building SAM Model and SAM Predictor
    variant = config['sam_variant'].lower()
    if variant == 'mobile':
        try:
            MOBILE_SAM_CHECKPOINT_PATH = config['MOBILE_SAM_CHECKPOINT_PATH']
            checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        except:
            MOBILE_SAM_CHECKPOINT_PATH = os.path.join(PARENT_FOLDER, config['MOBILE_SAM_CHECKPOINT_PATH'])
            checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)

        # Building Mobile SAM model
        mobile_sam = setup_mobile_sam()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        sam = SamPredictor(mobile_sam)
    elif variant == 'original':
        SAM_ENCODER_VERSION = config['SAM_ENCODER_VERSION']
        try:
            SAM_CHECKPOINT_PATH = config['SAM_CHECKPOINT_PATH']
            sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
                device=device)
        except:
            SAM_CHECKPOINT_PATH = os.path.join(PARENT_FOLDER, config['SAM_CHECKPOINT_PATH'])
            sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
                device=device)
        
        sam = SamPredictor(sam)
    

    elif variant == 'hq':
        print("Use HQ SAM!!!!!!! ---------------")
        SAM_ENCODER_VERSION = config['SAM_ENCODER_VERSION']
        # from segment_anything_hq import sam_model_registry_hq
        assert config['SAM_ENCODER_VERSION'] == "vit_h"
        # HQSAM_CHECKPOINT_PATH = config['HQSAM_CHECKPOINT_PATH']
        SAM_CHECKPOINT_PATH = "/juno/u/ziangcao/Juno_CodeBase/IPRL_codeBase/Vision_Pipeline/mm-lfd/mm_lfd/Vision_module/utils/Customized_DEVA/saves/sam_hq_vit_h.pth"
        
        sam = sam_hq_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
                device=device)
        
        sam = SamPredictor(sam)
        # exit("HQ is not supported yet")

    elif variant == 'sam_fast':
        # from segment_anything_fast import sam_model_fast_registry, SamAutomaticMaskGenerator, SamPredictor
        from segment_anything_fast import sam_model_fast_registry, SamAutomaticMaskGenerator


        # NOT USEFULL AT ALL!!!
        # import torch._dynamo
        # torch._dynamo.config.suppress_errors = True
        
        assert config['SAM_ENCODER_VERSION'] == "vit_h"

        SAM_ENCODER_VERSION = config['SAM_ENCODER_VERSION']
        try:
            SAM_CHECKPOINT_PATH = config['SAM_CHECKPOINT_PATH']
            # Change to sam_model_fast_registry
            sam = sam_model_fast_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
                device=device)
        except:
            SAM_CHECKPOINT_PATH = os.path.join(PARENT_FOLDER, config['SAM_CHECKPOINT_PATH'])
            # Change to sam_model_fast_registry
            sam = sam_model_fast_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
                device=device)
        
        sam = SamPredictor(sam)

    return gd_model, sam


def segment_with_text(config: Dict, gd_model: GroundingDINOModel, sam: SamPredictor,
                      image: np.ndarray, prompts: List[str],
                      min_side: int) -> (torch.Tensor, List[ObjectInfo]):
    """
    config: the global configuration dictionary
    image: the image to segment; should be a numpy array; H*W*3; unnormalized (0~255)
    prompts: list of class names

    Returns: a torch index mask of the same size as image; H*W
             a list of segment info, see object_utils.py for definition
    """

    BOX_THRESHOLD = TEXT_THRESHOLD = config['DINO_THRESHOLD']
    NMS_THRESHOLD = config['DINO_NMS_THRESHOLD']

    sam.set_image(image, image_format='RGB')

    # detect objects
    # GroundingDINO uses BGR
    detections = gd_model.predict_with_classes(image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                               classes=prompts,
                                               box_threshold=BOX_THRESHOLD,
                                               text_threshold=TEXT_THRESHOLD)

    ######## NOTE: This is a hack to remove the detections that exceed the 90% of image size ########
    # Remove the detections that exceed the 90% of image size
    detections = detections[detections.area < 0.9 * image.shape[0] * image.shape[1]]


    nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy),
                                  torch.from_numpy(detections.confidence),
                                  NMS_THRESHOLD).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    result_masks = []
    for box in detections.xyxy:
        masks, scores, _ = sam.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])

    detections.mask = np.array(result_masks)    # list of masks, [(H, W), (H, W), ...)]

    h, w = image.shape[:2]
    if min_side > 0:
        scale = min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=gd_model.device)
    curr_id = 1
    segments_info = []

    # sort by descending area to preserve the smallest object
    for i in np.flip(np.argsort(detections.area)):
        mask = detections.mask[i]
        confidence = detections.confidence[i]
        class_id = detections.class_id[i]
        mask = torch.from_numpy(mask.astype(np.float32))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (new_h, new_w), mode='bilinear')[0, 0]
        mask = (mask > 0.5).float()

        if mask.sum() > 0:
            output_mask[mask > 0] = curr_id
            segments_info.append(ObjectInfo(id=curr_id, category_id=class_id, score=confidence))
            curr_id += 1

    return output_mask, segments_info
