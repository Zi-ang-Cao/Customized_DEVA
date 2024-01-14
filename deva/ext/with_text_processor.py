from os import path
from typing import Dict, List

import cv2
import torch
import numpy as np

from deva.inference.object_info import ObjectInfo
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.frame_utils import FrameInfo
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.ext.grounding_dino import segment_with_text
try:
    from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
    # not sure why this happens sometimes
    from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import SamPredictor

import torch.nn.functional as F


def make_segmentation_with_text(cfg: Dict, image_np: np.ndarray, gd_model: GroundingDINOModel,
                                sam_model: SamPredictor, prompts: List[str],
                                min_side: int) -> (torch.Tensor, List[ObjectInfo]):
    mask, segments_info = segment_with_text(cfg, gd_model, sam_model, image_np, prompts, min_side)
    return mask, segments_info


@torch.inference_mode()
def process_frame_with_text(deva: DEVAInferenceCore,
                            gd_model: GroundingDINOModel,
                            sam_model: SamPredictor,
                            frame_path: str,
                            result_saver: ResultSaver,
                            ti: int,
                            image_np: np.ndarray = None) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config
    raw_prompt = cfg['prompt']
    prompts = raw_prompt.split('.')

    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })


    parsed_img = None


    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            mask, segments_info = make_segmentation_with_text(cfg, image_np, gd_model, sam_model,
                                                              prompts, new_min_side)
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            deva.add_to_temporary_buffer(frame_info)

            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np

                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = deva.incorporate_detection(this_image, mask, new_segments_info)
                deva.next_voting_frame += cfg['detection_every']

                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np,
                                       prompts=prompts)

                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np,
                                           prompts=prompts)

                deva.clear_buffer()
        else:
            # standard propagation
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np,
                                   prompts=prompts)

    elif cfg['temporal_setting'] == 'online':
        # FOR MM_LFD, USE THIS Branch ONLY!!!!

        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            mask, segments_info, parsed_img = make_segmentation_with_text(cfg, image_np, gd_model, sam_model,
                                                              prompts, new_min_side)
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info)
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)
        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np,
                               prompts=prompts)
        

        return parsed_img





@torch.inference_mode()
def refined_processor_with_text(deva: DEVAInferenceCore,
                            gd_model: GroundingDINOModel,
                            sam_model: SamPredictor,
                            ti: int,
                            image_np: np.ndarray = None,
                            obj_ids = None,
                            ) -> None:
    
    # image_np, if given, should be in RGB
    assert image_np is not None


    cfg = deva.config
    raw_prompt = cfg['prompt']
    prompts = raw_prompt.split('.')

    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    frame_info = FrameInfo(image, None, None, ti, {
        'frame': ["refined_processor_with_text_NULL_FRAME_NAME"],
        'shape': [h, w],
    })


    if cfg['temporal_setting'] == 'semionline':
        raise NotImplementedError

    elif cfg['temporal_setting'] == 'online':
        # FOR MM_LFD, USE THIS Branch ONLY!!!!

        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            mask, segments_info = make_segmentation_with_text(cfg, image_np, gd_model, sam_model,
                                                              prompts, new_min_side)
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info)

            # seg = parsed_img
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)

        shape=(h, w)

        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,
                                                                                                0]
        
        # Probability mask -> index mask
        index_mask = torch.argmax(prob, dim=0)    # (H, W), either 0 or 1 - n (for n objects)
        index_mask = index_mask.cpu().numpy()
        
        if obj_ids is None:
            # cared_prompt_ids = list(range(len(prompts)))
            obj_ids = np.unique(index_mask)[1:]    # 0 is the background
            
        seg = np.isin(index_mask, obj_ids).astype(np.uint8) * 255
        seg = np.repeat(seg[:, :, None], 3, axis=-1)
        
        return seg
