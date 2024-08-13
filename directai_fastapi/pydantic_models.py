import uuid
import json
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import redis.asyncio as redis

class DeployResponse(BaseModel):
    deployed_id: str
    message: str

class HTTPExceptionResponse(BaseModel):
    status_code: int
    message: str
    data: Optional[str] = None

class SingleClassifierClass(BaseModel):
    name: str
    examples_to_include: List[str]
    examples_to_exclude: List[str] = []
    
    class Config:
        orm_mode = True

class ClassifierDeploy(BaseModel):
    classifier_configs: List[SingleClassifierClass]
    deployed_id: Optional[str] = None
    augment_examples: Optional[bool] = True
    
    class Config:
        orm_mode = True
    
    async def save_configuration(self, config_cache: redis.Redis) -> dict:
        print(f"Classifier Configs: {self.classifier_configs}")
        for classifier_config in self.classifier_configs:
            print(classifier_config.examples_to_include)
            if len(classifier_config.examples_to_include) == 0:
                raise HTTPException(
                    status_code = 422,
                    detail = f"Model lacks example_to_include for {classifier_config.name} class."
                )
        labels = [c.name for c in self.classifier_configs]
        inc_sub_labels_dict: dict[str, List[str]] = {c.name:c.examples_to_include for c in self.classifier_configs}
        exc_sub_labels_dict: dict[str, List[str]] = {c.name:c.examples_to_exclude for c in self.classifier_configs}
        controls: List = []
        config_dict = {
            "labels": labels,
            "inc_sub_labels_dict": inc_sub_labels_dict,
            "exc_sub_labels_dict": exc_sub_labels_dict,
            "controls": controls,
            "augment_examples": self.augment_examples
        }
        
        if self.deployed_id is not None:
            key_exists = await config_cache.exists(self.deployed_id)
        else:
            key_exists = False
            
        if not key_exists:
            if self.deployed_id is not None:
                message = f"{self.deployed_id} model not found. Generated new model." 
            else:
                message = "New model deployed." 
            self.deployed_id = str(uuid.uuid4())
        else:
            message = "Model updated."
        
        assert self.deployed_id is not None, "deployed_id should not be None at this point"
        await config_cache.set(self.deployed_id,json.dumps(config_dict))
        return {
            'deployed_id': self.deployed_id,
            'message': message
        }

class ClassifierResponse(BaseModel):
    scores: Dict[str, float]
    pred: str
    raw_scores: Dict[str, float]

class SingleDetectorClass(BaseModel):
    name: str
    examples_to_include: List[str]
    examples_to_exclude: List[str] = []
    detection_threshold: float = 0.1

class DetectorDeploy(BaseModel):
    detector_configs: List[SingleDetectorClass]
    nms_threshold: float = 0.4
    class_agnostic_nms: Optional[bool] = True
    deployed_id: Optional[str] = None
    augment_examples: Optional[bool] = True
    
    class Config:
        orm_mode = True
    
    async def save_configuration(self, config_cache: redis.Redis) -> dict:
        for detector_config in self.detector_configs:
            if len(detector_config.examples_to_include) == 0:
                raise HTTPException(
                    status_code = 422,
                    detail = f"Model lacks example_to_include for {detector_config.name} class."
                )
        # Translating into Backend
        config_dict = self.dict()
        for i, single_config in enumerate(config_dict['detector_configs']):
            single_config['incs'] = single_config['examples_to_include']
            single_config['excs'] = single_config['examples_to_exclude']
            single_config['img_incs'] = []
            single_config['img_excs'] = []
            single_config['thresh'] = single_config['detection_threshold']
            del single_config['examples_to_include']
            del single_config['examples_to_exclude']
            del single_config['detection_threshold']
            config_dict['detector_configs'][i] = single_config
        config_dict['nms_thresh'] = config_dict['nms_threshold']
        del config_dict['nms_threshold']
        config_dict['augment_examples'] = config_dict.get('augment_examples', True)
        config_dict['class_agnostic_nms'] = config_dict.get('class_agnostic_nms', True)
        
        if self.deployed_id is not None:
            key_exists = await config_cache.exists(self.deployed_id)
        else:
            key_exists = False
        
        if not key_exists:
            if self.deployed_id is not None:
                message = f"{self.deployed_id} model not found. Generated new model." 
            else:
                message = "New model deployed."
            self.deployed_id = str(uuid.uuid4())
        else:
            message = "Model updated."
        assert self.deployed_id is not None, "deployed_id should not be None at this point"
        await config_cache.set(self.deployed_id, json.dumps(config_dict))
        return {
            'deployed_id': self.deployed_id,
            'message': message
        }


