from sagemaker.huggingface import HuggingFaceModel

# IAM 역할과 모델의 이름(Hugging Face Hub의 모델 경로)
role = 'arn:aws:iam::211125418324:role/SagemakerFullAccess'
hub = {'HF_MODEL_ID':'kreimben/CodeMind-gemma', 'HF_TASK':'text-generation'}

# 모델 객체 생성
huggingface_model = HuggingFaceModel(
    transformers_version='4.39.3',
    pytorch_version='2.2.2',
    py_version='py311',
    env=hub,
    role=role
)

# 모델 배포
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
