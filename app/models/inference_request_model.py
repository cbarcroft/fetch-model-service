from pydantic import BaseModel

class InferenceRequestModel(BaseModel):
   input: str