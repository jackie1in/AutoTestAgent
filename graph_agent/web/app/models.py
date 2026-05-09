from pydantic import BaseModel


class PlaybackRequest(BaseModel):
    intent: str
    test_data: dict = {}
    wait_for_network: bool = True


class NLResolveRequest(BaseModel):
    query: str
    top_k: int = 3


class NLPlaybackRequest(BaseModel):
    query: str
    test_data: dict = {}
    wait_for_network: bool = True


class KnowledgeSettingsRequest(BaseModel):
    trigger_profile: str
