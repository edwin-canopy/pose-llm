


class PoseSpeechUserTextCollator:
    """
    Models data for training pretrained base pose llm with:
    - user text input (non-streaming)
    - outputs interleaved assistant text, audio and pose
    """
    pass


class PoseSpeechUserAudioCollator:
    """
    Models data for training pretrained base pose llm with:
    - user raw audio input (non-streaming)
    - outputs interleaved assistant text, audio and pose
    """
    pass


class PoseStreamingUserAudioCollator:
    """
    Models data for training pretrained base pose llm in streaming mode:
    - models interleaved user audio (raw mels), user transcript, assistant transcript, audio and pose
    """
    pass