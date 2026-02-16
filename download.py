from huggingface_hub import snapshot_download


if __name__ == "__main__":
    snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
  
