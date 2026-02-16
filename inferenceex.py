import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import time
import torchaudio


def create_embedding():
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')\
    embeddingcr = cosyvoice.frontend.save_voice_profile("voicePreviewData.mp3", prompt_text="Amidst the market stores and cobblestone streets, she discovered the true hard beat of her kingdom.", save_path="embeddings/speaker1.pt", resample_rate=cosyvoice.sample_rate)
    print ("created embedding")
    
    first = time.time()
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot_from_profile(tts_text="Hello, this is my pre-saved voice talking without re-processing the audio file.", model_input="embeddings/speaker1.pt", 
    stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    print ("generated audio")
    print(f"time taken in ms for the inference: {time.time() - first}")


def main():
    create_embedding()


if __name__ == '__main__':
    main()
