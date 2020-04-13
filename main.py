from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import torch
import sys

if __name__ == '__main__':
    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" %
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))

    encoder.load_model(Path('encoder/saved_models/pretrained.pt'))
    synthesizer = Synthesizer(Path("synthesizer/saved_models/logs-pretrained/taco_pretrained"))
    vocoder.load_model(Path('vocoder/saved_models/pretrained/pretrained.pt'))

    voice = 'voices/peabody/voice.wav'

    try:
        preprocessed_wav = encoder.preprocess_wav(voice)
        embed = encoder.embed_utterance(preprocessed_wav)

        text = "Hello Carina. Hello Carina Hello Carina Hello Carina Hello Carina Hello Carina Hello Carina This is Kevin Smith, happy new translation around the Sun."

        texts = [text]
        embeds = [embed]

        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]

        generated_wav = vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        fpath = "demo_output_00.wav"
        librosa.output.write_wav(fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)

        print("\nSaved output as %s\n\n" % fpath)

    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")
