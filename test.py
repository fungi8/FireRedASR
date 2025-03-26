from fireredasr.models.fireredasr import FireRedAsr

batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["examples/wav/multidomain_sample1.wav",
                  "examples/wav/multidomain_sample2.wav",
                  "examples/wav/multidomain_sample3.wav",
                  "examples/wav/multidomain_sample4.wav",
                  "examples/wav/multidomain_sample5.wav",
                  "examples/wav/multidomain_sample6.wav",
                  "examples/wav/multidomain_sample7.wav",
                  "examples/wav/multidomain_sample8.wav",
                  "examples/wav/sichuan_dialect_sample.wav",
                  "examples/wav/wuu_dialect_sample.wav",
                  "examples/wav/xiang_dialect_sample.wav",
                  "examples/wav/zgyu_dialect_sample.wav",
                  "examples/wav/cantonese_dialect_sample.wav"]

# FireRedASR-AED
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,
        "beam_size": 3,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.25,
        "aed_length_penalty": 0.6,
        "eos_penalty": 1.0
    }
)
print(results)