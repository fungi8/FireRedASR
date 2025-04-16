from fireredasr.models.fireredasr import FireRedAsr

# batch_uttids = ["multidomain_sample1",
#                "multidomain_sample2",
#                "multidomain_sample3",
#                "multidomain_sample4",
#                "multidomain_sample5",
#                "multidomain_sample6",
#                "multidomain_sample7",
#                "multidomain_sample8",
#                "sichuan_dialect_sample",
#                "wuu_dialect_sample",
#                "xiang_dialect_sample",
#                "zgyu_dialect_sample",
#                "cantonese_dialect_sample"]
# batch_wav_path_list = ["examples/wav/multidomain_sample1.wav",
#                   "examples/wav/multidomain_sample2.wav",
#                   "examples/wav/multidomain_sample3.wav",
#                   "examples/wav/multidomain_sample4.wav",
#                   "examples/wav/multidomain_sample5.wav",
#                   "examples/wav/multidomain_sample6.wav",
#                   "examples/wav/multidomain_sample7.wav",
#                   "examples/wav/multidomain_sample8.wav",
#                   "examples/wav/sichuan_dialect_sample.wav",
#                   "examples/wav/wuu_dialect_sample.wav",
#                   "examples/wav/xiang_dialect_sample.wav",
#                   "examples/wav/zgyu_dialect_sample.wav",
#                   "examples/wav/cantonese_dialect_sample.wav"]

# FireRedASR-AED
# model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")
#
# for i in range(len(batch_wav_path_list)):
#     print("Batch wav path: {}".format(batch_wav_path_list[i]), "uttid: {}".format(batch_uttids[i]))
#     result = model.transcribe(
#         [batch_uttids[i]],
#         [batch_wav_path_list[i]],
#         {
#             "use_gpu": 1,
#             "beam_size": 3,
#             "nbest": 1,
#             "decode_max_len": 0,
#             "softmax_smoothing": 1.25,
#             "aed_length_penalty": 0.6,
#             "eos_penalty": 1.0
#         }
#     )
#     print(result)



batch_uttid = ["班前会录音"]
batch_wav_path = ["examples/wav/班前会录音.mp3"]

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