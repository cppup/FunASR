from funasr import AutoModel
import soundfile
import os

# 流式配置
chunk_size = [0, 10, 5]  # [0, 10, 5] = 600ms 延迟
encoder_chunk_look_back = 4  # encoder 回看块数
decoder_chunk_look_back = 1  # decoder 回看块数

model = AutoModel(model="paraformer-zh-streaming")

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
speech, sample_rate = soundfile. read(wav_file)
chunk_stride = chunk_size[1] * 960  # 600ms = 9600 采样点

cache = {}
total_chunk_num = int(len(speech) - 1) // chunk_stride + 1

for i in range(total_chunk_num):
    speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
    is_final = i == total_chunk_num - 1
    
    res = model.generate(
        input=speech_chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back
    )
    print(f"Chunk {i}: {res}")
