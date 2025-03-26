import os
import uuid
import json
import requests
import tempfile
import logging
from flask import Flask, request, jsonify
import soundfile as sf
import numpy as np
from fireredasr.models.fireredasr import FireRedAsr

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量存储模型
asr_model = None

def load_model():
    """加载ASR模型"""
    global asr_model
    if asr_model is None:
        try:
            # 从预训练模型目录加载模型
            model_dir = "pretrained_models/FireRedASR-AED-L"
            asr_model = FireRedAsr.from_pretrained("aed", model_dir)
            logger.info("ASR模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise e
    return asr_model

def download_file(url, local_path):
    """从URL下载文件到本地路径"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"文件下载失败: {str(e)}")
        return False

def process_audio(file_path):
    """处理音频文件，检测并处理多声道"""
    try:
        # 使用soundfile读取音频
        data, samplerate = sf.read(file_path)
        
        # 检查是否为双声道(立体声)
        is_stereo = len(data.shape) > 1 and data.shape[1] >= 2
        
        if is_stereo:
            # 分离左右声道
            left_channel = data[:, 0]
            right_channel = data[:, 1]
            
            # 为左右声道创建临时文件
            left_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            right_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            # 保存分离的声道
            sf.write(left_temp.name, left_channel, samplerate)
            sf.write(right_temp.name, right_channel, samplerate)
            
            return {
                "is_stereo": True,
                "left_path": left_temp.name,
                "right_path": right_temp.name,
                "original_path": file_path
            }
        else:
            # 单声道直接返回原文件路径
            return {
                "is_stereo": False,
                "original_path": file_path
            }
    except Exception as e:
        logger.error(f"音频处理失败: {str(e)}")
        raise e

def cleanup_files(file_paths):
    """清理临时文件"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"已删除临时文件: {path}")
        except Exception as e:
            logger.error(f"删除临时文件失败 {path}: {str(e)}")


@app.route('/api/asr', methods=['POST'])
def asr_transcribe():
    """ASR转录接口"""
    try:
        # 获取请求参数
        request_data = request.get_json()
        if not request_data or 'url' not in request_data:
            return jsonify({"error": "请求必须包含音频文件URL"}), 400
        
        audio_url = request_data.get('url')
        use_gpu = request_data.get('use_gpu', 1)
        beam_size = request_data.get('beam_size', 3)
        
        # 创建临时目录用于存放下载的文件
        temp_dir = tempfile.mkdtemp()
        unique_id = str(uuid.uuid4())
        temp_file = os.path.join(temp_dir, f"{unique_id}.wav")
        
        # 下载文件
        if not download_file(audio_url, temp_file):
            cleanup_files([temp_file])
            return jsonify({"error": "音频文件下载失败"}), 400
        
        # 处理音频文件(检测声道)
        audio_info = process_audio(temp_file)
        
        # 加载模型
        model = load_model()
        
        # 设置转录参数
        transcribe_args = {
            "use_gpu": use_gpu,
            "beam_size": beam_size,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.25,
            "aed_length_penalty": 0.6,
            "eos_penalty": 1.0
        }
        
        result = {}
        
        # 根据是否为立体声进行不同处理
        if audio_info["is_stereo"]:
            print("audio_info", audio_info)
            # 处理左声道
            left_results = model.transcribe(
                [f"{unique_id}"],
                [audio_info["left_path"]],
                transcribe_args
            )
            
            # 处理右声道
            right_results = model.transcribe(
                [f"{unique_id}"],
                [audio_info["right_path"]],
                transcribe_args
            )
            
            # 合并结果
            result = {
                "is_stereo": True,
                "left_channel": left_results[0],
                "right_channel": right_results[0],
                "rtf": left_results[0]["rtf"]  # 使用左声道的RTF作为整体RTF
            }
            
            # 清理临时文件
            cleanup_files([
                audio_info["original_path"],
                audio_info["left_path"],
                audio_info["right_path"]
            ])
        else:
            # 单声道处理
            transcribe_results = model.transcribe(
                [unique_id],
                [audio_info["original_path"]],
                transcribe_args
            )
            
            result = {
                "is_stereo": False,
                "transcript": transcribe_results[0],
                "rtf": transcribe_results[0]["rtf"]
            }
            
            # 清理临时文件
            cleanup_files([audio_info["original_path"]])

        print("result", result)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

if __name__ == '__main__':
    # 预加载模型
    try:
        load_model()
    except Exception as e:
        logger.error(f"启动服务时无法加载模型: {str(e)}")
    
    # 启动Flask服务
    app.run(host='0.0.0.0', port=5000, debug=False) 