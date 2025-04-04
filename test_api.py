#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import argparse
import json
import time

def test_asr():
    url_list = ["https://sto-servicequality.oss-cn-shanghai.aliyuncs.com/f1b6f83e055241c39155f9edce89a93b.wav?Expires=1743050004&OSSAccessKeyId=LTAI5t5YHQ1jJR3oCK9ZhMpd&Signature=KXOC6tkEqmgmcDb88mzhGhvCl8I%3D",
                "https://sto-servicequality.oss-cn-shanghai.aliyuncs.com/f1b6f83e055241c39155f9edce89a93b.wav?Expires=1743050004&OSSAccessKeyId=LTAI5t5YHQ1jJR3oCK9ZhMpd&Signature=KXOC6tkEqmgmcDb88mzhGhvCl8I%3D",
                "https://sto-servicequality.oss-cn-shanghai.aliyuncs.com/f1b6f83e055241c39155f9edce89a93b.wav?Expires=1743050004&OSSAccessKeyId=LTAI5t5YHQ1jJR3oCK9ZhMpd&Signature=KXOC6tkEqmgmcDb88mzhGhvCl8I%3D",
                "https://sto-servicequality.oss-cn-shanghai.aliyuncs.com/f1b6f83e055241c39155f9edce89a93b.wav?Expires=1743050004&OSSAccessKeyId=LTAI5t5YHQ1jJR3oCK9ZhMpd&Signature=KXOC6tkEqmgmcDb88mzhGhvCl8I%3D",
                "https://sto-servicequality.oss-cn-shanghai.aliyuncs.com/f1b6f83e055241c39155f9edce89a93b.wav?Expires=1743050004&OSSAccessKeyId=LTAI5t5YHQ1jJR3oCK9ZhMpd&Signature=KXOC6tkEqmgmcDb88mzhGhvCl8I%3D"]
    for url in url_list:
        test_asr_api("http://localhost:5000/api/asr", url, 1, 1)

def test_asr_api(url, audio_url, use_gpu=1, beam_size=1):
    """测试ASR API"""
    print(f"正在发送请求到 {url}，处理音频 {audio_url}...")
    
    # 准备请求数据
    payload = {
        "url": audio_url,
        "use_gpu": use_gpu,
        "beam_size": beam_size
    }
    
    # 发送请求并计时
    start_time = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start_time
    
    print(f"请求耗时: {elapsed:.2f} 秒")
    
    # 检查响应
    if response.status_code == 200:
        # 请求成功
        result = response.json()
        print("\n转录结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        if result.get("is_stereo", False):
            print("\n左声道文本:")
            print(result["left_channel"]["text"])
            print("\n右声道文本:")
            print(result["right_channel"]["text"])
        else:
            print("\n文本:")
            print(result["transcript"]["text"])
            
        print(f"\nRTF: {result.get('rtf', '未知')}")
    else:
        # 请求失败
        print(f"请求失败，状态码: {response.status_code}")
        print("错误信息:", response.text)

if __name__ == "__main__":
    test_asr()
    # parser = argparse.ArgumentParser(description="测试FireRedASR Web服务API")
    # parser.add_argument("--server", default="http://localhost:5000/api/asr",
    #                     help="API服务器URL，默认为 http://localhost:5000/api/asr")
    # parser.add_argument("--audio", required=True,
    #                     help="要处理的音频文件URL")
    # parser.add_argument("--gpu", type=int, default=1,
    #                     help="是否使用GPU加速")
    # parser.add_argument("--beam", type=int, default=1,
    #                     help="Beam搜索大小，默认为3")
    #
    # args = parser.parse_args()
    #
    # test_asr_api(args.server, args.audio, args.gpu, args.beam)