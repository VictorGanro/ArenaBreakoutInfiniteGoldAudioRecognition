import os
import json
import time
import warnings
from collections import Counter
import numpy as np
import scipy.signal as signal
import librosa
import soundcard as sc

warnings.filterwarnings("ignore")

# ================= 1. 全局配置与参数设置 =================
JSON_PATH = "VoiceSource/data.json"
VOICE_DIR = "VoiceSource"
CONFIG_PATH = "config.json"

# 识别参数的默认值（当 config.json 不存在或字段缺失时使用）
DEFAULT_CONFIG = {
    "sample_rate": 22050,        # 统一采样率
    "fft_window_size": 2048,     # FFT 窗口大小
    "hop_size": 512,             # 帧步长
    "energy_threshold": 0.005,   # 最低激活音量门槛
    "debounce_silence_time": 0.5,  # 停顿多长时间无更新后，正式结算输出（秒）
    "max_hold_time": 1.5,        # 最长挂起等待时间，超过此时间强制输出（秒）
    "cooldown_time": 1.0,        # 输出后的全局冷却时间（秒）
}

# 运行时使用的全局参数（由 apply_config 填充）
SAMPLE_RATE = DEFAULT_CONFIG["sample_rate"]
FFT_WINDOW_SIZE = DEFAULT_CONFIG["fft_window_size"]
HOP_SIZE = DEFAULT_CONFIG["hop_size"]
ENERGY_THRESHOLD = DEFAULT_CONFIG["energy_threshold"]
DEBOUNCE_SILENCE_TIME = DEFAULT_CONFIG["debounce_silence_time"]
MAX_HOLD_TIME = DEFAULT_CONFIG["max_hold_time"]
COOLDOWN_TIME = DEFAULT_CONFIG["cooldown_time"]


def load_config():
    """从 config.json 读取识别参数，缺失字段回退到默认值。"""
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"⚠️  读取 config.json 失败，使用默认配置: {e}")
    return cfg


def apply_config(cfg):
    """将配置字典应用到运行时全局参数（UI 启动识别前会调用）。"""
    global SAMPLE_RATE, FFT_WINDOW_SIZE, HOP_SIZE, ENERGY_THRESHOLD
    global DEBOUNCE_SILENCE_TIME, MAX_HOLD_TIME, COOLDOWN_TIME
    SAMPLE_RATE = int(cfg.get("sample_rate", DEFAULT_CONFIG["sample_rate"]))
    FFT_WINDOW_SIZE = int(cfg.get("fft_window_size", DEFAULT_CONFIG["fft_window_size"]))
    HOP_SIZE = int(cfg.get("hop_size", DEFAULT_CONFIG["hop_size"]))
    ENERGY_THRESHOLD = float(cfg.get("energy_threshold", DEFAULT_CONFIG["energy_threshold"]))
    DEBOUNCE_SILENCE_TIME = float(cfg.get("debounce_silence_time", DEFAULT_CONFIG["debounce_silence_time"]))
    MAX_HOLD_TIME = float(cfg.get("max_hold_time", DEFAULT_CONFIG["max_hold_time"]))
    COOLDOWN_TIME = float(cfg.get("cooldown_time", DEFAULT_CONFIG["cooldown_time"]))


# 模块加载时应用一次配置
apply_config(load_config())

# ================= 2. 音频指纹提取算法 =================
def extract_fingerprints(samples, sr=SAMPLE_RATE):
    """提取携带相对时间戳的频域峰值指纹"""
    if len(samples) < FFT_WINDOW_SIZE:
        return {}
    
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    _, _, spec = signal.stft(
        samples, 
        fs=sr, 
        nperseg=FFT_WINDOW_SIZE, 
        noverlap=FFT_WINDOW_SIZE - HOP_SIZE
    )
    spec_mag = np.abs(spec)
    
    max_val = np.max(spec_mag)
    if max_val > 0:
        spec_mag = spec_mag / max_val

    hashes = {}
    num_freq_bins, num_time_bins = spec_mag.shape
    bands = [(0, 12), (12, 28), (28, 50), (50, 90)]
    
    for t in range(num_time_bins):
        for f_min, f_max in bands:
            if f_max < num_freq_bins:
                sub_spec = spec_mag[f_min:f_max, t]
                max_f = np.argmax(sub_spec) + f_min
                max_v = spec_mag[max_f, t]
                
                if max_v > 0.18:
                    hashes[max_f] = hashes.get(max_f, []) + [t]
                    
    return hashes

# ================= 3. 构建模板指纹库 =================
def build_target_fingerprints(item_data, voice_dir):
    print("--------------------------------------------------")
    print("正在预加载并分析 VoiceSource 模板音频指纹...")
    target_fps = {}

    for file_name in item_data.keys():
        file_path = os.path.join(voice_dir, file_name)
        
        if not os.path.exists(file_path):
            alt_ext = ".mp3" if file_name.endswith(".mp4") else ".mp4"
            alt_path = os.path.join(voice_dir, os.path.splitext(file_name)[0] + alt_ext)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                print(f"⚠️  警告: 找不到文件 -> {file_path}")
                continue
        
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            
            all_fps = []
            max_fp_count = 0
            
            for speed in [0.95, 1.0, 1.05]:
                if speed == 1.0:
                    y_stretched = y
                else:
                    y_stretched = librosa.effects.time_stretch(y, rate=speed)
                    
                fps = extract_fingerprints(y_stretched, sr)
                all_fps.append(fps)
                max_fp_count = max(max_fp_count, len(fps))
                
            target_fps[file_name] = {
                "fps_list": all_fps,
                "total_features": max_fp_count
            }
            print(f"✅ 成功加载: {file_name:<10} | 基础特征数: {max_fp_count}")
            
        except Exception as e:
            print(f"❌ 加载文件 {file_name} 失败: {e}")

    print(f"模板库加载完成，共有效加载 {len(target_fps)} 个文件。")
    print("--------------------------------------------------\n")
    return target_fps

# ================= 4. 时序对齐匹配逻辑 =================
def match_fingerprints(current_hashes, target_fps_list):
    """计算当前音频与模板的最高时间差对齐得分"""
    best_score = 0
    
    for ref_hashes in target_fps_list:
        offsets = []
        for freq in current_hashes:
            if freq in ref_hashes:
                for t_curr in current_hashes[freq]:
                    for t_ref in ref_hashes[freq]:
                        offsets.append(t_curr - t_ref)
                        
        if offsets:
            offset_counts = Counter(offsets)
            _, count = offset_counts.most_common(1)[0]
            if count > best_score:
                best_score = count
                
    return best_score

# ================= 5. 带“防抖/停顿合并”机制的实时监听引擎 =================
def start_realtime_listener(item_data, target_fingerprints, stop_event=None):
    try:
        speaker = sc.default_speaker()
        mics = sc.all_microphones(include_loopback=True)
        
        loopback_mic = None
        for mic in mics:
            if mic.isloopback and speaker.name in mic.name:
                loopback_mic = mic
                break
                
        if loopback_mic is None:
            for mic in mics:
                if mic.isloopback:
                    loopback_mic = mic
                    break

        if loopback_mic is None:
            print("❌ 错误: 未能在 Windows 系统中检测到任何 Loopback 设备。")
            return

    except Exception as e:
        print(f"❌ 初始化内录设备失败: {e}")
        return

    print(f"🎧 成功锁定内录通道 -> [{loopback_mic.name}]")

    buffer_duration = 2.0  # 稍微加长缓存区至 2.0 秒，能完整容纳“两声狗叫及中间停顿”
    buffer_size = int(SAMPLE_RATE * buffer_duration)
    ring_buffer = np.zeros(buffer_size, dtype=np.float32)

    last_output_time = 0
    record_sr = 48000
    block_size = int(record_sr * 0.1)

    # 挂起防抖状态变量
    pending_match = None  # 保存结构: {"name": str, "score": int, "ratio": float, "first_seen": float, "last_updated": float, "rms": float}

    print("🚀 智能防抖合并匹配引擎已就绪，正在监听系统音频...\n")

    with loopback_mic.recorder(samplerate=record_sr, channels=1) as recorder:
        while True:
            if stop_event is not None and stop_event.is_set():
                print("⏹️ 已手动停止监听。")
                break
            current_time = time.time()
            data = recorder.record(numframes=block_size)
            data_np = data.squeeze()

            if record_sr != SAMPLE_RATE:
                data_np = librosa.resample(data_np, orig_sr=record_sr, target_sr=SAMPLE_RATE)

            chunk_len = len(data_np)
            if chunk_len > 0:
                ring_buffer = np.roll(ring_buffer, -chunk_len)
                ring_buffer[-chunk_len:] = data_np

            rms_energy = np.sqrt(np.mean(ring_buffer ** 2))

            # 只有在不在冷却期时，才参与识别
            if current_time - last_output_time > COOLDOWN_TIME:
                
                # 1. 如果有声音，进行特征提取与比对
                if rms_energy > ENERGY_THRESHOLD:
                    current_hashes = extract_fingerprints(ring_buffer, SAMPLE_RATE)
                    
                    if len(current_hashes) > 0:
                        best_match = None
                        max_ratio = 0.0
                        matched_score = 0
                        
                        for audio_name, target_info in target_fingerprints.items():
                            score = match_fingerprints(current_hashes, target_info["fps_list"])
                            total_features = target_info["total_features"]
                            ratio = score / total_features if total_features > 0 else 0
                            
                            if score >= 4 and ratio >= 0.30 and ratio > max_ratio:
                                max_ratio = ratio
                                matched_score = score
                                best_match = audio_name
                        
                        # 如果匹配到了有效的目标
                        if best_match:
                            if pending_match is None:
                                # 首次捕捉到，开启“防抖挂起”状态，不立刻输出
                                pending_match = {
                                    "name": best_match,
                                    "score": matched_score,
                                    "ratio": max_ratio,
                                    "first_seen": current_time,
                                    "last_updated": current_time,
                                    "rms": rms_energy
                                }
                            else:
                                # 如果是同一个音频（或更优结果），更新挂起状态，不断刷新 last_updated 时间
                                if best_match == pending_match["name"]:
                                    pending_match["last_updated"] = current_time
                                    if matched_score >= pending_match["score"]:
                                        pending_match["score"] = matched_score
                                        pending_match["ratio"] = max_ratio
                                        pending_match["rms"] = rms_energy
                                else:
                                    # 如果中途换成了得分更高的另一个音频
                                    if max_ratio > pending_match["ratio"]:
                                        pending_match["name"] = best_match
                                        pending_match["score"] = matched_score
                                        pending_match["ratio"] = max_ratio
                                        pending_match["last_updated"] = current_time

            # 2. 检查防抖结算条件：如果存在挂起项，且满足以下条件之一则【正式输出】
            if pending_match is not None:
                time_since_last_update = current_time - pending_match["last_updated"]
                total_hold_time = current_time - pending_match["first_seen"]
                
                # 条件 A：声音停顿时间达到了 DEBOUNCE_SILENCE_TIME (比如两声狗叫叫完了)
                # 条件 B：挂起总时间达到了 MAX_HOLD_TIME (防止无限等待)
                if time_since_last_update >= DEBOUNCE_SILENCE_TIME or total_hold_time >= MAX_HOLD_TIME:
                    target_name = pending_match["name"]
                    print("=" * 50)
                    print(f"🎯 最终识别成功 | 音频: {target_name} (综合得分: {pending_match['score']} | 匹配度: {pending_match['ratio']*100:.1f}% | 音量: {pending_match['rms']:.4f})")
                    print(f"📦 关联数据: {json.dumps(item_data[target_name], ensure_ascii=False)}")
                    print("=" * 50 + "\n")
                    
                    # 结算后重置
                    ring_buffer.fill(0)
                    last_output_time = current_time
                    pending_match = None

            time.sleep(0.02)

# ================= 6. 主程序入口 =================
if __name__ == "__main__":
    if not os.path.exists(JSON_PATH):
        print(f"❌ 错误: 找不到 json 配置文件 ({JSON_PATH})")
        exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        item_data = json.load(f)

    target_fingerprints = build_target_fingerprints(item_data, VOICE_DIR)

    if not target_fingerprints:
        print("❌ 未生成有效指纹，程序退出。")
        exit(1)

    try:
        start_realtime_listener(item_data, target_fingerprints)
    except KeyboardInterrupt:
        print("\n已手动停止监听。")