# -*- coding: utf-8 -*-
"""
暗区音频识别 —— 配置管理界面
直接运行：python ui.py
"""

import os
import re
import sys
import json
import shutil
from threading import Event

from PySide6.QtCore import Qt, Signal, QThread, QUrl
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QTabWidget,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QPlainTextEdit,
    QSplitter,
    QHeaderView,
    QAbstractItemView,
    QMessageBox,
    QFileDialog,
    QInputDialog,
)

# ================= 路径常量 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(BASE_DIR, "VoiceSource")
JSON_PATH = os.path.join(VOICE_DIR, "data.json")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# 支持的音频扩展名
AUDIO_EXTS = (".mp3", ".mp4", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus")

# 识别参数默认值（与 main2.py 中的 DEFAULT_CONFIG 保持一致）
DEFAULT_CONFIG = {
    "sample_rate": 22050,
    "fft_window_size": 2048,
    "hop_size": 512,
    "energy_threshold": 0.005,
    "debounce_silence_time": 0.5,
    "max_hold_time": 1.5,
    "cooldown_time": 1.0,
}


def list_audio_files(directory):
    """返回指定目录下所有音频文件的文件名（按名称排序）。"""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    )


def resolve_audio_path(file_name):
    """返回条目名对应的实际音频文件路径（自动兼容扩展名不一致，如 .mp4 条目对应 .mp3 文件）。"""
    path = os.path.join(VOICE_DIR, file_name)
    if os.path.exists(path):
        return path
    base = os.path.splitext(file_name)[0].lower()
    for f in list_audio_files(VOICE_DIR):
        if os.path.splitext(f)[0].lower() == base:
            return os.path.join(VOICE_DIR, f)
    return None


def find_associated_key(file_name, item_data):
    """返回 item_data 中与 file_name 关联的键（精确匹配或同基名），无则返回 None。"""
    if file_name in item_data:
        return file_name
    base = os.path.splitext(file_name)[0].lower()
    for key in item_data:
        if os.path.splitext(key)[0].lower() == base:
            return key
    return None


def is_audio_associated(file_name, item_data):
    """判断某个音频文件是否在 data.json 中有关联条目（兼容扩展名不一致）。"""
    return find_associated_key(file_name, item_data) is not None


def load_json(path, default):
    """读取 JSON 文件，失败时返回默认值。"""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            QMessageBox.warning(None, "读取失败", f"读取 {path} 失败：\n{e}")
    return default


def save_json(path, data):
    """写入 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


class StreamRedirector:
    """将 print 输出逐行转发到 Qt 信号（用于识别日志实时显示）。"""

    def __init__(self, emit):
        self._emit = emit
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._emit(line)

    def flush(self):
        if self._buf.strip():
            self._emit(self._buf)
            self._buf = ""


class RecognitionThread(QThread):
    """在后台线程中运行指纹构建与实时监听，避免阻塞界面。"""

    log = Signal(str)

    def __init__(self, config, item_data, stop_event):
        super().__init__()
        self.config = config
        self.item_data = item_data
        self.stop_event = stop_event

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirector(self.log.emit)
        sys.stderr = StreamRedirector(self.log.emit)
        try:
            # 延迟导入，避免界面启动时加载 librosa/soundcard 而卡顿
            import main2

            main2.JSON_PATH = JSON_PATH
            main2.VOICE_DIR = VOICE_DIR
            main2.apply_config(self.config)

            self.log.emit("🚀 开始构建模板指纹库...")
            target_fps = main2.build_target_fingerprints(self.item_data, main2.VOICE_DIR)

            if not target_fps:
                self.log.emit("❌ 未生成有效指纹，请检查音频文件与数据配置。")
                return

            self.log.emit("▶️ 开始实时监听...")
            main2.start_realtime_listener(
                self.item_data, target_fps, stop_event=self.stop_event
            )
        except Exception as e:
            self.log.emit(f"❌ 识别运行异常: {e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("暗区音频识别 · 配置管理")
        self.resize(1020, 720)

        self.item_data = load_json(JSON_PATH, {})
        self.current_audio = None
        self.dirty = False
        self.stop_event = Event()
        self.thread = None

        # 确保必要目录存在
        os.makedirs(VOICE_DIR, exist_ok=True)

        self._build_ui()
        self._load_config_to_widgets()
        self.refresh_audio_list()
        self.refresh_audio_files()
        self.refresh_play_list()
        self._update_status()

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_run_tab(), "识别运行")
        self.tabs.addTab(self._build_config_tab(), "识别参数")
        self.tabs.addTab(self._build_items_tab(), "物品数据")
        self.tabs.addTab(self._build_audio_tab(), "音频文件")
        

        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_config_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("算法与防抖参数")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.sample_rate = QSpinBox()
        self.sample_rate.setRange(8000, 192000)
        self.sample_rate.setSingleStep(100)
        form.addRow("采样率 (Sample Rate)：", self.sample_rate)

        self.fft_window = QComboBox()
        self.fft_window.addItems(["256", "512", "1024", "2048", "4096", "8192"])
        form.addRow("FFT 窗口大小：", self.fft_window)

        self.hop_size = QSpinBox()
        self.hop_size.setRange(64, 4096)
        self.hop_size.setSingleStep(64)
        form.addRow("帧步长 (Hop Size)：", self.hop_size)

        self.energy_threshold = QDoubleSpinBox()
        self.energy_threshold.setRange(0.0, 1.0)
        self.energy_threshold.setDecimals(4)
        self.energy_threshold.setSingleStep(0.001)
        form.addRow("激活音量门槛：", self.energy_threshold)

        self.debounce_silence = QDoubleSpinBox()
        self.debounce_silence.setRange(0.0, 60.0)
        self.debounce_silence.setDecimals(2)
        self.debounce_silence.setSingleStep(0.1)
        form.addRow("防抖停顿时间 (秒)：", self.debounce_silence)

        self.max_hold = QDoubleSpinBox()
        self.max_hold.setRange(0.0, 60.0)
        self.max_hold.setDecimals(2)
        self.max_hold.setSingleStep(0.1)
        form.addRow("最长挂起时间 (秒)：", self.max_hold)

        self.cooldown = QDoubleSpinBox()
        self.cooldown.setRange(0.0, 60.0)
        self.cooldown.setDecimals(2)
        self.cooldown.setSingleStep(0.1)
        form.addRow("全局冷却时间 (秒)：", self.cooldown)

        layout.addWidget(group)

        btn_row = QHBoxLayout()
        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_config)
        self.reset_config_btn = QPushButton("恢复默认")
        self.reset_config_btn.clicked.connect(self.reset_config)
        btn_row.addWidget(self.save_config_btn)
        btn_row.addWidget(self.reset_config_btn)
        btn_row.addStretch(1)

        tip = QLabel("提示：参数保存到 config.json，启动识别时自动应用。")
        tip.setStyleSheet("color: gray;")
        layout.addLayout(btn_row)
        layout.addWidget(tip)
        layout.addStretch(1)
        return page

    def _build_items_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)

        # 左侧：音频条目列表
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("音频条目（data.json 的键）："))
        self.audio_list = QListWidget()
        self.audio_list.currentRowChanged.connect(self._on_current_row_changed)
        left_layout.addWidget(self.audio_list)

        left_btns = QHBoxLayout()
        self.add_audio_btn = QPushButton("添加音频")
        self.add_audio_btn.clicked.connect(self.add_audio_item)
        self.del_audio_btn = QPushButton("删除音频")
        self.del_audio_btn.clicked.connect(self.delete_audio_item)
        left_btns.addWidget(self.add_audio_btn)
        left_btns.addWidget(self.del_audio_btn)
        left_layout.addLayout(left_btns)

        # 右侧：格子与物品映射表
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("格子尺寸 → 物品列表（物品用「、」分隔）："))
        self.grid_table = QTableWidget(0, 2)
        self.grid_table.setHorizontalHeaderLabels(["格子尺寸", "物品列表"])
        self.grid_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.grid_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.grid_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.grid_table.itemChanged.connect(self._on_table_changed)
        right_layout.addWidget(self.grid_table)

        right_btns = QHBoxLayout()
        self.add_grid_btn = QPushButton("添加格子")
        self.add_grid_btn.clicked.connect(self.add_grid)
        self.del_grid_btn = QPushButton("删除格子")
        self.del_grid_btn.clicked.connect(self.delete_grid)
        right_btns.addWidget(self.add_grid_btn)
        right_btns.addWidget(self.del_grid_btn)
        right_btns.addStretch(1)
        right_layout.addLayout(right_btns)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # 底部操作条
        bottom = QHBoxLayout()
        self.save_data_btn = QPushButton("保存数据")
        self.save_data_btn.clicked.connect(self.save_data)
        self.reload_data_btn = QPushButton("重新加载")
        self.reload_data_btn.clicked.connect(self.reload_data)
        bottom.addWidget(self.save_data_btn)
        bottom.addWidget(self.reload_data_btn)
        bottom.addStretch(1)
        layout.addLayout(bottom)
        return page

    def _build_audio_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("VoiceSource 目录下的音频文件（带 [已关联] 表示在 data.json 中已有条目）："))
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        btns = QHBoxLayout()
        self.import_btn = QPushButton("导入音频")
        self.import_btn.clicked.connect(self.import_audio)
        self.del_file_btn = QPushButton("删除音频")
        self.del_file_btn.clicked.connect(self.delete_audio_file)
        self.refresh_file_btn = QPushButton("刷新")
        self.refresh_file_btn.clicked.connect(self.refresh_audio_files)
        btns.addWidget(self.import_btn)
        btns.addWidget(self.del_file_btn)
        btns.addWidget(self.refresh_file_btn)
        btns.addStretch(1)
        layout.addLayout(btns)
        return page

    def _build_run_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QHBoxLayout()
        self.run_button = QPushButton("开始识别")
        self.run_button.setMinimumHeight(36)
        self.run_button.clicked.connect(self.on_run_clicked)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.clear_log)
        top.addWidget(self.run_button)
        top.addWidget(self.clear_log_btn)
        top.addStretch(1)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        # 左侧：音频播放列表（列出音频与关联数据，点击 ▶ 播放）
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("音频列表（点击 ▶ 播放）："))
        self.play_list = QListWidget()
        self.play_list.setMinimumWidth(280)
        left_layout.addWidget(self.play_list)

        # 右侧：识别日志
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("识别日志："))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.log_view)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # 音频播放器（用于播放模板音频）
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.errorOccurred.connect(self._on_player_error)
        self._playing_button = None
        self._playing_file = None
        return page

    # ------------------------------------------------------------------
    # 识别参数页
    # ------------------------------------------------------------------
    def collect_config(self):
        return {
            "sample_rate": self.sample_rate.value(),
            "fft_window_size": int(self.fft_window.currentText()),
            "hop_size": self.hop_size.value(),
            "energy_threshold": round(self.energy_threshold.value(), 4),
            "debounce_silence_time": round(self.debounce_silence.value(), 2),
            "max_hold_time": round(self.max_hold.value(), 2),
            "cooldown_time": round(self.cooldown.value(), 2),
        }

    def _load_config_to_widgets(self):
        cfg = load_json(CONFIG_PATH, DEFAULT_CONFIG)
        cfg = {**DEFAULT_CONFIG, **cfg}
        self.sample_rate.setValue(int(cfg["sample_rate"]))
        self.fft_window.setCurrentText(str(cfg["fft_window_size"]))
        self.hop_size.setValue(int(cfg["hop_size"]))
        self.energy_threshold.setValue(float(cfg["energy_threshold"]))
        self.debounce_silence.setValue(float(cfg["debounce_silence_time"]))
        self.max_hold.setValue(float(cfg["max_hold_time"]))
        self.cooldown.setValue(float(cfg["cooldown_time"]))

    def save_config(self):
        save_json(CONFIG_PATH, self.collect_config())
        self.statusBar().showMessage(f"已保存识别参数到 {CONFIG_PATH}", 5000)

    def reset_config(self):
        self._apply_config_to_widgets(DEFAULT_CONFIG)

    def _apply_config_to_widgets(self, cfg):
        self.sample_rate.setValue(int(cfg["sample_rate"]))
        self.fft_window.setCurrentText(str(cfg["fft_window_size"]))
        self.hop_size.setValue(int(cfg["hop_size"]))
        self.energy_threshold.setValue(float(cfg["energy_threshold"]))
        self.debounce_silence.setValue(float(cfg["debounce_silence_time"]))
        self.max_hold.setValue(float(cfg["max_hold_time"]))
        self.cooldown.setValue(float(cfg["cooldown_time"]))

    # ------------------------------------------------------------------
    # 物品数据页
    # ------------------------------------------------------------------
    def refresh_audio_list(self, select=None):
        """重建左侧音频条目列表，并刷新右侧表格。"""
        self.audio_list.blockSignals(True)
        self.audio_list.clear()
        for name in self.item_data:
            self.audio_list.addItem(name)

        self.current_audio = None
        if select is not None and select in self.item_data:
            names = list(self.item_data.keys())
            self.audio_list.setCurrentRow(names.index(select))
            self.current_audio = select
        self.audio_list.blockSignals(False)
        self.refresh_grid_table()

    def _on_current_row_changed(self, row):
        """用户切换左侧选中项时，先保存旧表格编辑，再载入新条目。"""
        self.flush_grid_table()
        if 0 <= row < self.audio_list.count():
            self.current_audio = self.audio_list.item(row).text()
        else:
            self.current_audio = None
        self.refresh_grid_table()

    def refresh_grid_table(self):
        """把当前选中音频的「格子→物品」映射渲染到右侧表格。"""
        self.grid_table.blockSignals(True)
        self.grid_table.setRowCount(0)
        data = self.item_data.get(self.current_audio, {}) if self.current_audio else {}
        for grid, items in data.items():
            row = self.grid_table.rowCount()
            self.grid_table.insertRow(row)
            items_text = "、".join(items) if isinstance(items, list) else str(items)
            self.grid_table.setItem(row, 0, QTableWidgetItem(str(grid)))
            self.grid_table.setItem(row, 1, QTableWidgetItem(items_text))
        self.grid_table.blockSignals(False)

    def flush_grid_table(self):
        """把右侧表格的编辑结果写回 self.item_data[self.current_audio]。"""
        audio = self.current_audio
        if audio is None:
            return
        data = {}
        for row in range(self.grid_table.rowCount()):
            grid_item = self.grid_table.item(row, 0)
            items_item = self.grid_table.item(row, 1)
            grid = (grid_item.text().strip() if grid_item else "").lower()
            items_text = items_item.text() if items_item else ""
            items = [s.strip() for s in re.split(r"[、,，;；\n]+", items_text) if s.strip()]
            if grid:
                data[grid] = items
        self.item_data[audio] = data

    def _on_table_changed(self, *args):
        self._mark_dirty()

    def _mark_dirty(self):
        self.dirty = True
        self._update_status()

    def add_audio_item(self):
        self.flush_grid_table()
        files = list_audio_files(VOICE_DIR)
        if not files:
            QMessageBox.information(
                self, "提示",
                "VoiceSource 目录中没有音频文件。\n请先到「音频文件」页导入音频。",
            )
            return
        text, ok = QInputDialog.getItem(
            self, "添加音频条目", "选择要关联的音频文件：", files, 0, False
        )
        if not ok or not text:
            return
        if text in self.item_data:
            QMessageBox.warning(self, "提示", f"音频条目「{text}」已存在。")
            return
        self.item_data[text] = {}
        self.refresh_audio_list(select=text)
        self._mark_dirty()

    def delete_audio_item(self):
        row = self.audio_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先在左侧选择一个音频条目。")
            return
        name = self.audio_list.item(row).text()
        ret = QMessageBox.question(
            self, "确认删除",
            f"确定删除音频条目「{name}」吗？\n（仅从 data.json 移除，不会删除音频文件）",
        )
        if ret != QMessageBox.Yes:
            return
        self.item_data.pop(name, None)
        self.current_audio = None
        self.refresh_audio_list()
        self._mark_dirty()

    def add_grid(self):
        if self.current_audio is None:
            QMessageBox.information(self, "提示", "请先在左侧选择一个音频条目。")
            return
        self.flush_grid_table()
        text, ok = QInputDialog.getText(
            self, "添加格子", "输入格子尺寸（例如 2x2、1x3）："
        )
        if not ok or not text.strip():
            return
        grid = text.strip().lower()
        if not re.fullmatch(r"\d+x\d+", grid):
            QMessageBox.warning(self, "格式错误", "格子尺寸格式应为「宽x高」，例如 2x2、1x3。")
            return
        data = self.item_data[self.current_audio]
        if grid in data:
            QMessageBox.warning(self, "提示", f"格子「{grid}」已存在。")
            return
        data[grid] = []
        self.refresh_grid_table()
        self._mark_dirty()

    def delete_grid(self):
        if self.current_audio is None:
            return
        row = self.grid_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先在右侧选择一个格子行。")
            return
        grid_item = self.grid_table.item(row, 0)
        if grid_item is None:
            return
        grid = grid_item.text().strip().lower()
        self.flush_grid_table()
        self.item_data[self.current_audio].pop(grid, None)
        self.refresh_grid_table()
        self._mark_dirty()

    def write_data_file(self):
        save_json(JSON_PATH, self.item_data)

    def save_data(self, silent=False):
        self.flush_grid_table()
        self.write_data_file()
        self.dirty = False
        self._update_status()
        if not silent:
            self.statusBar().showMessage(f"已保存物品数据到 {JSON_PATH}", 5000)

    def reload_data(self):
        if self.dirty:
            ret = QMessageBox.question(
                self, "未保存更改", "有未保存的更改，重新加载将丢失这些更改。是否继续？"
            )
            if ret != QMessageBox.Yes:
                return
        self.item_data = load_json(JSON_PATH, {})
        self.dirty = False
        self.current_audio = None
        self.refresh_audio_list()
        self.refresh_audio_files()
        self._update_status()

    # ------------------------------------------------------------------
    # 音频文件页
    # ------------------------------------------------------------------
    def refresh_audio_files(self):
        self.file_list.clear()
        for f in list_audio_files(VOICE_DIR):
            item = QListWidgetItem()
            item.setText(f + ("  [已关联]" if is_audio_associated(f, self.item_data) else ""))
            item.setData(Qt.UserRole, f)
            self.file_list.addItem(item)

    def import_audio(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.mp3 *.mp4 *.wav *.ogg *.flac *.m4a *.aac *.wma *.opus)",
        )
        if not files:
            return
        imported = []
        for src in files:
            base = os.path.basename(src)
            dst = os.path.join(VOICE_DIR, base)
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            if os.path.exists(dst):
                ret = QMessageBox.question(
                    self, "文件已存在", f"「{base}」已存在，是否覆盖？"
                )
                if ret != QMessageBox.Yes:
                    continue
            try:
                shutil.copy2(src, dst)
                imported.append(base)
            except Exception as e:
                QMessageBox.critical(self, "导入失败", f"导入「{base}」失败：\n{e}")
        self.refresh_audio_files()
        if imported:
            self.statusBar().showMessage(f"成功导入 {len(imported)} 个音频文件。", 5000)

    def delete_audio_file(self):
        item = self.file_list.currentItem()
        if item is None:
            QMessageBox.information(self, "提示", "请先选择一个音频文件。")
            return
        fname = item.data(Qt.UserRole)
        key = find_associated_key(fname, self.item_data)
        referenced = key is not None
        msg = f"确定删除音频文件「{fname}」吗？"
        if referenced:
            msg += "\n该文件在 data.json 中有关联条目，删除时会一并移除条目。"
        ret = QMessageBox.question(self, "确认删除", msg)
        if ret != QMessageBox.Yes:
            return
        try:
            os.remove(os.path.join(VOICE_DIR, fname))
        except OSError as e:
            QMessageBox.critical(self, "删除失败", f"删除失败：\n{e}")
            return
        if referenced:
            self.flush_grid_table()
            self.item_data.pop(key, None)
            self.write_data_file()
            self.current_audio = None
            self.dirty = False
            self.refresh_audio_list()
        self.refresh_audio_files()
        self._update_status()

    # ------------------------------------------------------------------
    # 识别运行页
    # ------------------------------------------------------------------
    def on_run_clicked(self):
        if self.thread is not None and self.thread.isRunning():
            self.stop_recognition()
        else:
            self.start_recognition()

    def start_recognition(self):
        # 先同步保存配置与数据，确保识别使用界面上的最新值
        self.save_config()
        self.flush_grid_table()
        if self.dirty:
            self.save_data(silent=True)

        if not self.item_data:
            QMessageBox.warning(self, "提示", "data.json 为空，请先在「物品数据」页添加条目。")
            return

        self.stop_event.clear()
        self.thread = RecognitionThread(self.collect_config(), self.item_data, self.stop_event)
        self.thread.log.connect(self.append_log)
        self.thread.finished.connect(self.on_recognition_finished)
        self.thread.start()

        self.run_button.setText("停止识别")
        self.append_log("════════ 开始识别 ════════")

    def stop_recognition(self):
        if self.thread is not None and self.thread.isRunning():
            self.stop_event.set()
            self.append_log("正在停止...")

    def on_recognition_finished(self):
        self.run_button.setText("开始识别")

    def append_log(self, text):
        self.log_view.appendPlainText(text)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self):
        self.log_view.clear()

    def _on_tab_changed(self, index):
        """切到「识别运行」页时刷新音频播放列表，确保与最新数据一致。"""
        if index == 3:
            self.refresh_play_list()

    def refresh_play_list(self):
        """重建侧边音频播放列表：每个条目展示文件名、关联数据与播放按钮。"""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.stop()
        self._playing_button = None
        self._playing_file = None

        self.play_list.clear()
        for name, data in self.item_data.items():
            item = QListWidgetItem()
            widget = self._make_play_row(name, data)
            item.setSizeHint(widget.sizeHint())
            self.play_list.addItem(item)
            self.play_list.setItemWidget(item, widget)

    def _make_play_row(self, name, data):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(4, 6, 4, 6)

        play_btn = QPushButton("▶")
        play_btn.setFixedSize(34, 30)
        play_btn.setToolTip(f"播放 {name}")
        play_btn.clicked.connect(lambda checked=False, n=name, b=play_btn: self.toggle_play(n, b))
        h.addWidget(play_btn, 0, Qt.AlignTop)

        v = QVBoxLayout()
        v.setSpacing(2)
        name_lbl = QLabel(name)
        font = name_lbl.font()
        font.setBold(True)
        name_lbl.setFont(font)
        v.addWidget(name_lbl)

        lines = []
        for grid, items in data.items():
            text = "、".join(items) if isinstance(items, list) else str(items)
            lines.append(f"{grid} → {text}")
        data_lbl = QLabel("\n".join(lines) if lines else "（暂无关联数据）")
        data_lbl.setStyleSheet("color: #666666;")
        v.addWidget(data_lbl)

        h.addLayout(v, 1)
        return row

    def toggle_play(self, name, button):
        """播放 / 停止某个模板音频。"""
        if self._playing_file == name:
            self.player.stop()
            return
        path = resolve_audio_path(name)
        if path is None:
            QMessageBox.warning(self, "提示", f"找不到音频文件：{name}")
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()
        if self._playing_button is not None:
            self._playing_button.setText("▶")
        button.setText("⏸")
        self._playing_button = button
        self._playing_file = name

    def _on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState and self._playing_button is not None:
            self._playing_button.setText("▶")
            self._playing_button = None
            self._playing_file = None

    def _on_player_error(self, error, error_string):
        if self._playing_button is not None:
            self._playing_button.setText("▶")
            self._playing_button = None
            self._playing_file = None
        self.append_log(f"⚠️ 音频播放失败：{error_string}")

    # ------------------------------------------------------------------
    # 状态与关闭
    # ------------------------------------------------------------------
    def _update_status(self):
        msg = f"数据文件：{JSON_PATH}"
        if self.dirty:
            msg += "   ● 有未保存的更改"
        self.statusBar().showMessage(msg)

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            ret = QMessageBox.question(self, "确认退出", "识别正在运行，确定退出吗？")
            if ret != QMessageBox.Yes:
                event.ignore()
                return
            self.stop_event.set()
            self.thread.wait(3000)

        if self.dirty:
            self.flush_grid_table()
            ret = QMessageBox.question(
                self,
                "未保存更改",
                "物品数据有未保存的更改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if ret == QMessageBox.Save:
                self.save_data(silent=True)
            elif ret == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
