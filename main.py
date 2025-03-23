from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
from bpm import BPMDetector
import sounddevice as sd

class BPMApp(App):
    def build(self):
        # 创建主布局
        self.main_layout = GridLayout(cols=1, spacing=10, padding=10)
        
        # 创建设备选择下拉框
        self.devices = self._get_audio_devices()
        self.device_spinner = Spinner(
            text='选择音频设备',
            values=[f"{i}: {name}" for i, name in self.devices],
            size_hint_y=None,
            height='48dp'
        )
        
        # BPM显示标签
        self.bpm_label = Label(
            text='当前BPM: --',
            font_size='30sp',
            size_hint_y=None,
            height='48dp'
        )
        
        # 状态标签
        self.status_label = Label(
            text='等待开始...',
            size_hint_y=None,
            height='48dp'
        )
        
        # 开始/停止按钮
        self.start_stop_button = Button(
            text='开始检测',
            size_hint_y=None,
            height='48dp',
            background_color=(0.3, 0.6, 0.3, 1)
        )
        self.start_stop_button.bind(on_press=self.toggle_detection)
        
        # 将组件添加到布局中
        self.main_layout.add_widget(self.device_spinner)
        self.main_layout.add_widget(self.bpm_label)
        self.main_layout.add_widget(self.status_label)
        self.main_layout.add_widget(self.start_stop_button)
        
        self.detector = None
        self.is_detecting = False
        
        return self.main_layout
    
    def _get_audio_devices(self):
        """获取可用的音频设备列表"""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    devices.append((i, device['name']))
        except Exception as e:
            print(f"获取音频设备列表出错: {e}")
        return devices
    
    def toggle_detection(self, instance):
        if not self.is_detecting:
            # 开始检测
            try:
                device_id = int(self.device_spinner.text.split(':')[0])
                self.detector = BPMDetector(device=device_id)
                self.detector.start()
                if self.detector.running:
                    self.start_stop_button.text = '停止检测'
                    self.start_stop_button.background_color = (0.8, 0.3, 0.3, 1)
                    self.status_label.text = '正在检测BPM...'
                    Clock.schedule_interval(self.update_bpm, 1)
                    self.is_detecting = True
                else:
                    self.status_label.text = '设备启动失败'
            except Exception as e:
                self.status_label.text = f'错误: {str(e)}'
        else:
            # 停止检测
            if self.detector:
                self.detector.stop()
            Clock.unschedule(self.update_bpm)
            self.bpm_label.text = '当前BPM: --'
            self.start_stop_button.text = '开始检测'
            self.start_stop_button.background_color = (0.3, 0.6, 0.3, 1)
            self.status_label.text = '等待开始...'
            self.is_detecting = False
    
    def update_bpm(self, dt):
        if self.detector and self.detector.current_bpm:
            self.bpm_label.text = f'当前BPM: {self.detector.current_bpm:.1f}'
            return True
        return True
    
    def on_stop(self):
        """应用程序停止时的清理工作"""
        if self.detector:
            self.detector.stop()

if __name__ == '__main__':
    Window.clearcolor = (0.9, 0.9, 0.9, 1)  # 设置背景色为浅灰色
    BPMApp().run()
