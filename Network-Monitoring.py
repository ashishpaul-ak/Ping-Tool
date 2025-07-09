import sys
import os
import time
import socket
import subprocess
import warnings
from datetime import datetime

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Redirect stderr to suppress OpenGL warnings
import io
import contextlib
stderr_redirect = io.StringIO()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTabWidget,
    QGridLayout, QFileDialog, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QMessageBox, QMenu,
    QComboBox, QProgressBar, QStyle, QAction, QStyledItemDelegate
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont

# Suppress OpenGL warnings during pyqtgraph import
with contextlib.redirect_stderr(stderr_redirect):
    import pyqtgraph as pg
    import numpy as np
    from pyqtgraph import DateAxisItem

# --- Optimized Batch Ping Worker ---
class BatchPingWorker(QThread):
    ping_results = pyqtSignal(dict)
    def __init__(self, ips, interval, batch_size=20):  # Increased batch size
        super().__init__()
        self.ips = ips
        self.interval = interval
        self.batch_size = batch_size
        self.running = True
        self.paused = False
        self._ping_cache = {}  # Cache for quick responses
        
    def run(self):
        while self.running:
            if not self.paused:
                try:
                    results = self._fast_batch_ping()
                    if self.running:  # Check again before emitting
                        self.ping_results.emit(results)
                except RuntimeError:
                    # Interpreter is shutting down, exit gracefully
                    break
            time.sleep(self.interval)
    
    def _fast_batch_ping(self):
        results = {}
        # Use larger batches for better efficiency
        for i in range(0, len(self.ips), self.batch_size):
            batch = self.ips[i:i + self.batch_size]
            batch_results = self._ping_batch_optimized(batch)
            results.update(batch_results)
        return results
    
    def _ping_batch_optimized(self, ips):
        results = {}
        import concurrent.futures
        
        # Check if we should continue
        if not self.running:
            return results
            
        # Use more workers for better parallelism
        max_workers = min(len(ips), 8)  # Increased workers
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all pings at once
                future_to_ip = {executor.submit(self._ping_single_optimized, ip): ip for ip in ips}
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_ip, timeout=5):
                    if not self.running:  # Check if we should stop
                        break
                    ip = future_to_ip[future]
                    try:
                        result = future.result(timeout=1)  # Reduced timeout
                        results[ip] = result
                    except Exception:
                        results[ip] = {'status': 'dead', 'rtt': 0, 'time': datetime.now()}
        except RuntimeError:
            # Interpreter is shutting down, return empty results
            pass
        return results
    
    def _ping_single_optimized(self, ip):
        result = {'status': 'dead', 'rtt': 0, 'time': datetime.now()}
        try:
            if sys.platform == 'win32':
                # Use faster ping parameters
                cmd = ['ping', '-n', '1', '-w', '300', ip]  # Reduced timeout further
            else:
                cmd = ['ping', '-c', '1', '-W', '1', ip]
            
            # Use subprocess.run with timeout for better control
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5)
            
            if process.returncode == 0 and ('time=' in process.stdout or 'time<' in process.stdout):
                result['status'] = 'alive'
                for line in process.stdout.splitlines():
                    if 'time=' in line or 'time<' in line:
                        try:
                            time_part = line.split('time=')[-1].split('ms')[0].strip()
                            result['rtt'] = float(time_part.replace('<', '0.5'))
                            break  # Exit after first match
                        except ValueError:
                            continue
        except Exception:
            pass
        return result
    
    def pause(self):
        self.paused = True
    def resume(self):
        self.paused = False
    def stop(self):
        self.running = False
        self.paused = True  # Also pause to prevent new operations

# --- Hostname Resolver ---
class HostnameResolver(QThread):
    hostname_resolved = pyqtSignal(str, str)
    def __init__(self, ip):
        super().__init__()
        self.ip = ip
    def run(self):
        try:
            hostname = socket.getfqdn(self.ip)
            if hostname != self.ip:
                self.hostname_resolved.emit(self.ip, hostname)
            else:
                self.hostname_resolved.emit(self.ip, "")
        except Exception:
            self.hostname_resolved.emit(self.ip, "")

# --- Graph Window ---
class PingGraphWindow(QMainWindow):
    def __init__(self, ip, dark_mode, timeframe):
        super().__init__()
        self.ip = ip
        self.timeframe = timeframe
        self.setWindowTitle(self._make_title())
        self.setGeometry(300, 200, 600, 400)
        self.graph_widget = MiniPingGraph(ip=ip, dark_mode=dark_mode)
        self.graph_widget.set_time_scale(timeframe)
        self.setCentralWidget(self.graph_widget)
        # Add Close button
        close_btn = QPushButton("Close", self)
        close_btn.setToolTip("Close this graph window")
        close_btn.clicked.connect(self.close)
        close_btn.setFixedWidth(80)
        self.graph_widget.layout.addWidget(close_btn)
        self.set_theme(dark_mode)
        self._update_graph_timer = None
        self.is_paused = False

    def _make_title(self):
        tf = self.timeframe
        if tf < 60:
            tf_str = f"{tf} sec"
        else:
            tf_str = f"{tf//60} min"
        return f"Ping Graph - {self.ip} [{tf_str}]"
    def set_theme(self, dark_mode):
        print(f"[PingGraphWindow] set_theme called for {self.windowTitle()} dark_mode={dark_mode}")
        self.graph_widget.set_theme(dark_mode)
        # Set the window background color to match dark/light mode
        if dark_mode:
            self.setStyleSheet("background-color: #232629; color: #fff;")
        else:
            self.setStyleSheet("background-color: #fff; color: #000;")
    def set_time_scale(self, seconds):
        self.time_scale = seconds
        self.graph_widget.set_time_scale(seconds)
        self.setWindowTitle(self._make_title())
    def update_data(self, result):
        if not self.is_paused:
            self.graph_widget.update_data(result)
    def pause_updates(self):
        self.is_paused = True
        if hasattr(self, '_update_graph_timer') and self._update_graph_timer:
            self._update_graph_timer.stop()
    def resume_updates(self):
        self.is_paused = False
        if hasattr(self, '_update_graph_timer') and self._update_graph_timer:
            self._update_graph_timer.start()

# --- Mini Graph Widget ---
class MiniPingGraph(QWidget):
    def __init__(self, ip=None, dark_mode=True):
        super().__init__()
        self.ip = ip
        self.dark_mode = dark_mode
        self.max_points = 200  # Reduced for better performance
        self.time_scale = 300
        self.data_index = 0
        self.last_update = 0
        self.update_interval = 2.0  # Increase update interval for better performance
        self.packet_sent = 0
        self.packet_recv = 0
        self.packet_lost = 0
        self.history = []  # Store (timestamp, rtt, status)
        self.max_history_size = 3000 # Reduced for better performance
        self.bar_item = None
        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        # Controls
        self.copy_btn = QPushButton("Copy Stats")
        self.copy_btn.setToolTip("Copy summary stats to clipboard")
        self.copy_btn.clicked.connect(self.copy_stats)
        self.save_btn = QPushButton("Save as PNG")
        self.save_btn.setToolTip("Save the graph as a PNG image")
        self.save_btn.clicked.connect(self.save_png)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.copy_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)
        # Summary
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-size: 11px; padding: 2px 0 2px 0;")
        self.summary_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.summary_label)
        # Separator
        self.separator = QLabel()
        self.separator.setFixedHeight(1)
        self.layout.addWidget(self.separator)
        # Plot with DateAxis
        self.axis = DateAxisItem(orientation='bottom')
        self.plot = pg.PlotWidget(axisItems={'bottom': self.axis}, background=None)
        self.plot.setMinimumHeight(300)
        self.plot.setMinimumWidth(500)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.showAxis('left')
        self.plot.showAxis('bottom')
        self.plot.setLabel('left', 'RTT (ms)')
        self.plot.setLabel('bottom', 'Time')
        # Move legend above the plot
        self.legend = pg.LegendItem(offset=(0, -30))
        self.legend.setParentItem(self.plot.graphicsItem())
        self.legend.anchor((0, 1), (0, 0), offset=(0, -10))
        self.layout.addWidget(self.plot)
        # Plot curves for min, max, avg
        self.curve_min = self.plot.plot([], [], pen=pg.mkPen('blue', width=2), name='Min RTT')
        self.curve_max = self.plot.plot([], [], pen=pg.mkPen((128,0,255), width=2), name='Max RTT')
        self.curve_avg = self.plot.plot([], [], pen=pg.mkPen('orange', width=2), name='Avg RTT (Reachable)')
        self.curve_avg_lost = self.plot.plot([], [], pen=pg.mkPen('red', width=2, style=Qt.DashLine), name='Avg RTT (Lost)')
        self.legend.clear()
        self.legend.addItem(self.curve_min, 'Min RTT')
        self.legend.addItem(self.curve_max, 'Max RTT')
        self.legend.addItem(self.curve_avg, 'Avg RTT (Reachable)')
        self.legend.addItem(self.curve_avg_lost, 'Avg RTT (Lost)')
        self.loss_regions = []
        self.loss_lines = []
        self.loss_texts = []
        self.all_loss_regions = [] # Store all detected loss regions permanently
        self.max_avg_marker = None
        self.min_avg_marker = None
        self.max_avg_text = None
        self.min_avg_text = None
        self.latency_bands = []
        self.set_theme(self.dark_mode)
    def set_time_scale(self, seconds):
        self.time_scale = seconds
        self.max_points = seconds
        self.bar_item = None
        self.update_graph_from_history()
        # Update window title if in a QMainWindow
        if self.parent() and hasattr(self.parent(), 'setWindowTitle'):
            tf = self.time_scale
            if tf < 60:
                tf_str = f"{tf} sec"
            else:
                tf_str = f"{tf//60} min"
            self.parent().setWindowTitle(f"Ping Graph - {self.ip} [{tf_str}]")
    def set_theme(self, dark_mode):
        self.dark_mode = dark_mode
        if dark_mode:
            bg = '#232629'
            fg = '#fff'
            sep_color = '#444'
            self.summary_label.setStyleSheet("color: #fff; font-size: 11px; padding: 2px 0 2px 0;")
            self.setStyleSheet("background-color: #232629; color: #fff;")
        else:
            bg = '#fff'
            fg = '#000'
            sep_color = '#bbb'
            self.summary_label.setStyleSheet("color: #000; font-size: 11px; padding: 2px 0 2px 0;")
            self.setStyleSheet("background-color: #fff; color: #000;")
        self.plot.setBackground(bg)
        self.plot.getAxis('left').setTextPen(fg)
        self.plot.getAxis('bottom').setTextPen(fg)
        self.plot.getAxis('left').setPen(fg)
        self.plot.getAxis('bottom').setPen(fg)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.separator.setStyleSheet(f"background: {sep_color}; margin: 4px 0 4px 0;")
        self.plot.setTitle(f"Round-Trip-Time (ms) - {self.ip} - {self.time_scale//60} Minutes", color=fg, size='12pt')
        self.update_graph_from_history()
    def update_data(self, ping_result):
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        self.last_update = current_time
        status = ping_result['status']
        rtt = ping_result['rtt']
        self.packet_sent += 1
        if status == 'alive':
            self.packet_recv += 1
            self.history.append((current_time, rtt, status))
        else:
            self.packet_lost += 1
            self.history.append((current_time, 0, status))
        # Prune history to last 5,000 points (reduced for better memory management)
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
        self.update_graph_from_history()
    def update_graph_from_history(self):
        # Only show data within the current timeframe
        now = time.time()
        window_start = now - self.time_scale
        
        # Optimize filtering with list comprehension
        filtered = [(t, rtt, status) for (t, rtt, status) in self.history if t >= window_start]
        
        if not filtered:
            # Clear all data efficiently
            self.curve_min.setData([], [])
            self.curve_max.setData([], [])
            self.curve_avg.setData([], [])
            self.curve_avg_lost.setData([], [])
            
            # Batch remove items for better performance
            for items in [self.loss_regions, self.loss_lines, self.loss_texts, self.latency_bands]:
                for item in items:
                    self.plot.removeItem(item)
                items.clear()
            
            self.summary_label.setText("")
            return
        x_vals = np.array([t for (t, _, _) in filtered])
        y_vals = np.array([rtt for (_, rtt, _) in filtered])
        # Use nan for lost pings (RTT == 0) so they don't plot
        y_vals_for_plot = np.where(y_vals > 0, y_vals, np.nan)
        rtts = y_vals[y_vals > 0] # Keep rtts with values > 0 for summary stats
        # Store full history loss regions (calculated once or updated incrementally)
        self._calculate_all_loss_regions()
        # Filter loss regions to display based on current timeframe
        window_start = now - self.time_scale
        display_loss_regions = [(s, e, c) for (s, e, c) in self.all_loss_regions if max(s, window_start) < min(e, now)]
        # Optimized binning for better performance
        n_bins = min(50, max(5, int(self.time_scale // 4)))  # Reduced bins for speed
        bin_size = max(1, int(self.time_scale / n_bins))
        min_rtts = []
        max_rtts = []
        avg_rtts = []
        avg_rtts_lost = []
        times = []
        if len(x_vals) > 0:
            t0 = x_vals[0]
            t1 = x_vals[-1]
            bins = np.arange(t0, t1 + bin_size, bin_size)
            inds = np.digitize(x_vals, bins)
            for i in range(1, len(bins)):
                bin_rtts = y_vals_for_plot[inds == i]
                bin_times = x_vals[inds == i]
                if len(bin_rtts) > 0:
                    min_rtts.append(np.min(bin_rtts))
                    max_rtts.append(np.max(bin_rtts))
                    avg = np.mean(bin_rtts)
                    times.append(bins[i-1])
                    if np.all(bin_rtts == 0):
                        avg_rtts_lost.append(avg)
                        avg_rtts.append(np.nan)
                    else:
                        avg_rtts.append(avg)
                        avg_rtts_lost.append(np.nan)
        # Plot min, max, avg
        # Use different pens based on dark/light mode and line type
        min_pen = pg.mkPen('blue', width=1, style=Qt.DotLine) if self.dark_mode else pg.mkPen(QColor(0, 0, 150), width=1, style=Qt.DotLine)
        max_pen = pg.mkPen((128,0,255), width=1, style=Qt.DotLine) if self.dark_mode else pg.mkPen(QColor(150, 0, 150), width=1, style=Qt.DotLine)
        avg_reachable_pen = pg.mkPen('orange', width=3)
        avg_lost_pen = pg.mkPen('red', width=3, style=Qt.DashLine)

        self.curve_min.setData(times, min_rtts, pen=min_pen)
        self.curve_max.setData(times, max_rtts, pen=max_pen)
        self.curve_avg.setData(times, avg_rtts, pen=avg_reachable_pen)
        self.curve_avg_lost.setData(times, avg_rtts_lost, pen=avg_lost_pen)
        # Clear old loss region items before redrawing
        for item_list in [self.loss_regions, self.loss_lines, self.loss_texts]:
            for item in item_list:
                self.plot.removeItem(item)
            item_list.clear()
        # Add loss regions (dark red background), vertical lines, and text
        for start, end, count in self.all_loss_regions:
            # Check if the loss region is at least partially within the current view timeframe
            if max(start, window_start) < min(end, now) and count > 4:
                # Add loss region background
                region = pg.LinearRegionItem([start, end], orientation=pg.LinearRegionItem.Vertical)
                region.setBrush(pg.mkBrush(150, 0, 0, 80))
                region.setMovable(False)
                region.setZValue(-5) # Ensure region is behind lines
                self.plot.addItem(region)
                self.loss_regions.append(region)
                # Add vertical lines at start and end of loss
                vline1 = pg.InfiniteLine(pos=start, angle=90, pen=pg.mkPen('red', width=2, style=Qt.DashLine))
                vline2 = pg.InfiniteLine(pos=end, angle=90, pen=pg.mkPen('red', width=2, style=Qt.DashLine))
                self.plot.addItem(vline1)
                self.plot.addItem(vline2)
                self.loss_lines.extend([vline1, vline2])
                # Annotate number of lost pings, downtime, and timestamps
                start_dt = datetime.fromtimestamp(start).strftime('%H:%M:%S')
                end_dt = datetime.fromtimestamp(end).strftime('%H:%M:%S')
                downtime = end - start
                if downtime < 60:
                    downtime_str = f"{downtime:.1f}s"
                else:
                    mins = int(downtime // 60)
                    secs = downtime % 60
                    downtime_str = f"{mins}m {secs:.1f}s"
                # Position text below the x-axis and within the loss region bounds
                text_pos_x = start + (end - start) / 2  # Center horizontally within the region
                # Ensure text is horizontally within the current view before adding
                if max(start, self.plot.viewRange()[0][0]) < min(end, self.plot.viewRange()[0][1]):
                    # Create text item
                    # Use a simple TextItem with HTML formatting
                    html_text = f"<div style='text-align:center; color:red;'><b>Lost: {count}</b><br>Downtime: {downtime_str}<br>{start_dt} - {end_dt}</div>"

                    # Position text slightly below 0 RTT
                    text_pos_y = 0 # Will be adjusted slightly below 0 by the anchor point

                    # Use pg.TextItem
                    txt = pg.TextItem(html=html_text, anchor=(0.5, 0)) # anchor=(0.5, 0) puts the bottom-center at the specified pos
                    txt.setPos(text_pos_x, text_pos_y)

                    self.plot.addItem(txt)
                    self.loss_texts.append(txt)
        # Remove old latency bands
        for band in self.latency_bands:
            self.plot.removeItem(band)
        self.latency_bands = []
        # Add latency bands (green <100ms, yellow 100-200ms, red >200ms)
        y_ranges = [(0, 100, (0, 255, 0, 40)), (100, 200, (255, 255, 0, 40)), (200, 10000, (255, 0, 0, 40))]
        for y0, y1, color in y_ranges:
            band = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Horizontal)
            band.setRegion([y0, y1])
            band.setBrush(pg.mkBrush(color))
            band.setMovable(False)
            band.setZValue(-10)
            self.plot.addItem(band)
            self.latency_bands.append(band)
        # Set y and x range
        if len(rtts) > 0:
            max_y = max(np.max(rtts) * 1.2, 10)
            self.plot.setYRange(-50, max_y)
        if len(times) > 0:
            self.plot.setXRange(times[0], times[-1])
        now_str = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        if self.time_scale >= 60:
            title = f"Round-Trip-Time (ms) - {self.ip} - {self.time_scale//60} Minutes - {now_str}"
        else:
            title = f"Round-Trip-Time (ms) - {self.ip} - {self.time_scale} Seconds - {now_str}"
        self.plot.setTitle(title, color='#fff' if self.dark_mode else '#000', size='12pt')
        min_rtt = np.min(rtts) if len(rtts) > 0 else 0
        avg_rtt = np.mean(rtts) if len(rtts) > 0 else 0
        max_rtt = np.max(rtts) if len(rtts) > 0 else 0
        cur_rtt = rtts[-1] if len(rtts) > 0 else 0
        summary = f"<b>Min</b> {min_rtt:.2f} ms &nbsp; <b>Avg</b> {avg_rtt:.2f} ms &nbsp; <b>Max</b> {max_rtt:.2f} ms &nbsp; <b>Cur</b> {cur_rtt:.2f} ms "
        summary += f"<span style='color:#888;'>&nbsp;|&nbsp;</span> Packets: Sent={self.packet_sent}, Recv={self.packet_recv}, Lost={self.packet_lost}"
        self.summary_label.setText(summary)
        # Remove old avg RTT markers/texts
        if self.max_avg_marker:
            self.plot.removeItem(self.max_avg_marker)
            self.max_avg_marker = None
        if self.min_avg_marker:
            self.plot.removeItem(self.min_avg_marker)
            self.min_avg_marker = None
        if self.max_avg_text:
            self.plot.removeItem(self.max_avg_text)
            self.max_avg_text = None
        if self.min_avg_text:
            self.plot.removeItem(self.min_avg_text)
            self.min_avg_text = None
        # Mark and label highest/lowest avg RTT points
        valid_avg = [(t, v) for t, v in zip(times, avg_rtts) if not np.isnan(v)]
        if valid_avg:
            max_idx = np.argmax([v for _, v in valid_avg])
            min_idx = np.argmin([v for _, v in valid_avg])
            max_t, max_v = valid_avg[max_idx]
            min_t, min_v = valid_avg[min_idx]
            self.max_avg_marker = pg.ScatterPlotItem([max_t], [max_v], symbol='o', size=12, brush=pg.mkBrush('red'))
            self.min_avg_marker = pg.ScatterPlotItem([min_t], [min_v], symbol='o', size=12, brush=pg.mkBrush('green'))
            self.plot.addItem(self.max_avg_marker)
            self.plot.addItem(self.min_avg_marker)
            self.max_avg_text = pg.TextItem(f"Max Avg: {max_v:.1f} ms", color='red', anchor=(0,1))
            self.max_avg_text.setPos(max_t, max_v)
            self.plot.addItem(self.max_avg_text)
            self.min_avg_text = pg.TextItem(f"Min Avg: {min_v:.1f} ms", color='green', anchor=(0,1))
            self.min_avg_text.setPos(min_t, min_v)
            self.plot.addItem(self.min_avg_text)
    def copy_stats(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.summary_label.text().replace('<br>', '\n').replace('&nbsp;', ' '))
    def save_png(self):
        exporter = pg.exporters.ImageExporter(self.plot.plotItem)
        filename, _ = QFileDialog.getSaveFileName(self, "Save Graph as PNG", f"ping_graph_{self.ip}.png", "PNG Files (*.png)")
        if filename:
            exporter.export(filename)
    def _calculate_all_loss_regions(self):
        # This method calculates contiguous loss regions from the full history
        self.all_loss_regions = []
        loss_start_time = None
        loss_count = 0
        # Sort history by timestamp to ensure correct order
        sorted_history = sorted(self.history, key=lambda item: item[0])

        for i, (timestamp, rtt, status) in enumerate(sorted_history):
            if status == 'dead' or rtt == 0:
                if loss_start_time is None:
                    loss_start_time = timestamp
                loss_count += 1
            elif loss_start_time is not None:
                # Loss ended
                loss_end_time = timestamp
                self.all_loss_regions.append((loss_start_time, loss_end_time, loss_count))
                loss_start_time = None
                loss_count = 0
        # Handle loss extending to the end of history
        if loss_start_time is not None:
             # Need to estimate the end time for the last loss segment
             # Assuming the last ping interval is representative
            estimated_end_time = sorted_history[-1][0] + (sorted_history[-1][0] - sorted_history[-2][0] if len(sorted_history) > 1 else 1)
            self.all_loss_regions.append((loss_start_time, estimated_end_time, loss_count))

# --- Status Icon Delegate for Traceroute Table ---
class StatusIconDelegate(QStyledItemDelegate):
    def __init__(self):
        super().__init__()
    
    def paint(self, painter, option, index):
        # Simple delegate that just paints the text
        super().paint(painter, option, index)

# --- Hop Latency Widget for Traceroute Table ---
class HopLatencyWidget(QWidget):
    def __init__(self, dark_mode=True):
        super().__init__()
        self.dark_mode = dark_mode
        self.history = [] # Store (timestamp, rtt, status)
        self.max_history_points = 60 # Keep last 60 seconds of data (or pings)
        self.setMinimumHeight(40)
        self.setMouseTracking(True) # Enable mouse tracking for potential future tooltips
        self.tooltip_text = ""
        self._last_update_time = 0
        self._update_interval = 0.5 # Throttle paint updates

        # Store latest RTT stats
        self.min_rtt = -1
        self.max_rtt = -1
        self.current_rtt = -1
        self.ping_status = 'unknown'

        # Define latency band thresholds and colors
        self.threshold_green = 100
        self.threshold_yellow = 200
        self.color_green = QColor(0, 255, 0, 40)
        self.color_yellow = QColor(255, 255, 0, 40)
        self.color_red = QColor(255, 0, 0, 40)

    def set_theme(self, dark_mode):
        self.dark_mode = dark_mode
        self.update() # Redraw with new theme

    def update_data(self, data):
        # data should be a dictionary with 'status', 'rtt', 'min_rtt', 'max_rtt', 'avg_rtt'
        self.ping_status = data.get('status', 'unknown')
        self.current_rtt = data.get('rtt', -1)
        self.min_rtt = data.get('min_rtt', -1)
        self.max_rtt = data.get('max_rtt', -1)
        # self.avg_rtt = data.get('avg_rtt', -1) # Not used for drawing the bar

        # Add data to history for the simplified line graph (if we decide to keep it)
        current_time = time.time()
        self.history.append((current_time, self.current_rtt, self.ping_status))
        if len(self.history) > self.max_history_points:
            self.history = self.history[-self.max_history_points:]

        # Request repaint, but throttle updates
        if current_time - self._last_update_time > self._update_interval:
            self._last_update_time = current_time
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        padding = 2

        # Draw background latency bands
        # Green: <100ms, Yellow: 100-200ms, Red: >200ms
        # Dynamic scaling based on max observed RTT, but with a minimum to show bands
        max_observed_rtt = max(self.max_rtt, self.current_rtt, 0) if self.max_rtt != -1 else max(self.current_rtt, 0)
        # Ensure a minimum scale to show the bands clearly even with low RTT or no data yet
        min_display_scale = self.threshold_yellow * 1.5 # Ensure yellow band is visible
        max_display_rtt = max(max_observed_rtt * 1.2, min_display_scale, 100.0) # Scale up a bit, ensure min 100

        # Calculate band heights based on a relative scale up to max_display_rtt
        scale_factor = (height - 2 * padding) / max_display_rtt if max_display_rtt > 0 else 0

        # Calculate Y coordinates for the band boundaries (inverted for drawing)
        y_at_0 = height - padding
        # Clamp threshold Y coordinates to be within the drawable area (padding to height-padding)
        y_at_green_thresh = max(padding, height - padding - min(self.threshold_green * scale_factor, height - 2 * padding))
        y_at_yellow_thresh = max(padding, height - padding - min(self.threshold_yellow * scale_factor, height - 2 * padding))
        y_at_max_display = padding

        # Draw bands
        # Green band (<100ms): from y_at_green_thresh down to y_at_0
        painter.fillRect(padding, int(y_at_green_thresh), width - 2 * padding, int(y_at_0 - y_at_green_thresh), self.color_green)
        # Yellow band (100-200ms): from y_at_yellow_thresh down to y_at_green_thresh
        painter.fillRect(padding, int(y_at_yellow_thresh), width - 2 * padding, int(y_at_green_thresh - y_at_yellow_thresh), self.color_yellow)
        # Red band (>200ms): from padding down to y_at_yellow_thresh
        painter.fillRect(padding, padding, width - 2 * padding, int(y_at_yellow_thresh - padding), self.color_red)

        # Draw the horizontal latency bar and current RTT marker
        if self.ping_status == 'alive' and self.min_rtt != -1 and self.max_rtt != -1 and self.current_rtt != -1 and scale_factor > 0:
            # Scale RTT values to y-coordinates, clamp to max_display_rtt
            y_min = height - padding - min(self.min_rtt, max_display_rtt) * scale_factor
            y_max = height - padding - min(self.max_rtt, max_display_rtt) * scale_factor
            y_current = height - padding - min(self.current_rtt, max_display_rtt) * scale_factor

            # Ensure y_min is above y_max for drawing the rectangle
            rect_y_start = min(y_min, y_max)
            rect_height = max(y_min, y_max) - rect_y_start

            # Draw the horizontal bar (Min to Max RTT range)
            painter.setPen(Qt.NoPen)
            # Use a color that stands out against the bands, maybe a darker shade or white outline
            bar_color = QColor(50, 100, 200, 150) # A shade of blue with transparency
            painter.setBrush(QBrush(bar_color))
            painter.drawRect(padding, int(rect_y_start), width - 2 * padding, int(rect_height))

            # Draw marker for Current RTT
            marker_color = QColor(0, 0, 0) if self.dark_mode else QColor(255, 255, 255) # Black or white marker
            painter.setPen(QPen(marker_color, 1))
            painter.setBrush(QBrush(marker_color))
            painter.drawEllipse(int(width / 2) - 3, int(y_current) - 3, 6, 6) # Draw a circle in the middle horizontally

        # Draw a red X if status is dead
        elif self.ping_status == 'dead':
             painter.setPen(QPen(Qt.red, 2))
             x_center = width / 2
             y_center = height / 2
             size = 8
             painter.drawLine(x_center - size, y_center - size, x_center + size, y_center + size)
             painter.drawLine(x_center + size, y_center - size, x_center - size, y_center + size)

    # Optional: Add tooltip on hover for the last RTT
    # def mouseMoveEvent(self, event):
    #     # Check if mouse is over the last data point or generally in the widget
    #     if self.history:
    #         last_rtt = self.history[-1][1]
    #         last_status = self.history[-1][2]
    #         if last_status == 'alive':
    #             tooltip_text = f"Latest RTT: {last_rtt:.1f} ms"
    #         else:
    #             tooltip_text = "Latest Ping: Lost"
    #         if tooltip_text != self.tooltip_text:
    #             self.setToolTip(tooltip_text)
    #             self.tooltip_text = tooltip_text
    #     else:
    #          self.setToolTip("")
    #          self.tooltip_text = ""

# --- Main Window ---
class NetworkPingMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network Ping Monitor")
        self.setGeometry(100, 100, 1200, 800)
        self.dark_mode = True
        self.apply_theme()
        self.batch_ping_worker = None
        self.hostname_resolvers = {}
        self.ping_data = {}
        self.graph_windows = {}
        self.graph_timeframe_minutes = 5
        self.is_monitoring = False
        self.is_stopping = False  # Add flag for stopping state
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        self.ping_tab = QWidget()
        self.tabs.addTab(self.ping_tab, "Ping Monitor")
        self.setup_ping_tab()
        self.statusBar().showMessage("Ready")
        self.ping_interval = 1
        self.warning_threshold = 100
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(2000)  # Reduce UI updates to every 2 seconds for better performance
        self.stop_check_timer = QTimer()  # Add timer for checking stop status
        self.stop_check_timer.timeout.connect(self.check_stop_status)
        self.stop_check_timer.setInterval(200)  # Check every 200ms
    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; }
                QGroupBox { border: 1px solid #3a3a3a; border-radius: 5px; margin-top: 1ex; padding: 10px; }
                QGroupBox::title { color: #ffffff; }
                QLineEdit, QSpinBox { background-color: #3a3a3a; color: #ffffff; border: 1px solid #4a4a4a; padding: 5px; }
                QPushButton { background-color: #3a3a3a; color: #ffffff; border: 1px solid #4a4a4a; padding: 5px 15px; border-radius: 3px; }
                QPushButton:hover { background-color: #404040; }
                QPushButton:disabled { background-color: #2b2b2b; color: #666666; }
                QTableWidget { background-color: #18191a; color: #ffffff; gridline-color: #3a3a3a; selection-background-color: #404040; alternate-background-color: #232629; }
                QHeaderView::section { background-color: #3a3a3a; color: #ffffff; padding: 4px; border: 1px solid #4a4a4a; }
                QComboBox { background-color: #3a3a3a; color: #ffffff; border: 1px solid #4a4a4a; padding: 5px; }
                QTabWidget::pane { border: 1px solid #3a3a3a; background: #2b2b2b; }
                QTabBar::tab { background: #3a3a3a; color: #ffffff; border: 1px solid #3a3a3a; padding: 8px 12px; }
                QTabBar::tab:selected { background: #2b2b2b; border-bottom-color: #2b2b2b; }
                QTabBar::tab:hover { background: #404040; }
            """)
            if hasattr(self, 'results_table'):
                self.results_table.setStyleSheet(
                    "QTableWidget { alternate-background-color: #232629; background-color: #18191a; color: #fff; }"
                )
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #ffffff; color: #000000; }
                QGroupBox { border: 1px solid #cccccc; border-radius: 5px; margin-top: 1ex; padding: 10px; }
                QGroupBox::title { color: #000000; }
                QLineEdit, QSpinBox { background-color: #ffffff; color: #000000; border: 1px solid #cccccc; padding: 5px; }
                QPushButton { background-color: #f0f0f0; color: #000000; border: 1px solid #cccccc; padding: 5px 15px; border-radius: 3px; }
                QPushButton:hover { background-color: #e0e0e0; }
                QPushButton:disabled { background-color: #f8f8f8; color: #888888; }
                QTableWidget { background-color: #ffffff; color: #000000; gridline-color: #cccccc; selection-background-color: #e0e0e0; alternate-background-color: #f7f7f7; }
                QHeaderView::section { background-color: #f0f0f0; color: #000000; padding: 4px; border: 1px solid #cccccc; }
                QComboBox { background-color: #ffffff; color: #000000; border: 1px solid #cccccc; padding: 5px; }
                QTabWidget::pane { border: 1px solid #cccccc; background: #ffffff; }
                QTabBar::tab { background: #f0f0f0; color: #000000; border: 1px solid #cccccc; padding: 8px 12px; }
                QTabBar::tab:selected { background: #ffffff; border-bottom-color: #ffffff; }
                QTabBar::tab:hover { background: #e0e0e0; }
            """)
            if hasattr(self, 'results_table'):
                self.results_table.setStyleSheet("")
    # --- Ping Tab ---
    def setup_ping_tab(self):
        layout = QVBoxLayout()
        self.ping_tab.setLayout(layout)
        heading = QLabel("<b>Ping Monitor</b>")
        heading.setStyleSheet("font-size: 16px; margin-bottom: 8px;")
        layout.addWidget(heading)
        input_group = QGroupBox("IP Address Input")
        input_layout = QGridLayout()
        input_group.setLayout(input_layout)
        input_layout.addWidget(QLabel("IP address range:"), 0, 0)
        self.ip1 = QSpinBox(); self.ip1.setRange(0, 255); self.ip1.setValue(192)
        self.ip2 = QSpinBox(); self.ip2.setRange(0, 255); self.ip2.setValue(168)
        self.ip3 = QSpinBox(); self.ip3.setRange(0, 255); self.ip3.setValue(1)
        self.ip4_start = QSpinBox(); self.ip4_start.setRange(1, 254); self.ip4_start.setValue(1)
        self.ip4_end = QSpinBox(); self.ip4_end.setRange(1, 254); self.ip4_end.setValue(254)
        ip_range_layout = QHBoxLayout()
        ip_range_layout.addWidget(self.ip1)
        ip_range_layout.addWidget(QLabel("."))
        ip_range_layout.addWidget(self.ip2)
        ip_range_layout.addWidget(QLabel("."))
        ip_range_layout.addWidget(self.ip3)
        ip_range_layout.addWidget(QLabel("."))
        ip_range_layout.addWidget(self.ip4_start)
        ip_range_layout.addWidget(QLabel("→"))
        ip_range_layout.addWidget(self.ip4_end)
        input_layout.addLayout(ip_range_layout, 0, 1, 1, 2)
        input_layout.addWidget(QLabel("OR enter IPs (comma or space separated):"), 1, 0)
        self.ip_input = QLineEdit(); self.ip_input.setPlaceholderText("e.g. 8.8.8.8, 1.1.1.1")
        input_layout.addWidget(self.ip_input, 1, 1, 1, 2)
        input_layout.addWidget(QLabel("OR upload IP list file:"), 2, 0)
        self.file_btn = QPushButton("Browse...")
        self.file_btn.clicked.connect(self.browse_file)
        input_layout.addWidget(self.file_btn, 2, 1)
        self.file_label = QLabel("No file selected")
        input_layout.addWidget(self.file_label, 2, 2)
        input_layout.addWidget(QLabel("Ping Interval (seconds):"), 3, 0)
        self.interval_spinner = QSpinBox(); self.interval_spinner.setRange(1, 60); self.interval_spinner.setValue(1)
        self.interval_spinner.valueChanged.connect(self.set_ping_interval)
        input_layout.addWidget(self.interval_spinner, 3, 1)
        input_layout.addWidget(QLabel("Warning Threshold (ms):"), 4, 0)
        self.threshold_spinner = QSpinBox(); self.threshold_spinner.setRange(10, 1000); self.threshold_spinner.setValue(100)
        self.threshold_spinner.valueChanged.connect(self.set_warning_threshold)
        input_layout.addWidget(self.threshold_spinner, 4, 1)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Alive", "Dead"])
        self.filter_combo.setCurrentIndex(1) # Default to Alive
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_combo)
        self.theme_btn = QPushButton("Toggle Dark Mode")
        self.theme_btn.clicked.connect(self.toggle_theme)
        filter_layout.addWidget(self.theme_btn)
        timeframe_layout = QHBoxLayout()
        timeframe_layout.addWidget(QLabel("Graph Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems([
            "30 sec", "1 min", "2 min", "5 min", "10 min", "15 min", "30 min", "60 min"
        ])
        self.timeframe_combo.setCurrentIndex(3)  # Default to 5 min
        self.timeframe_combo.currentTextChanged.connect(self.update_graph_timeframe)
        self.timeframe_combo.setToolTip("Select the time window for graph data")
        timeframe_layout.addWidget(self.timeframe_combo)
        timeframe_layout.addStretch()
        filter_layout.addLayout(timeframe_layout)
        filter_layout.addStretch()
        input_layout.addLayout(filter_layout, 5, 0, 1, 3)
        btn_layout = QHBoxLayout()
        # Replace Start/Stop buttons with a single Toggle button
        self.toggle_monitor_btn = QPushButton("Start Monitoring")
        self.toggle_monitor_btn.setToolTip("Start or pause monitoring.")
        self.toggle_monitor_btn.clicked.connect(self.toggle_monitoring)
        btn_layout.addWidget(self.toggle_monitor_btn)
        # self.start_btn = QPushButton("Start Monitoring") # Removed
        # self.start_btn.clicked.connect(self.start_monitoring) # Removed
        # self.start_btn.setToolTip("Start monitoring the specified IP addresses.") # Removed
        # btn_layout.addWidget(self.start_btn) # Removed
        # self.stop_btn = QPushButton("Stop Monitoring") # Removed
        # self.stop_btn.clicked.connect(self.stop_monitoring) # Removed
        # self.stop_btn.setToolTip("Stop monitoring the IP addresses.") # Removed
        # self.stop_btn.setEnabled(False) # Removed
        # btn_layout.addWidget(self.stop_btn) # Removed
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_monitoring)
        self.clear_btn.setToolTip("Clear all monitoring data.")
        btn_layout.addWidget(self.clear_btn)
        # Add Open/Close All Graphs buttons
        self.open_all_graphs_btn = QPushButton("Open All Graphs")
        self.open_all_graphs_btn.setToolTip("Open graph windows for all visible IPs.")
        self.open_all_graphs_btn.clicked.connect(self.open_all_graphs)
        btn_layout.addWidget(self.open_all_graphs_btn)
        self.close_all_graphs_btn = QPushButton("Close All Graphs")
        self.close_all_graphs_btn.setToolTip("Close all opened graph windows.")
        self.close_all_graphs_btn.clicked.connect(self.close_all_graphs)
        btn_layout.addWidget(self.close_all_graphs_btn)
        btn_layout.addStretch()
        input_layout.addLayout(btn_layout, 6, 0, 1, 3)
        layout.addWidget(input_group)
        results_heading = QLabel("<b>Ping Results</b>")
        results_heading.setStyleSheet("font-size: 15px; margin: 8px 0 4px 0;")
        layout.addWidget(results_heading)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels([
            "IP Address", "Hostname", "Current RTT (ms)",
            "Min RTT (ms)", "Max/Avg RTT (ms)", "View Graph"
        ])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setToolTip("Ping results for all monitored IPs")
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self.show_context_menu)
        # Highlight row on hover
        self.results_table.setStyleSheet(self.results_table.styleSheet() + "\nQTableWidget::item:hover { background: #4444aa; color: #fff; }")
        layout.addWidget(self.results_table, 1)
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open IP List File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.file_label.setText(os.path.basename(file_path))
            try:
                with open(file_path, 'r') as file:
                    ips = [line.strip() for line in file if line.strip()]
                    self.ip_input.setText(", ".join(ips))
            except Exception as e:
                QMessageBox.warning(self, "File Error", f"Error reading file: {e}")
                self.file_label.setText("No file selected")
    def set_ping_interval(self, value):
        self.ping_interval = value
        self.statusBar().showMessage(f"Ping interval set to {value} seconds.")
    def set_warning_threshold(self, value):
        self.warning_threshold = value
        self.statusBar().showMessage(f"Warning threshold set to {value} ms.")
    def validate_ips(self):
        valid_ips = []
        manual_text = self.ip_input.text().strip()
        if manual_text:
            for part in manual_text.replace(',', ' ').split():
                ip = part.strip()
                if ip:
                    try:
                        socket.inet_aton(ip)
                        valid_ips.append(ip)
                    except Exception:
                        pass
        base = f"{self.ip1.value()}.{self.ip2.value()}.{self.ip3.value()}"
        start = self.ip4_start.value()
        end = self.ip4_end.value()
        if start > end:
            QMessageBox.warning(self, "Invalid Range", "Start of range must be less than or equal to end.")
            return []
        for i in range(start, end+1):
            ip = f"{base}.{i}"
            try:
                socket.inet_aton(ip)
                valid_ips.append(ip)
            except Exception:
                pass
        valid_ips = list(dict.fromkeys(valid_ips))
        return valid_ips
    def start_monitoring(self):
        # If resuming, use existing IPs and data
        if not self.is_monitoring and self.ping_data:
            # Resume existing monitoring
            if self.batch_ping_worker:
                self.batch_ping_worker.resume()
            
            # Resume all graph windows
            for ip, win in self.graph_windows.items():
                if win:
                    win.resume_updates()
            
            # Get existing IPs for status message
            ips = list(self.ping_data.keys())
        else:
            # Start new monitoring
            ips = self.validate_ips()
            if not ips:
                QMessageBox.warning(self, "No IPs", "Please enter valid IP addresses.")
                return
            self.results_table.setRowCount(0)
            self.ping_data.clear()
            # Clear all loss regions from previous runs
            for ip in self.graph_windows:
                if ip in self.graph_windows and self.graph_windows[ip] is not None:
                    self.graph_windows[ip].graph_widget.all_loss_regions = []

            for ip in ips:
                row_position = self.results_table.rowCount()
                self.results_table.insertRow(row_position)
                ip_item = QTableWidgetItem(ip)
                self.results_table.setItem(row_position, 0, ip_item)
                self.results_table.setItem(row_position, 1, QTableWidgetItem(""))
                for col in range(2, 5):
                    self.results_table.setItem(row_position, col, QTableWidgetItem("-"))
                btn = QPushButton("View Graph")
                btn.setToolTip(f"Show detailed graph for {ip}")
                btn.setEnabled(False)
                btn.clicked.connect(lambda checked, ip=ip: self.open_graph_window(ip))
                self.results_table.setCellWidget(row_position, 5, btn)
                self.ping_data[ip] = {
                    'status': 'unknown',
                    'current_rtt': 0,
                    'min_rtt': 9999,
                    'max_rtt': 0,
                    'avg_rtt': 0,
                    'total_rtt': 0,
                    'count': 0
                }
                self.resolve_hostname(ip)
            
            # Create single batch worker for all IPs with optimized settings
            batch_size = min(30, max(10, len(ips) // 3))  # Larger adaptive batch size
            self.batch_ping_worker = BatchPingWorker(ips, self.ping_interval, batch_size)
            self.batch_ping_worker.ping_results.connect(self.update_batch_ping_results)
            self.batch_ping_worker.start()

        # Auto-resize columns
        self.results_table.resizeColumnsToContents()
        self.results_table.resizeRowsToContents()
        
        # Scroll to first alive host if any
        for row in range(self.results_table.rowCount()):
            ip = self.results_table.item(row, 0).text()
            if ip in self.ping_data and self.ping_data[ip]['status'] == 'alive':
                self.results_table.scrollToItem(self.results_table.item(row, 0))
                break

        self.toggle_monitor_btn.setEnabled(True)
        self.statusBar().showMessage(f"Monitoring {len(ips)} IP addresses...")
        # Apply the initial filter (Alive)
        self.apply_filter(self.filter_combo.currentText())
        self.is_monitoring = True
        self.toggle_monitor_btn.setText("Pause Monitoring")
        
        # Disable input fields while monitoring
        self.ip1.setEnabled(False)
        self.ip2.setEnabled(False)
        self.ip3.setEnabled(False)
        self.ip4_start.setEnabled(False)
        self.ip4_end.setEnabled(False)
        self.ip_input.setEnabled(False)
        self.file_btn.setEnabled(False)
        self.interval_spinner.setEnabled(False)
        self.threshold_spinner.setEnabled(False)
    def resolve_hostname(self, ip):
        resolver = HostnameResolver(ip)
        resolver.hostname_resolved.connect(self.update_hostname)
        resolver.start()
        self.hostname_resolvers[ip] = resolver
    def update_hostname(self, ip, hostname):
        for row in range(self.results_table.rowCount()):
            if self.results_table.item(row, 0).text() == ip:
                self.results_table.setItem(row, 1, QTableWidgetItem(hostname))
                break
    def stop_monitoring(self):
        if self.is_stopping:  # Prevent multiple stop attempts
            return
            
        self.is_stopping = True
        self.toggle_monitor_btn.setEnabled(False)  # Disable button while stopping
        self.statusBar().showMessage("Pausing monitoring...")
        
        # Pause batch worker
        if self.batch_ping_worker and self.batch_ping_worker.isRunning():
            self.batch_ping_worker.pause()
        
        # Pause all graph windows
        for ip, win in self.graph_windows.items():
            if win:
                win.pause_updates()
        
        # Update UI state
        self.toggle_monitor_btn.setEnabled(True)
        self.statusBar().showMessage("Monitoring paused")
        self.is_monitoring = False
        self.toggle_monitor_btn.setText("Resume Monitoring")
        self.is_stopping = False
        
        # Enable input fields
        self.ip1.setEnabled(True)
        self.ip2.setEnabled(True)
        self.ip3.setEnabled(True)
        self.ip4_start.setEnabled(True)
        self.ip4_end.setEnabled(True)
        self.ip_input.setEnabled(True)
        self.file_btn.setEnabled(True)
        self.interval_spinner.setEnabled(True)
        self.threshold_spinner.setEnabled(True)

    def check_stop_status(self):
        # Check if batch worker has stopped
        if self.batch_ping_worker and self.batch_ping_worker.isRunning():
            return
        
        # Worker has stopped, clean up
        self.stop_check_timer.stop()
        self.is_stopping = False
        
        # Don't close graph windows during pause
        # Just update UI state
        self.toggle_monitor_btn.setEnabled(True)
        self.statusBar().showMessage("Monitoring paused")
        self.is_monitoring = False
        self.toggle_monitor_btn.setText("Resume Monitoring")
        
        # Enable input fields
        self.ip1.setEnabled(True)
        self.ip2.setEnabled(True)
        self.ip3.setEnabled(True)
        self.ip4_start.setEnabled(True)
        self.ip4_end.setEnabled(True)
        self.ip_input.setEnabled(True)
        self.file_btn.setEnabled(True)
        self.interval_spinner.setEnabled(True)
        self.threshold_spinner.setEnabled(True)

    def clear_monitoring(self):
        # First stop all monitoring
        self.stop_monitoring()
        
        # Close all graph windows
        for ip, win in list(self.graph_windows.items()):
            if win:
                win.close()
            del self.graph_windows[ip]
        
        # Clear the results table
        self.results_table.setRowCount(0)
        self.ping_data.clear()
        self.batch_ping_worker = None
        
        # Clear input fields and reset selections
        self.ip1.setValue(192)
        self.ip2.setValue(168)
        self.ip3.setValue(1)
        self.ip4_start.setValue(1)
        self.ip4_end.setValue(254)
        self.ip_input.clear()
        self.file_label.setText("No file selected")
        self.filter_combo.setCurrentIndex(1)  # Default to Alive
        self.timeframe_combo.setCurrentIndex(3)  # Default to 5 min
        
        self.statusBar().showMessage("Cleared all monitoring data and inputs")

    def update_batch_ping_results(self, results):
        # Process all ping results at once for better performance
        updated_rows = set()  # Track which rows need updating
        
        for ip, result in results.items():
            if ip in self.ping_data:
                row = self._update_single_ping_result(ip, result)
                if row is not None:
                    updated_rows.add(row)
        
        # Only apply filter if there were updates
        if updated_rows:
            self.apply_filter(self.filter_combo.currentText())
    
    def _update_single_ping_result(self, ip, result):
        for row in range(self.results_table.rowCount()):
            if self.results_table.item(row, 0).text() == ip:
                ip_item = self.results_table.item(row, 0)
                
                # Update background color efficiently
                if result['status'] == 'alive':
                    color = QColor(255, 200, 0) if result['rtt'] > self.warning_threshold else QColor(0, 200, 0)
                else:
                    color = QColor(255, 0, 0)
                ip_item.setBackground(color)
                
                if result['status'] == 'alive':
                    rtt = result['rtt']
                    self.ping_data[ip]['status'] = 'alive'
                    self.ping_data[ip]['current_rtt'] = rtt
                    self.ping_data[ip]['min_rtt'] = min(self.ping_data[ip]['min_rtt'], rtt)
                    self.ping_data[ip]['max_rtt'] = max(self.ping_data[ip]['max_rtt'], rtt)
                    self.ping_data[ip]['total_rtt'] += rtt
                    self.ping_data[ip]['count'] += 1
                    self.ping_data[ip]['avg_rtt'] = self.ping_data[ip]['total_rtt'] / self.ping_data[ip]['count']
                    
                    # Update table items efficiently
                    self.results_table.setItem(row, 2, QTableWidgetItem(f"{rtt:.1f}"))
                    self.results_table.setItem(row, 3, QTableWidgetItem(f"{self.ping_data[ip]['min_rtt']:.1f}"))
                    self.results_table.setItem(row, 4, QTableWidgetItem(
                        f"{self.ping_data[ip]['max_rtt']:.1f} / {self.ping_data[ip]['avg_rtt']:.1f}"
                    ))
                    
                    btn = self.results_table.cellWidget(row, 5)
                    if btn:
                        btn.setEnabled(True)
                else:
                    self.ping_data[ip]['status'] = 'dead'
                    for col in range(2, 5):
                        self.results_table.setItem(row, col, QTableWidgetItem("-"))
                    btn = self.results_table.cellWidget(row, 5)
                    if btn:
                        btn.setEnabled(False)
                
                # Add tooltip to row
                tooltip = f"IP: {ip}\nStatus: {self.ping_data[ip]['status']}\nCurrent RTT: {self.ping_data[ip]['current_rtt']} ms"
                for col in range(self.results_table.columnCount()):
                    item = self.results_table.item(row, col)
                    if item:
                        item.setToolTip(tooltip)
                
                return row  # Return row number for tracking updates
        return None
    def apply_filter(self, filter_text):
        for row in range(self.results_table.rowCount()):
            ip = self.results_table.item(row, 0).text()
            if ip in self.ping_data:
                status = self.ping_data[ip]['status']
                if filter_text == "All" or \
                   (filter_text == "Alive" and status == 'alive') or \
                   (filter_text == "Dead" and status == 'dead'):
                    self.results_table.setRowHidden(row, False)
                else:
                    self.results_table.setRowHidden(row, True)
        # Ensure columns are resized after filtering
        self.results_table.resizeColumnsToContents()
        self.results_table.resizeRowsToContents()
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        for win in self.graph_windows.values():
            if hasattr(win, 'set_theme'):
                print(f"[NetworkPingMonitor] toggle_theme: updating graph window {win.windowTitle()} to dark_mode={self.dark_mode}")
                win.set_theme(self.dark_mode)
            if hasattr(win, 'set_time_scale'):
                win.set_time_scale(self.graph_timeframe_minutes * 60)
    def update_ui(self):
        QApplication.processEvents()
    def open_graph_window(self, ip):
        # Limit number of open graph windows to prevent resource exhaustion
        if len(self.graph_windows) >= 20:  # Maximum 20 graph windows
            QMessageBox.warning(self, "Too Many Windows", "Maximum 20 graph windows allowed. Please close some before opening new ones.")
            return
            
        if ip in self.graph_windows and self.graph_windows[ip] is not None:
            self.graph_windows[ip].raise_()
            self.graph_windows[ip].activateWindow()
            return
        win = PingGraphWindow(ip=ip, dark_mode=self.dark_mode, timeframe=self.graph_timeframe_minutes * 60)
        self.graph_windows[ip] = win
        win.show()
        def update_graph():
            if ip in self.ping_data:
                result = {
                    'status': self.ping_data[ip]['status'],
                    'rtt': self.ping_data[ip]['current_rtt']
                }
                win.update_data(result)
        # Create timer with proper cleanup
        timer = QTimer()
        timer.timeout.connect(update_graph)
        timer.start(3000)  # Update each graph window every 3 seconds for better performance
        win._update_graph_timer = timer  # Store reference to prevent garbage collection
        
        # Ensure timer is stopped when window is destroyed
        win.destroyed.connect(timer.stop)
        # Use a proper method to handle window destruction
        win.destroyed.connect(lambda checked, ip=ip: self._remove_graph_window(ip))
    def _remove_graph_window(self, ip):
        """Safely remove graph window from tracking"""
        if ip in self.graph_windows:
            del self.graph_windows[ip]
    
    def update_graph_timeframe(self, text):
        # Map the combo text to seconds
        if "sec" in text:
            seconds = int(text.split()[0])
        else:
            minutes = int(text.split()[0])
            seconds = minutes * 60
        self.graph_timeframe_minutes = seconds // 60 if seconds >= 60 else 0
        for win in self.graph_windows.values():
            if hasattr(win, 'set_time_scale'):
                win.set_time_scale(seconds)
        self.statusBar().showMessage(f"Graph timeframe updated to {text}")
    def show_context_menu(self, position):
        selected_items = self.results_table.selectedItems()
        if selected_items:
            menu = QMenu()
            trace_action = menu.addAction("Run Traceroute")
            graph_action = menu.addAction("Show Graph")
            action = menu.exec_(self.results_table.viewport().mapToGlobal(position))
            if action == trace_action:
                ip = self.results_table.item(self.results_table.currentRow(), 0).text()
                # If traceroute tab is implemented, set the IP and switch tab
                # self.trace_ip_input.setText(ip)
                # self.tabs.setCurrentWidget(self.traceroute_tab)
                # self.start_traceroute()
                QMessageBox.information(self, "Traceroute", f"Traceroute to {ip} not yet implemented.")
            elif action == graph_action:
                ip = self.results_table.item(self.results_table.currentRow(), 0).text()
                self.open_graph_window(ip)
    def open_all_graphs(self):
        # Open graph windows for all currently visible IPs
        for row in range(self.results_table.rowCount()):
            if not self.results_table.isRowHidden(row):
                ip = self.results_table.item(row, 0).text()
                self.open_graph_window(ip)
    def close_all_graphs(self):
        # Close all opened graph windows
        for ip, win in list(self.graph_windows.items()):
            if win: # Check if window still exists
                win.close()
        self.graph_windows.clear()
    def closeEvent(self, event):
        # Ensure all threads are stopped on app close
        print("[NetworkPingMonitor] closeEvent: Stopping all threads...")
        
        # Stop all timers first
        self.update_timer.stop()
        self.stop_check_timer.stop()
        
        # Stop Batch Ping Worker with immediate stop
        if self.batch_ping_worker and self.batch_ping_worker.isRunning():
            self.batch_ping_worker.stop()
            self.batch_ping_worker.wait(1000)  # Wait up to 1 second for clean shutdown
        
        # Stop Traceroute Worker
        if hasattr(self, 'traceroute_worker') and self.traceroute_worker and self.traceroute_worker.isRunning():
            self.traceroute_worker.stop()
        
        # Stop all hop ping workers for traceroute
        if hasattr(self, 'hop_ping_workers'):
            for ip, worker in list(self.hop_ping_workers.items()):
                if worker and worker.isRunning():
                    worker.stop()
        
        # Stop all graph window timers
        for ip, win in list(self.graph_windows.items()):
            if win and hasattr(win, '_update_graph_timer') and win._update_graph_timer:
                win._update_graph_timer.stop()
        
        # Wait a short time for threads to stop
        QTimer.singleShot(500, lambda: self.finalize_close(event))

    def finalize_close(self, event):
        # Final cleanup after threads have had time to stop
        self.batch_ping_worker = None
        if hasattr(self, 'traceroute_worker'):
            self.traceroute_worker = None
        if hasattr(self, 'hop_ping_workers'):
            self.hop_ping_workers.clear()
        
        print("[NetworkPingMonitor] closeEvent: All threads stopped. Accepting close event.")
        event.accept()

    def setup_traceroute_tab(self):
        layout = QVBoxLayout()
        self.traceroute_tab.setLayout(layout)
        heading = QLabel("<b>Traceroute Analysis</b>")
        heading.setStyleSheet("font-size: 16px; margin-bottom: 8px;")
        layout.addWidget(heading)
        input_group = QGroupBox("Traceroute Input")
        input_layout = QGridLayout()
        input_group.setLayout(input_layout)
        input_layout.addWidget(QLabel("Target IP address:"), 0, 0)
        self.trace_ip_input = QLineEdit(); self.trace_ip_input.setToolTip("Enter the IP to trace route to")
        self.trace_ip_input.setPlaceholderText("e.g. 8.8.8.8")
        input_layout.addWidget(self.trace_ip_input, 0, 1)
        self.trace_start_btn = QPushButton("Start Traceroute"); self.trace_start_btn.setToolTip("Begin traceroute to the target IP")
        self.trace_start_btn.clicked.connect(self.start_traceroute)
        input_layout.addWidget(self.trace_start_btn, 0, 2)
        self.trace_stop_btn = QPushButton("Stop Traceroute"); self.trace_stop_btn.setToolTip("Stop the current traceroute")
        self.trace_stop_btn.clicked.connect(self.stop_traceroute)
        self.trace_stop_btn.setEnabled(False)
        input_layout.addWidget(self.trace_stop_btn, 0, 3)
        layout.addWidget(input_group)
        self.traceroute_progress = QProgressBar()
        self.traceroute_progress.setRange(0, 30)
        self.traceroute_progress.setValue(0)
        self.traceroute_progress.setTextVisible(True)
        self.traceroute_progress.setFormat("Hops: %v")
        layout.addWidget(self.traceroute_progress)
        # Updated table structure for continuous monitoring
        self.traceroute_table = QTableWidget(0, 9)
        self.traceroute_table.setHorizontalHeaderLabels([
            "Hop", "Count", "IP Address", "Hostname", "Avg (ms)", "Min (ms)", "Cur (ms)", "PL%", "Latency Graph"
        ])
        header = self.traceroute_table.horizontalHeader()
        # Set stretch for most columns, fixed/content for others
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents) # Hop
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents) # Count
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents) # IP Address
        header.setSectionResizeMode(3, QHeaderView.Stretch) # Hostname
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents) # Avg
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents) # Min
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents) # Cur
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents) # PL%
        header.setSectionResizeMode(8, QHeaderView.Stretch) # Latency Graph
        self.traceroute_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.traceroute_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.traceroute_table.customContextMenuRequested.connect(self.show_hop_context_menu)
        # Set custom delegate for status icons in the first column
        self.status_icon_delegate = StatusIconDelegate()
        self.traceroute_table.setItemDelegateForColumn(0, self.status_icon_delegate)
        # Custom painting for status icons - COMMENTED OUT
        # self.traceroute_table.viewport().paintEvent = self.paint_traceroute_viewport
        layout.addWidget(self.traceroute_table)

    def start_traceroute(self):
        target_ip = self.trace_ip_input.text().strip()
        if not target_ip:
            QMessageBox.warning(self, "No IP", "Please enter an IP address to trace.")
            return
        self.traceroute_table.setRowCount(0)
        self.stop_traceroute() # Stop any existing traceroute and ping workers
        # Clear any existing hop ping workers
        if hasattr(self, 'hop_ping_workers'):
            for worker in self.hop_ping_workers.values():
                worker.stop()
            self.hop_ping_workers.clear()
        self.hop_ping_workers = {}
        self.hop_ping_data = {}
        self.hop_graph_widgets = {}

        self.traceroute_worker = TracerouteWorker(target_ip)
        self.traceroute_worker.trace_hop.connect(self.add_trace_hop)
        self.traceroute_worker.trace_complete.connect(self.trace_finished)
        self.traceroute_worker.start()
        self.trace_start_btn.setEnabled(False)
        self.trace_stop_btn.setEnabled(True)
        self.statusBar().showMessage(f"Running traceroute to {target_ip}...")

    def add_trace_hop(self, hop, ip, rtt_str):
        # RTT string from traceroute worker might contain multiple values
        # For the initial display, we might just show the first RTT or an average if available
        row = self.traceroute_table.rowCount()
        self.traceroute_table.insertRow(row)
        # Hop #
        # Status Icon and Hop #
        hop_item = QTableWidgetItem(str(hop))
        # Set initial status color (e.g., grey for pending or unknown)
        initial_status_color = QColor(150, 150, 150) # Grey
        hop_item.setBackground(initial_status_color)
        hop_item.setToolTip(f"Hop {hop} - {ip} (Pending)")
        self.traceroute_table.setItem(row, 0, hop_item)
        # Count (Placeholder)
        self.traceroute_table.setItem(row, 1, QTableWidgetItem("0"))
        # IP Address
        ip_item = QTableWidgetItem(ip)
        self.traceroute_table.setItem(row, 2, ip_item)
        # Hostname (Placeholder, will be resolved)
        self.traceroute_table.setItem(row, 3, QTableWidgetItem(""))
        # Avg, Min, Cur RTT (Placeholders)
        self.traceroute_table.setItem(row, 4, QTableWidgetItem("-")) # Avg
        self.traceroute_table.setItem(row, 5, QTableWidgetItem("-")) # Min
        self.traceroute_table.setItem(row, 6, QTableWidgetItem("-")) # Cur
        # PL% (Placeholder)
        self.traceroute_table.setItem(row, 7, QTableWidgetItem("-"))
        # Latency Graph (Placeholder or initial simple representation)
        # We'll add the actual graph widget later once pinging starts

        # Create and configure HopLatencyWidget for the table cell
        hop_graph_widget = HopLatencyWidget(dark_mode=self.dark_mode)
        self.traceroute_table.setCellWidget(row, 8, hop_graph_widget)
        self.hop_graph_widgets[ip] = hop_graph_widget

        # Start hostname resolution
        if ip and ip != "Request timed out" and ip.count('.') == 3:
            self.resolve_traceroute_hostname(ip, row)

            # Set up continuous pinging and graph for this hop
            if ip not in self.hop_ping_workers:
                # Initialize data structure for this hop
                self.hop_ping_data[ip] = {
                    'status': 'unknown',
                    'current_rtt': 0,
                    'min_rtt': 9999,
                    'max_rtt': 0,
                    'avg_rtt': 0,
                    'total_rtt': 0,
                    'count': 0,
                    'lost_count': 0,
                    'history': [] # Store (timestamp, rtt, status) for the hop graph
                }

                # Start PingWorker for this hop
                worker = BatchPingWorker([ip], self.ping_interval, batch_size=1) # Use the main ping interval for now
                worker.ping_results.connect(lambda results: self.update_hop_ping_result(ip, results.get(ip, {'status': 'dead', 'rtt': 0})))
                worker.start()
                self.hop_ping_workers[ip] = worker

    def update_hop_ping_result(self, ip, result):
        # Find the row for this IP in the traceroute table
        row = -1
        for i in range(self.traceroute_table.rowCount()):
            if self.traceroute_table.item(i, 2).text() == ip:
                row = i
                break
        if row == -1:
            return # IP not found in table (shouldn't happen if logic is correct)

        # Update ping data for the hop
        if ip in self.hop_ping_data:
            hop_data = self.hop_ping_data[ip]
            hop_data['count'] += 1
            status = result['status']
            rtt = result['rtt']

            if status == 'alive':
                hop_data['status'] = 'alive'
                hop_data['current_rtt'] = rtt
                hop_data['min_rtt'] = min(hop_data['min_rtt'], rtt) if hop_data['min_rtt'] != 9999 else rtt
                hop_data['max_rtt'] = max(hop_data['max_rtt'], rtt)
                hop_data['total_rtt'] += rtt
                hop_data['avg_rtt'] = hop_data['total_rtt'] / (hop_data['count'] - hop_data['lost_count']) if (hop_data['count'] - hop_data['lost_count']) > 0 else 0
            else:
                hop_data['status'] = 'dead'
                hop_data['current_rtt'] = 0
                hop_data['lost_count'] += 1

            # Update table cells
            self.traceroute_table.setItem(row, 1, QTableWidgetItem(str(hop_data['count'])))
            self.traceroute_table.setItem(row, 4, QTableWidgetItem(f"{hop_data['avg_rtt']:.1f}" if hop_data['avg_rtt'] > 0 else "-"))
            self.traceroute_table.setItem(row, 5, QTableWidgetItem(f"{hop_data['min_rtt']:.1f}" if hop_data['min_rtt'] != 9999 else "-"))
            self.traceroute_table.setItem(row, 6, QTableWidgetItem(f"{hop_data['current_rtt']:.1f}" if hop_data['current_rtt'] > 0 else "-"))
            # Calculate and update Packet Loss Percentage
            total_sent = hop_data['count']
            lost_count = hop_data['lost_count']
            pl_percent = (lost_count / total_sent) * 100 if total_sent > 0 else 0
            self.traceroute_table.setItem(row, 7, QTableWidgetItem(f"{pl_percent:.1f}%"))

            # Add data to the hop graph widget history
            if ip in self.hop_graph_widgets:
                hop_graph_widget = self.hop_graph_widgets[ip]
                hop_graph_widget.update_data({
                    'status': status,
                    'rtt': rtt,
                    'min_rtt': hop_data['min_rtt'],
                    'max_rtt': hop_data['max_rtt'],
                    'avg_rtt': hop_data['avg_rtt'] # Although avg is not explicitly shown in the image bar, it might be useful
                })

            # Update row color based on status/latency threshold
            if status == 'alive':
                if rtt > self.warning_threshold:
                     for col in range(self.traceroute_table.columnCount()):
                         item = self.traceroute_table.item(row, col)
                         if item:
                             item.setBackground(QColor(255, 200, 0))
                # Update status icon color
                status_item = self.traceroute_table.item(row, 0)
                if status_item:
                     status_item.setBackground(QColor(255, 200, 0))
                     status_item.setToolTip(f"Hop {status_item.text()} - {ip} (High Latency: {rtt:.1f} ms)")
                else:
                    status_item = self.traceroute_table.item(row, 0)
                    if status_item:
                        status_item.setBackground(QColor(255, 200, 0))
                        status_item.setToolTip(f"Hop {status_item.text()} - {ip} (High Latency: {rtt:.1f} ms)")
            else:
                for col in range(self.traceroute_table.columnCount()):
                    item = self.traceroute_table.item(row, col)
                    if item:
                        item.setBackground(QColor(255, 0, 0))
                # Update status icon color
                status_item = self.traceroute_table.item(row, 0)
                if status_item:
                    status_item.setBackground(QColor(0, 0, 0, 0)) # Transparent
                    status_item.setToolTip(f"Hop {status_item.text()} - {ip} (Dead)")

    def resolve_traceroute_hostname(self, ip, row):
        resolver = HostnameResolver(ip)
        resolver.hostname_resolved.connect(lambda ip, hostname: self.update_traceroute_hostname(row, hostname))
        resolver.start()

    def update_traceroute_hostname(self, row, hostname):
        self.traceroute_table.setItem(row, 2, QTableWidgetItem(hostname))

    def trace_finished(self):
        self.trace_start_btn.setEnabled(True)
        self.trace_stop_btn.setEnabled(False)
        self.statusBar().showMessage("Traceroute completed")

    def stop_traceroute(self):
        if hasattr(self, 'traceroute_worker') and self.traceroute_worker and self.traceroute_worker.isRunning():
            self.traceroute_worker.stop()
            # self.traceroute_worker.wait() # Removed to prevent GUI from hanging
            self.traceroute_worker = None
            self.trace_start_btn.setEnabled(True)
            self.trace_stop_btn.setEnabled(False)
            self.statusBar().showMessage("Traceroute stopped")

            # Stop all hop ping workers
            if hasattr(self, 'hop_ping_workers'):
                for worker in self.hop_ping_workers.values():
                    worker.stop()
                self.hop_ping_workers.clear()
            # Clear hop data and graph widgets
            if hasattr(self, 'hop_ping_data'):
                self.hop_ping_data.clear()
            if hasattr(self, 'hop_graph_widgets'):
                self.hop_graph_widgets.clear()

    def show_hop_context_menu(self, position):
        selected_items = self.traceroute_table.selectedItems()
        if selected_items:
            row = self.traceroute_table.currentRow()
            ip = self.traceroute_table.item(row, 1).text()
            if ip and ip.count('.') == 3:
                menu = QMenu()
                ping_action = menu.addAction("Start Monitoring Hop")
                action = menu.exec_(self.traceroute_table.viewport().mapToGlobal(position))
                if action == ping_action:
                    if ip not in self.ping_data:
                        self.ip_input.setText(ip)
                        self.tabs.setCurrentWidget(self.ping_tab)
                        self.start_monitoring()

    def toggle_monitoring(self):
        if self.is_monitoring:
            self.stop_monitoring()
            # stop_monitoring already updates is_monitoring and button text
        else:
            self.start_monitoring()
            # start_monitoring already updates is_monitoring and button text

# --- Traceroute Worker ---
class TracerouteWorker(QThread):
    trace_hop = pyqtSignal(int, str, str)
    trace_complete = pyqtSignal()
    def __init__(self, ip):
        super().__init__()
        self.ip = ip
        self.running = True
        self.process = None # Store the subprocess object
    def run(self):
        try:
            if sys.platform == 'win32':
                cmd = ['tracert', '-d', '-w', '500', self.ip]
                self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
                for line in iter(self.process.stdout.readline, ''):
                    if not self.running:
                        self.process.terminate()
                        break
                    line = line.strip()
                    if line and line[0].isdigit():
                        parts = line.split()
                        hop_num = parts[0]
                        try:
                            hop = int(hop_num)
                            if len(parts) >= 8 and parts[7].count('.') == 3:
                                hop_ip = parts[7]
                                hop_rtt = f"{parts[1]} ms {parts[3]} ms {parts[5]} ms"
                            elif "*" in line:
                                hop_ip = "Request timed out"
                                hop_rtt = "* * *"
                            else:
                                hop_ip = "Unknown"
                                hop_rtt = "* * *"
                            self.trace_hop.emit(hop, hop_ip, hop_rtt)
                        except ValueError:
                            pass
            else:
                cmd = ['traceroute', '-n', '-w', '1', self.ip]
                self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
                for line in iter(self.process.stdout.readline, ''):
                    if not self.running:
                        self.process.terminate()
                        break
                    line = line.strip()
                    if line and line[0].isdigit():
                        parts = line.split()
                        try:
                            hop = int(parts[0])
                            if len(parts) >= 4 and parts[1].count('.') == 3:
                                hop_ip = parts[1]
                                hop_rtt = f"{' '.join(parts[2:])}"
                            elif "*" in line:
                                hop_ip = "Request timed out"
                                hop_rtt = "* * *"
                            else:
                                hop_ip = "Unknown"
                                hop_rtt = "* * *"
                            self.trace_hop.emit(hop, hop_ip, hop_rtt)
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Error in traceroute to {self.ip}: {e}")
        self.trace_complete.emit()
    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.terminate()
            except OSError:
                pass # Process might already be terminated
        # self.wait() # Removed to prevent GUI from hanging

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NetworkPingMonitor()
    window.show()
    sys.exit(app.exec_()) 