import os
import sqlite3
from datetime import datetime, timedelta
import numpy as np
from io import BytesIO
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGridLayout, QWidget, 
                           QLabel, QScrollArea, QVBoxLayout, QHBoxLayout, 
                           QComboBox, QPushButton)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

db_path = os.path.expanduser("~/.config/snooper.db")

class ImageGridWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(20)
        self.images = []
        
    def add_image(self, image_path, category):
        self.images.append((image_path, category))
        
    def clear(self):
        self.images = []
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)
            
    def create_image_widget(self, image_path, category):
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setAlignment(Qt.AlignCenter)
        
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)
        image_label.setMinimumSize(QSize(400, 300))
        
        category_label = QLabel(category)
        category_label.setAlignment(Qt.AlignCenter)
        category_label.setStyleSheet("""
            font-weight: bold;
            color: #333333;
            font-size: 14px;
            padding: 5px;
        """)
        
        container_layout.addWidget(image_label)
        container_layout.addWidget(category_label)
        container.setLayout(container_layout)
        return container

    def relayout(self, width):
        # Remove all widgets from layout
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)
            
        # Calculate number of columns based on width
        # Each item is 400px wide plus spacing
        item_width = 400 + self.layout.spacing()
        num_columns = max(1, width // item_width)
        
        # Add widgets back in new arrangement
        for idx, (image_path, category) in enumerate(self.images):
            row = idx // num_columns
            col = idx % num_columns
            container = self.create_image_widget(image_path, category)
            self.layout.addWidget(container, row, col)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Activity Browser")
        self.setGeometry(100, 100, 1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.graphs = ActivityGraphs()
        self.layout.addWidget(self.graphs)
        
        separator = QWidget()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #cccccc;")
        self.layout.addWidget(separator)

        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_area.setWidget(self.scroll_content)
        
        self.image_grid = ImageGridWidget(self.scroll_content)
        self.scroll_content.setLayout(self.image_grid.layout)
        self.layout.addWidget(self.scroll_area)
        
        # Load activities immediately
        self.load_activities()
        
        # Setup resize timer for debouncing
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.handle_resize)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Reset the timer on each resize event
        self.resize_timer.start(150)  # 150ms debounce
        
    def handle_resize(self):
        # Calculate available width for grid
        scroll_width = self.scroll_area.viewport().width()
        self.image_grid.relayout(scroll_width)
    
    def load_activities(self):
        self.image_grid.clear()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, screenshot, category FROM activities")
        activities = cursor.fetchall()
        conn.close()
        
        for activity in activities:
            activity_id, screenshot_blob, category = activity
            category = category.split(" ")[0]
            image_path = self.blob_to_image(screenshot_blob, activity_id)
            if image_path:
                self.image_grid.add_image(image_path, category)
        
        # Initial layout
        self.handle_resize()
    
    def blob_to_image(self, blob, activity_id):
        try:
            image_data = BytesIO(blob)
            image = Image.open(image_data)
            image = image.convert("RGB")
            image_path = f"/tmp/screenshot_{activity_id}.png"
            image.save(image_path, quality=95)
            return image_path
        except Exception as e:
            print(f"Error processing image {activity_id}: {e}")
            return None

ACTIVITY_COLORS = {
    'Reading': '#FF9999',
    'Writing': '#66B2FF',
    'Coding': '#99FF99',
    'Watching': '#FFCC99',
    'Browsing': '#FF99CC',
    'Social': '#99CCFF',
    'Gaming': '#FF99FF',
    'Learning': '#FFFF99',
    'Working': '#99FFCC',
    'Listening': '#FFB366',
    'Designing': '#FF66B2',
    'Chatting': '#66FFB2',
    'Idle': '#B2B2B2',
    'Miscellaneous': '#E6E6E6'
}

class ActivityGraphs(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Add controls
        self.controls_layout = QHBoxLayout()
        
        # Time range selector
        self.time_range = QComboBox()
        self.time_range.addItems(['Last 24 Hours', 'Last Week', 'Last Month', 'All Time'])
        self.time_range.currentTextChanged.connect(self.update_graphs)
        
        # Refresh button
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.clicked.connect(self.update_graphs)
        
        self.controls_layout.addWidget(QLabel('Time Range:'))
        self.controls_layout.addWidget(self.time_range)
        self.controls_layout.addWidget(self.refresh_btn)
        self.controls_layout.addStretch()
        
        self.layout.addLayout(self.controls_layout)
        
        # Create matplotlib figures
        self.create_figures()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_graphs)
        self.update_timer.start(300000)  # Update every 5 minutes
        
        # Initial update
        self.update_graphs()

    def create_figures(self):
        # Create figure for both plots
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Create subplots
        self.pie_ax = self.figure.add_subplot(121)
        self.timeline_ax = self.figure.add_subplot(122)
        
        self.figure.set_facecolor('#f0f0f0')
        self.canvas.setStyleSheet("background-color: #f0f0f0;")

    def get_time_range_filter(self):
        current_range = self.time_range.currentText()
        now = datetime.now()
        
        if current_range == 'Last 24 Hours':
            start_time = now - timedelta(days=1)
        elif current_range == 'Last Week':
            start_time = now - timedelta(weeks=1)
        elif current_range == 'Last Month':
            start_time = now - timedelta(days=30)
        else:  # All Time
            return ""
            
        return f"WHERE timestamp >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'"

    def update_graphs(self):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        time_filter = self.get_time_range_filter()
        
        # Get activity distribution
        cursor.execute(f"""
            SELECT category, COUNT(*) as count 
            FROM activities 
            {time_filter}
            GROUP BY category
        """)
        distribution_data = cursor.fetchall()
        
        # Get timeline data
        cursor.execute(f"""
            SELECT timestamp, category 
            FROM activities 
            {time_filter}
            ORDER BY timestamp
        """)
        timeline_data = cursor.fetchall()
        
        conn.close()
        
        self.plot_distribution(distribution_data)
        self.plot_timeline(timeline_data)
        
        self.canvas.draw()

    def plot_distribution(self, data):
        self.pie_ax.clear()
        
        if not data:
            self.pie_ax.text(0.5, 0.5, 'No data available', 
                           ha='center', va='center')
            return
            
        labels = [item[0].split(' ')[0] for item in data]
        sizes = [item[1] for item in data]
        colors = [ACTIVITY_COLORS.get(label, '#CCCCCC') for label in labels]
        
        self.pie_ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90)
        self.pie_ax.axis('equal')
        self.pie_ax.set_title('Activity Distribution')

    def plot_timeline(self, data):
        self.timeline_ax.clear()
        
        if not data:
            self.timeline_ax.text(0.5, 0.5, 'No data available', 
                                ha='center', va='center')
            return
            
        timestamps = [datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S') 
                     for item in data]
        categories = [item[1].split(" ")[0] for item in data]
        
        unique_categories = list(set(categories))
        y_positions = range(len(unique_categories))
        
        for idx, category in enumerate(unique_categories):
            category_times = [t for t, c in zip(timestamps, categories) 
                            if c == category]
            self.timeline_ax.scatter(category_times, 
                                   [idx] * len(category_times),
                                   color=ACTIVITY_COLORS.get(category, '#CCCCCC'),
                                   alpha=0.6, s=50, label=category)
        
        self.timeline_ax.set_yticks(y_positions)
        self.timeline_ax.set_yticklabels(unique_categories)
        self.timeline_ax.set_title('Activity Timeline')
        
        plt.setp(self.timeline_ax.get_xticklabels(), rotation=45, 
                ha='right')
        
        self.timeline_ax.grid(True, alpha=0.3)
        self.figure.tight_layout()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
