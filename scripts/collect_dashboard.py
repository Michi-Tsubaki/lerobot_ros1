#!/usr/bin/env python3

import json
import sys

import rospy
from python_qt_binding.QtCore import QTimer
from python_qt_binding.QtGui import QColor
from python_qt_binding.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from std_msgs.msg import String


class StatusLight(QWidget):
    def __init__(self, label):
        super().__init__()
        self.dot = QLabel()
        self.dot.setFixedSize(16, 16)
        self.text = QLabel(label)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.dot)
        layout.addWidget(self.text)
        layout.addStretch()
        self.setLayout(layout)
        self.set_connected(False)

    def set_connected(self, connected):
        color = QColor("#16a34a" if connected else "#dc2626")
        self.dot.setStyleSheet(
            "border-radius: 8px; "
            f"background-color: {color.name()}; "
            "border: 1px solid rgba(0, 0, 0, 0.25);"
        )


class CollectDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("imitation_utils Collect Dashboard")
        self.command_pub = rospy.Publisher("/data_collector/command", String, queue_size=1)
        self.status_sub = rospy.Subscriber("/data_collector/status", String, self.status_cb)
        self.status = {}
        self.connection_widgets = {}

        self.phase_label = QLabel("-")
        self.episode_label = QLabel("-")
        self.frames_label = QLabel("-")
        self.saved_label = QLabel("-")
        self.task_label = QLabel("-")
        self.root_label = QLabel("-")
        self.message_label = QLabel("-")
        self.message_label.setWordWrap(True)

        self.robot_light = StatusLight("Robot")
        self.hand_light = StatusLight("Hands")
        self.images_light = StatusLight("Images")

        self.connection_box_layout = QVBoxLayout()
        connection_box = QGroupBox("Connections")
        connection_box.setLayout(self.connection_box_layout)

        top = QGridLayout()
        top.addWidget(QLabel("Phase"), 0, 0)
        top.addWidget(self.phase_label, 0, 1)
        top.addWidget(QLabel("Episode"), 1, 0)
        top.addWidget(self.episode_label, 1, 1)
        top.addWidget(QLabel("Frames"), 2, 0)
        top.addWidget(self.frames_label, 2, 1)
        top.addWidget(QLabel("Saved"), 3, 0)
        top.addWidget(self.saved_label, 3, 1)
        top.addWidget(QLabel("Task"), 4, 0)
        top.addWidget(self.task_label, 4, 1)
        top.addWidget(QLabel("Root"), 5, 0)
        top.addWidget(self.root_label, 5, 1)

        summary = QHBoxLayout()
        summary.addWidget(self.robot_light)
        summary.addWidget(self.hand_light)
        summary.addWidget(self.images_light)
        summary.addStretch()

        buttons = QHBoxLayout()
        for label, command in [
            ("Start", "start"),
            ("Finish", "finish"),
            ("Accept", "accept"),
            ("Reject", "reject"),
            ("Stop", "stop"),
        ]:
            button = QPushButton(label)
            button.clicked.connect(lambda _, cmd=command: self.send_command(cmd))
            buttons.addWidget(button)

        main = QVBoxLayout()
        main.addLayout(top)
        main.addLayout(summary)
        main.addWidget(connection_box)
        main.addLayout(buttons)
        main.addWidget(QLabel("Message"))
        main.addWidget(self.message_label)
        self.setLayout(main)
        self.resize(720, 420)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)

    def send_command(self, command):
        self.command_pub.publish(String(data=command))

    def status_cb(self, msg):
        try:
            self.status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.status = {"last_message": msg.data}

    def refresh(self):
        status = self.status
        self.phase_label.setText(str(status.get("phase", "-")))
        self.episode_label.setText(str(status.get("episode_index", "-")))
        self.frames_label.setText(str(status.get("frames", "-")))
        saved = status.get("saved_episodes", "-")
        target = status.get("target_episodes")
        self.saved_label.setText(f"{saved} / {target}" if target is not None else str(saved))
        self.task_label.setText(str(status.get("task_name", "-")))
        self.root_label.setText(str(status.get("root", "-")))
        self.message_label.setText(str(status.get("last_message", "-")))

        self.robot_light.set_connected(bool(status.get("robot_connected", False)))
        self.hand_light.set_connected(bool(status.get("hand_connected", False)))
        self.images_light.set_connected(bool(status.get("images_connected", False)))
        self.refresh_connections(status.get("connections", {}))

    def refresh_connections(self, connections):
        for name, item in sorted(connections.items()):
            widget = self.connection_widgets.get(name)
            if widget is None:
                widget = StatusLight(f"{name} ({item.get('topic', '')})")
                self.connection_widgets[name] = widget
                self.connection_box_layout.addWidget(widget)
            widget.set_connected(bool(item.get("connected", False)))


if __name__ == "__main__":
    rospy.init_node("collect_dashboard", anonymous=True)
    app = QApplication(sys.argv)
    dashboard = CollectDashboard()
    dashboard.show()
    sys.exit(app.exec_())
