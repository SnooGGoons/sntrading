import platform
import logging
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QApplication
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QObject
from PyQt6.QtGui import QFont
import subprocess
import os

try:
    import winsound
except ImportError:
    winsound = None

try:
    from AppKit import NSSound
except ImportError:
    NSSound = None

logger = logging.getLogger(__name__)

class NotificationManager(QObject):
    # Signal to ensure thread-safe popup display
    show_popup = pyqtSignal(str, str, int, str)
    show_regime_alert = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.active_notifications = []
        self.app = QApplication.instance()
        if not self.app:
            logger.error("No QApplication instance available")
            return

        # Connect signals to slots
        self.show_popup.connect(self._display_popup)
        self.show_regime_alert.connect(self._display_regime_alert)

    def send_popup(self, title: str, message: str, duration: int = 5000, signal_type: str = "NONE"):
        """Emit signal to display a popup notification in the main thread."""
        try:
            self.show_popup.emit(title, message, duration, signal_type)
            logger.info(f"Emitted popup signal: {title} - {message}")
        except Exception as e:
            logger.error(f"Failed to emit popup signal: {str(e)}")

    def _display_popup(self, title: str, message: str, duration: int, signal_type: str):
        """Display a popup notification in the main thread."""
        try:
            notification = PopupNotification(self.parent, title, message, duration, signal_type, self)
            self.active_notifications.append(notification)
            notification.show()
            QTimer.singleShot(duration + 500, lambda: self.remove_notification(notification))
            logger.info(f"Displayed popup notification: {title} - {message}")
            # Play sound in the main thread
            self.play_alert_sound(signal_type)
        except Exception as e:
            logger.error(f"Failed to display popup: {str(e)}")

    def send_regime_alert(self, regime: str):
        """Emit signal to notify the user of a market regime change."""
        try:
            if not regime or regime.lower() == "neutral":
                return
            self.show_regime_alert.emit(regime)
            logger.info(f"Emitted regime alert signal: {regime}")
        except Exception as e:
            logger.error(f"Failed to emit regime alert: {str(e)}")

    def _display_regime_alert(self, regime: str):
        """Display regime alert in the main thread."""
        try:
            title = "Market Regime Change"
            message = f"Market regime has changed to: {regime}"
            self._display_popup(title, message, 5000, "REGIME")
            logger.info(f"Displayed regime alert: {regime}")
        except Exception as e:
            logger.error(f"Failed to display regime alert: {str(e)}")

    def play_alert_sound(self, signal_type: str):
        """Play an alert sound based on the signal type (BUY, SELL, REGIME, etc.)."""
        try:
            system = platform.system()
            sound_file = None

            if signal_type == "BUY":
                sound_file = os.path.join("assets", "sounds", "buy_alert.wav")
            elif signal_type == "SELL":
                sound_file = os.path.join("assets", "sounds", "sell_alert.wav")
            elif signal_type == "REGIME":
                sound_file = os.path.join("assets", "sounds", "regime_alert.wav")
            else:
                return

            if not os.path.exists(sound_file):
                logger.warning(f"Sound file not found: {sound_file}")
                return

            if system == "Windows" and winsound:
                winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
            elif system == "Darwin" and NSSound:
                sound = NSSound.alloc().initWithContentsOfFile_byReference_(sound_file, True)
                if sound:
                    sound.play()
            elif system == "Linux":
                try:
                    subprocess.run(["canberra-gtk-play", "-f", sound_file], check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to play sound on Linux: {str(e)}")
            else:
                logger.warning(f"Sound playback not supported on {system}")
            logger.info(f"Played alert sound for {signal_type}: {sound_file}")
        except Exception as e:
            logger.error(f"Failed to play alert sound: {str(e)}")

    def remove_notification(self, notification):
        """Remove a notification from the active list and clean it up."""
        try:
            if notification in self.active_notifications:
                self.active_notifications.remove(notification)
                notification.deleteLater()
                logger.debug(f"Removed notification: {notification}")
        except Exception as e:
            logger.error(f"Failed to remove notification: {str(e)}")

class PopupNotification(QFrame):
    def __init__(self, parent, title: str, message: str, duration: int, signal_type: str, manager):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setObjectName("popupNotification")
        self.signal_type = signal_type
        self.manager = manager

        # Layout and styling
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #ffffff;")
        layout.addWidget(title_label)

        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("color: #dcdcdc;")
        message_label.setFont(QFont("Arial", 10))
        layout.addWidget(message_label)

        self.setStyleSheet("""
            QFrame#popupNotification {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #2c3e50, stop:1 #34495e);
                border: 1px solid #3498db;
                border-radius: 8px;
            }
            QFrame[signalType="BUY"] {
                border: 2px solid #2ecc71;
            }
            QFrame[signalType="SELL"] {
                border: 2px solid #e74c3c;
            }
            QFrame[signalType="REGIME"] {
                border: 2px solid #f1c40f;
            }
        """)
        self.setProperty("signalType", signal_type)

        # Position the notification
        try:
            screen = QApplication.instance().primaryScreen()
            if not screen:
                logger.error("No primary screen available")
                return
            screen_geometry = screen.availableGeometry()
            self.setFixedWidth(250)
            self.adjustSize()
            x = screen_geometry.width() - self.width() - 20
            y = screen_geometry.height() - self.height() - 20 - (60 * len(self.manager.active_notifications))
            self.move(x, y)
        except Exception as e:
            logger.error(f"Failed to position notification: {str(e)}")
            return

        # Start animations
        self.fade_in()
        QTimer.singleShot(duration, self.fade_out)

    def fade_in(self):
        """Animate the notification fading in."""
        try:
            self.setProperty("opacity", 0)
            self.anim = QPropertyAnimation(self, b"opacity")
            self.anim.setDuration(500)
            self.anim.setStartValue(0)
            self.anim.setEndValue(1)
            self.anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.anim.start()
        except Exception as e:
            logger.error(f"Failed to animate fade-in: {str(e)}")

    def fade_out(self):
        """Animate the notification fading out and close it."""
        try:
            self.anim = QPropertyAnimation(self, b"opacity")
            self.anim.setDuration(500)
            self.anim.setStartValue(1)
            self.anim.setEndValue(0)
            self.anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.anim.finished.connect(self.close)
            self.anim.start()
        except Exception as e:
            logger.error(f"Failed to animate fade-out: {str(e)}")

    def get_opacity(self):
        """Get the current opacity."""
        return self.property("opacity") or 1.0

    def set_opacity(self, value):
        """Set the opacity and update stylesheet."""
        self.setProperty("opacity", value)
        self.setStyleSheet(self.styleSheet())  # Trigger stylesheet update